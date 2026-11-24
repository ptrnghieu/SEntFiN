"""Command line entry point: ``python -m absa_data <command>``.

Commands:
  build         clean, split, attach aspect labels, export and profile the datasets
  annotate      label aspect categories with an LLM (OpenAI), resumable
  review-sheet  write a blind spreadsheet of SEntFiN test pairs for human aspect labels
  review-eval   compare filled review sheet with the LLM labels; store human labels as gold
  calibrate     fit FiQA score thresholds to SEntFiN's label semantics (bridge LLM)
  push-hub      publish data/processed to a Hugging Face dataset repo (gated if it holds FiQA)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from . import fiqa as fiqa_mod
from .entities import load_alias_table
from .schema import PAIR_COLUMNS
from .sentfin import load_sentfin
from .splits import (apply_manifest, assign_groups, check_no_leakage, connected_groups, mark_entity_seen,
                     save_manifest, stratified_group_split)

GUIDELINES_PATH = Path("reports/ASPECT_GUIDELINES.md")
QUALITY_REPORT_PATH = Path("reports/aspect_label_quality.md")
CALIBRATION_REPORT_PATH = Path("reports/sentiment_label_calibration.md")
REVIEW_SHEET_PATH = Path("annotations/review/sentfin_test_review_v1.xlsx")
HUMAN_LABELS_PATH = Path("annotations/human/sentfin_test_v1.csv")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_fiqa(args) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if args.no_fiqa:
        return None
    try:
        data_dir = Path(args.fiqa_dir) if args.fiqa_dir else fiqa_mod.download_fiqa(revision=args.fiqa_revision)
        return fiqa_mod.load_fiqa(data_dir)
    except Exception as e:  # auth / network / gate not accepted
        if args.require_fiqa:
            raise
        print(
            f"[warn] FiQA skipped ({type(e).__name__}: {str(e).splitlines()[0][:200]}).\n"
            f"       Accept the access form at https://huggingface.co/datasets/{fiqa_mod.FIQA_REPO}, then run\n"
            "       `huggingface-cli login` (or set HF_TOKEN), or pass --fiqa-dir with local parquet files.",
            file=sys.stderr,
        )
        return None


def _load_finentity(args, aliases):
    from . import finentity as fe_mod

    if getattr(args, "no_finentity", False):
        return None
    try:
        path = Path(args.finentity_path) if args.finentity_path else fe_mod.download_finentity()
        return fe_mod.load_finentity(path, aliases)
    except Exception as e:
        print(f"[warn] FinEntity skipped ({type(e).__name__}: {str(e).splitlines()[0][:200]})", file=sys.stderr)
        return None


def _split_texts(pairs: pd.DataFrame, texts: pd.DataFrame, manifest: Path, seed: int, resplit: bool) -> str:
    """Split a source whose texts may have no pairs (FinEntity): split pair rows by group, carry the
    split to every text of the group, freeze it in a manifest keyed by text_id."""
    from .text import dedup_key

    texts["dedup"] = texts["text"].map(dedup_key)
    texts["group_id"] = connected_groups(texts, ["paragraph_id", "dedup"])
    pairs["group_id"] = pairs["text_id"].map(texts.set_index("text_id")["group_id"])
    if manifest.exists() and not resplit:
        texts["split"] = apply_manifest(texts, manifest)
        note = f"frozen manifest {manifest}"
    else:
        by_group = pairs.assign(split=stratified_group_split(pairs, seed=seed)).groupby("group_id")["split"].first()
        texts["split"] = texts["group_id"].map(by_group).fillna("train")
        save_manifest(texts, manifest)
        note = f"new StratifiedGroupKFold split by paragraph (seed={seed}) -> {manifest}"
    pairs["split"] = pairs["text_id"].map(texts.set_index("text_id")["split"])
    texts.drop(columns="dedup", inplace=True)
    return note


def load_pairs(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load, clean and split all sources. Returns (pairs, texts, issues, meta)."""
    from .schema import TEXT_COLUMNS

    sentfin_csv = Path(args.sentfin_csv)
    aliases = load_alias_table(args.entity_list) if Path(args.entity_list).exists() else None

    sf, sf_issues = load_sentfin(sentfin_csv, aliases)
    sf["group_id"] = assign_groups(sf)
    manifest = Path(args.split_manifest)
    if manifest.exists() and not getattr(args, "resplit", False):
        sf["split"] = apply_manifest(sf, manifest)
        split_note = f"frozen manifest {manifest}"
    else:
        sf["split"] = stratified_group_split(sf, seed=args.seed)
        save_manifest(sf, manifest)
        split_note = f"new StratifiedGroupKFold split (seed={args.seed}) -> {manifest}"
    frames, issues, text_frames = [sf], [sf_issues], []

    fq = _load_fiqa(args)
    if fq is not None:
        fq_pairs, fq_issues = fq
        fq_pairs["group_id"] = assign_groups(fq_pairs)
        frames.append(fq_pairs)
        issues.append(fq_issues)

    fe = _load_finentity(args, aliases)
    fe_note = "not loaded"
    if fe is not None:
        fe_pairs, fe_texts, fe_issues = fe
        fe_note = _split_texts(fe_pairs, fe_texts, Path(args.finentity_split_manifest), args.seed,
                               getattr(args, "resplit", False))
        frames.append(fe_pairs)
        text_frames.append(fe_texts)
        issues.append(fe_issues)

    pairs = pd.concat(frames, ignore_index=True)
    # Provenance columns for sources whose loader does not set them (SEntFiN, FiQA: gold, partial targets).
    for col in ("paragraph_id", "target_annotation", "target_source", "sentiment_source", "target_type"):
        if col not in pairs:
            pairs[col] = None
    pairs["paragraph_id"] = pairs["paragraph_id"].fillna(pairs["text_id"])
    pairs["target_annotation"] = pairs["target_annotation"].fillna("partial")
    pairs["target_source"] = pairs["target_source"].fillna("gold")
    pairs["sentiment_source"] = pairs["sentiment_source"].fillna("gold")
    pairs["entity_seen_in_train"] = mark_entity_seen(pairs)

    from_pairs = pairs.drop_duplicates("text_id")[["text_id", "source", "split", "text_type", "text", "paragraph_id",
                                                   "target_annotation", "group_id"]].assign(sent_idx=0)
    texts = pd.concat([from_pairs[~from_pairs["source"].isin(["finentity"])]] + text_frames, ignore_index=True)
    texts["n_targets"] = texts["text_id"].map(pairs.groupby("text_id").size()).fillna(0).astype(int)
    texts = texts[TEXT_COLUMNS]

    from .completion import add_completed_targets
    pairs, texts, cinfo = add_completed_targets(pairs, texts, Path(args.annotations_dir), aliases)
    pairs["entity_seen_in_train"] = mark_entity_seen(pairs)
    check_no_leakage(pairs)
    check_no_leakage(texts.assign(source=texts["source"]))
    meta = {
        "SEntFiN csv sha256": _sha(sentfin_csv),
        "SEntFiN split": split_note,
        "FiQA": f"{fiqa_mod.FIQA_REPO}@{args.fiqa_revision[:12]}" if fq is not None else "not loaded",
        "FiQA score -> label": f"negative < {fiqa_mod.NEG_BELOW} <= neutral < {fiqa_mod.POS_FROM} <= positive",
        "FinEntity": fe_note,
        "LLM target completion": f"{cinfo['targets_added']} targets added in {cinfo['texts_completed']} texts",
    }
    return pairs, texts, pd.concat(issues, ignore_index=True), meta


def label_files(annotations_dir: Path) -> tuple[list[Path], list[Path]]:
    llm = sorted(p for p in (annotations_dir / "llm").glob("*.jsonl") if not p.name.endswith(".usage.jsonl"))
    human = sorted((annotations_dir / "human").glob("*.csv"))
    return llm, human


def build(args) -> None:
    from .annotate import PROMPT_VERSION
    from .aspects import attach_aspects
    from .export import export_dataset
    from .profile import build_report
    from .taxonomy import TAXONOMY_VERSION, guidelines_markdown

    from .calibration import (DEFAULT_FIQA_SIGMA, DEFAULT_SENTFIN_FLIP_RATE, FINBEN_THRESHOLDS, add_soft_labels,
                              attach_llm_sentiment, llm_sentiment_files, load_calibration, refresh_text_stats,
                              relabel_fiqa)

    pairs, texts, issues, meta = load_pairs(args)
    pairs, sinfo = attach_llm_sentiment(pairs, llm_sentiment_files(Path(args.annotations_dir)))
    if sinfo["dropped_unlabelled"]:
        print(f"[warn] {sinfo['dropped_unlabelled']} LLM-added targets have no sentiment yet and were left out; run "
              "`annotate --task sentiment --llm-targets`", file=sys.stderr)
    meta["LLM target completion"] += f"; {sinfo['with_sentiment']} with bridge sentiment"

    # Sentiment: one meaning of negative/neutral/positive across sources + soft labels.
    cal = load_calibration(Path(args.annotations_dir) / "calibration" / "fiqa_thresholds.json")
    if args.fiqa_thresholds == "calibrated" and cal is None:
        print("[warn] no annotations/calibration/fiqa_thresholds.json; using FinBen thresholds", file=sys.stderr)
    use_cal = args.fiqa_thresholds == "calibrated" and cal is not None
    neg_below, pos_from = (cal["neg_below"], cal["pos_from"]) if use_cal else FINBEN_THRESHOLDS
    sigma = args.fiqa_sigma if args.fiqa_sigma is not None else (cal or {}).get("fiqa_score_sigma", DEFAULT_FIQA_SIGMA)
    flip = (args.sentfin_flip_rate if args.sentfin_flip_rate is not None
            else (cal or {}).get("sentfin_flip_rate", DEFAULT_SENTFIN_FLIP_RATE))
    pairs, texts = refresh_text_stats(relabel_fiqa(pairs, neg_below, pos_from), texts)
    pairs = add_soft_labels(pairs, sigma, flip, neg_below, pos_from, (cal or {}).get("bridge_flip_by_confidence"))
    meta["FiQA score -> label"] = (f"negative < {neg_below} <= neutral < {pos_from} <= positive "
                                   f"({'calibrated to SEntFiN, see sentiment_label_calibration.md' if use_cal else 'FinBen'}); "
                                   "FinBen class kept in polarity_finben")
    meta["soft labels"] = f"FiQA: Normal(score, sigma={sigma}) around the thresholds; SEntFiN: ordinal smoothing, flip rate {flip}"

    llm_files, human_files = label_files(Path(args.annotations_dir))
    pairs, info = attach_aspects(pairs, llm_files, human_files)
    if info["target_mismatch"]:
        print(f"[warn] {info['target_mismatch']} aspect labels skipped: their target no longer matches the uid",
              file=sys.stderr)
    if info["llm_stale_prompt"]:
        print(f"[warn] {info['llm_stale_prompt']} LLM labels come from an older prompt version; "
              "re-run `annotate` to refresh them", file=sys.stderr)
    pairs = pairs[PAIR_COLUMNS]
    meta["aspect taxonomy"] = f"{TAXONOMY_VERSION} (prompt {PROMPT_VERSION})"
    meta["aspect label files"] = ", ".join(str(p) for p in llm_files + human_files) or "none"

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for source in ("sentfin", "fiqa", "finentity"):
        if (pairs["source"] == source).any():
            summary[source] = export_dataset(pairs[pairs["source"] == source], root / source,
                                             texts[texts["source"] == source])
    if len(summary) > 1:
        summary["combined"] = export_dataset(pairs, root / "combined", texts)
    issues.to_csv(root / "issues.csv", index=False)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(pairs, issues, meta), encoding="utf-8")
    GUIDELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDELINES_PATH.write_text(guidelines_markdown(), encoding="utf-8")

    for name, counts in summary.items():
        print(f"{name:9s} " + "  ".join(f"{k}={v}" for k, v in counts.items()
                                         if k.startswith(("pairs_", "sentences_", "instruct_aspect_"))))
    print(f"aspect labels: llm={info['llm_labels']} human={info['human_labels']} "
          f"(unlabelled pairs: {int(pairs['aspect'].isna().sum())})")
    print(f"issues logged: {len(issues)} -> {root / 'issues.csv'}")
    print(f"report: {report}")


def _select_text_ids(path: str) -> set[str]:
    p = Path(path)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p, sheet_name="review")
    return set(df["text_id"].astype(str))


def _task(name: str):
    from .annotate import ASPECT_TASK
    from .calibration import SENTIMENT_TASK

    return {"aspect": ASPECT_TASK, "sentiment": SENTIMENT_TASK}[name]


def _label_path(args, source: str) -> Path:
    if args.task == "sentiment" and getattr(args, "llm_targets", False):  # training labels for LLM-added targets
        return Path(f"{args.annotations_dir}/llm_sentiment/{source}_{args.model}.jsonl")
    if args.task == "sentiment":  # bridge labels for calibration only, never training labels
        return Path(f"{args.annotations_dir}/calibration/{source}_{args.model}_sentiment.jsonl")
    return Path(f"{args.annotations_dir}/llm/{source}_{args.model}.jsonl")


def annotate(args) -> None:
    from .annotate import Annotator, make_client, make_items, render_batch

    pairs, _, _, _ = load_pairs(args)
    pairs = pairs[pairs["source"] == args.source]
    if getattr(args, "llm_targets", False):
        pairs = pairs[pairs["target_source"] == "llm"]
    if args.splits:
        pairs = pairs[pairs["split"].isin(args.splits)]
    if args.text_ids:
        pairs = pairs[pairs["text_id"].isin(_select_text_ids(args.text_ids))]
    if pairs.empty:
        sys.exit(f"no {args.source} pairs selected (is the source loaded?)")
    task = _task(args.task)
    out = Path(args.out) if args.out else _label_path(args, args.source)
    print(f"{args.source}: {pairs['text_id'].nunique()} texts / {len(pairs)} pairs selected; task {task.name}, "
          f"model {args.model}, prompt {task.version}, output {out}")
    if args.dry_run:
        prompt, _ = render_batch(make_items(pairs)[: args.batch_size])
        print(f"--- instructions ({len(task.instructions)} chars) ---\n{task.instructions}\n"
              f"--- first batch input ---\n{prompt}")
        return
    annotator = Annotator(make_client(), model=args.model, reasoning_effort=args.reasoning_effort, task=task)
    stats = annotator.run(pairs, out, batch_size=args.batch_size, workers=args.workers, limit_texts=args.limit)
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}, indent=1))
    for e in stats["errors"]:
        print("  error:", e, file=sys.stderr)
    if stats["pairs_failed"]:
        print(f"[warn] {stats['pairs_failed']} pairs failed; run the same command again to retry them.",
              file=sys.stderr)


def review_sheet(args) -> None:
    from .review import sample_review_pairs, write_review_sheet

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists (it may hold your labels); pass --overwrite to replace it")
    pairs, _, _, _ = load_pairs(args)
    sample = sample_review_pairs(pairs, n_texts=args.n, seed=args.sample_seed)
    write_review_sheet(sample, out)
    print(f"{sample['text_id'].nunique()} headlines / {len(sample)} pairs -> {out}")


def review_eval(args) -> None:
    from .annotate import latest_labels
    from .review import evaluate_review, read_review

    review = read_review(args.sheet)
    llm = latest_labels(Path(args.llm_labels or f"{args.annotations_dir}/llm/sentfin_{args.model}.jsonl"))
    report, merged = evaluate_review(review, llm)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(report, encoding="utf-8")
    HUMAN_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(HUMAN_LABELS_PATH, index=False)
    print(report)
    print(f"\nhuman labels: {len(review)} -> {HUMAN_LABELS_PATH}; report -> {QUALITY_REPORT_PATH}")


def _bridge_check(df: pd.DataFrame) -> str:
    from sklearn.metrics import cohen_kappa_score, f1_score

    from .profile import md_table
    from .schema import LABELS

    y, yhat = df["polarity"], df["sentiment"]
    acc = (y == yhat).mean()
    ci = 1.96 * (acc * (1 - acc) / len(y)) ** 0.5
    head = md_table([{"pairs": len(y), "accuracy": f"{acc:.3f} ± {ci:.3f}",
                      "macro-F1": f"{f1_score(y, yhat, labels=list(LABELS), average='macro'):.3f}",
                      "Cohen's kappa": f"{cohen_kappa_score(y, yhat):.3f}"}])
    by_conf = md_table([{"LLM confidence": c, "pairs": int((df["confidence"] == c).sum()),
                         "accuracy": f"{(y[df['confidence'] == c] == yhat[df['confidence'] == c]).mean():.3f}"}
                        for c in ("high", "medium", "low") if (df["confidence"] == c).any()])
    cm = pd.crosstab(y, yhat).reindex(index=list(LABELS), columns=list(LABELS), fill_value=0)
    conf = md_table([{"gold \\ LLM": g, **{l: int(cm.loc[g, l]) for l in LABELS}} for g in LABELS])
    return f"{head}\n{by_conf}\n**Confusion (rows: SEntFiN gold)**\n\n{conf}"


def calibrate(args) -> None:
    """Fit FiQA score thresholds to SEntFiN's label semantics via the bridge annotator."""
    from .annotate import latest_labels
    from .calibration import (SENTIMENT_EXAMPLES, SENTIMENT_TASK, apply_thresholds, bootstrap_thresholds,
                              choose_thresholds, estimate_score_sigma, fit_thresholds, sentfin_duplicate_agreement)
    from .profile import md_table
    from .schema import LABELS

    args.task = "sentiment"
    pairs, _, _, _ = load_pairs(args)
    v = SENTIMENT_TASK.version

    def joined(source):
        lab = latest_labels(_label_path(args, source), v)
        if lab.empty:
            sys.exit(f"no sentiment labels for {source}: run `annotate --task sentiment --source {source}` first")
        lab = lab[lab["prompt_version"] == v]
        return pairs[pairs["source"] == source].merge(lab[["uid", "sentiment", "confidence"]], on="uid")

    sf, fq_all = joined("sentfin"), joined("fiqa")
    sf_eval = sf[sf["split"] != "test"]  # never touch SEntFiN test
    fq = fq_all[fq_all["split"] != "test"]  # thresholds are fitted on FiQA train + valid only
    fq_test = fq_all[fq_all["split"] == "test"]
    dist = lambda x: {l: f"{(x == l).mean():.1%}" for l in LABELS}  # noqa: E731
    finben = apply_thresholds(fq["score"], -0.1, 0.1)
    fit = fit_thresholds(fq["score"], fq["sentiment"])
    ci = bootstrap_thresholds(fq["score"], fq["sentiment"])
    chosen = choose_thresholds(fit, ci)
    calibrated = apply_thresholds(fq["score"], chosen["neg_below"], chosen["pos_from"])
    chosen_acc = float((calibrated == fq["sentiment"]).mean())
    sigma = estimate_score_sigma(pairs[pairs["source"] == "fiqa"])
    dup = sentfin_duplicate_agreement(args.sentfin_csv)

    flip_by_conf = {c: round(1 - float((g["sentiment"] == g["polarity"]).mean()), 3)
                    for c, g in sf_eval.groupby("confidence") if len(g) >= 3}
    test_agree = {"finben": float((apply_thresholds(fq_test["score"], -0.1, 0.1) == fq_test["sentiment"]).mean()),
                  "chosen": float((apply_thresholds(fq_test["score"], chosen["neg_below"], chosen["pos_from"])
                                   == fq_test["sentiment"]).mean())}
    result = {"neg_below": chosen["neg_below"], "pos_from": chosen["pos_from"],
              "fitted_on": "FiQA train+valid", "heldout_test_agreement": {k: round(v, 4) for k, v in test_agree.items()},
              "bridge_flip_by_confidence": flip_by_conf,
              "fitted": {"neg_below": fit["neg_below"], "pos_from": fit["pos_from"]}, **ci,
              "fit_accuracy": fit["accuracy"], "chosen_accuracy": round(chosen_acc, 4),
              "fiqa_pairs": fit["n"], "bridge": f"llm:{args.model}@{v}",
              "bridge_vs_sentfin_accuracy": round(float((sf_eval["sentiment"] == sf_eval["polarity"]).mean()), 4),
              "fiqa_score_sigma": sigma["sigma"], "sentfin_flip_rate": dup["flip_rate"]}
    out = Path(args.annotations_dir) / "calibration" / "fiqa_thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")

    rows = [{"labels": "SEntFiN gold (valid)", **dist(sf_eval["polarity"])},
            {"labels": "bridge LLM on SEntFiN (valid)", **dist(sf_eval["sentiment"])},
            {"labels": "FiQA train+valid, FinBen thresholds ±0.1", **dist(finben)},
            {"labels": f"FiQA, calibrated [{chosen['neg_below']}, {chosen['pos_from']})", **dist(calibrated)},
            {"labels": "bridge LLM on FiQA train+valid", **dist(fq["sentiment"])}]
    by_llm = fq.groupby("sentiment")["score"].describe()[["count", "25%", "50%", "75%"]].round(3)
    report = "\n".join([
        "# Calibrating FiQA's score thresholds to SEntFiN's labels\n",
        "_Generated by `python -m absa_data calibrate`._\n",
        "SEntFiN gives hard classes; FiQA gives scores in [-1, 1] that are cut into classes at thresholds. "
        "The FinBen thresholds (±0.1) make only 7.8 % of FiQA neutral against 38 % of SEntFiN, so the two "
        "datasets' \"neutral\" mean different things. An LLM taught SEntFiN's conventions with "
        f"{len(SENTIMENT_EXAMPLES)} SEntFiN *train* examples "
        "is used as a bridge annotator: its agreement with SEntFiN gold is checked on SEntFiN *valid*, then the "
        "FiQA thresholds that best reproduce its FiQA judgements are fitted. FiQA labels are still computed from "
        "the human scores.\n",
        f"## 1. Does the bridge annotator reproduce SEntFiN? (valid, {len(sf_eval)} pairs)\n",
        _bridge_check(sf_eval),
        "\n## 2. Label distributions\n", md_table(rows),
        "\n## 3. FiQA score by bridge label\n",
        md_table([{"bridge label": k, "pairs": int(r["count"]), "score q25": r["25%"], "median": r["50%"],
                   "q75": r["75%"]} for k, r in by_llm.iterrows()]),
        "\n## 4. Thresholds\n",
        md_table([
            {"thresholds": "FinBen", "neg_below": -0.1, "pos_from": 0.1,
             "agreement with bridge": f"{(finben == fq['sentiment']).mean():.1%}"},
            {"thresholds": "best fit", "neg_below": fit["neg_below"], "pos_from": fit["pos_from"],
             "agreement with bridge": f"{fit['accuracy']:.1%}"},
            {"thresholds": "**chosen**", "neg_below": f"**{chosen['neg_below']}**",
             "pos_from": f"**{chosen['pos_from']}**", "agreement with bridge": f"{chosen_acc:.1%}"},
        ]),
        f"\nBootstrap 95 % intervals of the best fit: neg_below {ci['neg_below_95ci']}, pos_from "
        f"{ci['pos_from_95ci']}. Rule: a FinBen threshold is kept unless its interval excludes it (the "
        "agreement surface is flat over wide ranges, so a marginally better point estimate is not evidence). "
        "The build uses the chosen thresholds for `polarity` / `label` and the soft labels. Thresholds are "
        f"fitted on FiQA train + valid ({len(fq)} pairs); on the held-out FiQA test ({len(fq_test)} pairs) the "
        f"chosen thresholds agree with the bridge on {test_agree['chosen']:.1%} vs {test_agree['finben']:.1%} for "
        "FinBen.\n",
        "## 5. Label noise estimates (for soft labels)\n",
        f"* FiQA score noise σ ≈ **{sigma['sigma']}** from {sigma['groups']} (text, target) pairs scored more than "
        f"once ({sigma['differences']} differences: {sigma['scores']}). Very small sample.",
        f"* SEntFiN: {dup['compared']} repeated judgements in duplicated headlines, {dup['conflicts']} disagree "
        f"({dup['by_pair']}) → per-label flip rate ≈ **{dup['flip_rate']}** (half the disagreement rate).",
        f"\nSaved to `{out}`.\n",
    ])
    CALIBRATION_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


def _code_url() -> str:
    """GitHub URL of the current commit (the code that produced the data), if available."""
    import subprocess

    try:
        remote = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True,
                                check=True).stdout.strip().removesuffix(".git")
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "absa_data"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "(not recorded)"
    if dirty:
        print("[warn] absa_data/ has uncommitted changes; the card links to the last commit", file=sys.stderr)
    return f"{remote}/tree/{sha}" if remote.startswith("https://github.com/") else f"commit {sha}"


def push_hub(args) -> None:
    from .hub import push, write_staging

    processed = Path(args.processed)
    if not (processed / "sentfin" / "pairs.parquet").exists():
        sys.exit(f"{processed} has no build output: run `python -m absa_data build` first")
    staging = write_staging(processed, Path("reports"), Path(args.staging), args.repo, _code_url(),
                            Path(args.annotations_dir) / "calibration" / "fiqa_thresholds.json")
    has_fiqa = (staging / "fiqa").exists()
    print(f"staged -> {staging} ({'with' if has_fiqa else 'without'} FiQA)")
    if args.dry_run:
        return
    gated = None if args.gated == "none" else args.gated
    if has_fiqa and gated is None and not args.private:
        sys.exit("refusing to publish FiQA data publicly without a gate (FiQA is gated by TheFinAI); "
                 "use --gated auto|manual or --private")
    result = push(staging, args.repo, gated=gated, public=not args.private,
                  commit_message=args.message or "Update processed SEntFiN + FiQA data")
    print(json.dumps(result, indent=1))


def complete_targets(args) -> None:
    from .annotate import make_client
    from .completion import (PROMPT_VERSION, TargetCompleter, hide_gold, items_from, latest_completions,
                             score_validation)

    pairs, texts, _, _ = load_pairs(args)
    pairs = pairs[(pairs["source"] == args.source) & (pairs["target_source"] == "gold")]
    texts = texts[texts["source"] == args.source]
    if args.splits:
        texts = texts[texts["split"].isin(args.splits)]
    if args.limit:
        texts = texts.head(args.limit)
    pairs = pairs[pairs["text_id"].isin(texts["text_id"])]
    out_dir = Path(args.annotations_dir) / "completion"
    if args.validate:
        if args.source != "finentity":
            sys.exit("--validate needs exhaustive gold: use --source finentity")
        given, hidden = hide_gold(pairs, texts, keep_prob=args.keep_prob, seed=args.seed)
        items = items_from(pairs, texts, given)
        out = out_dir / f"finentity_validation_{args.model}_keep{args.keep_prob}_seed{args.seed}.jsonl"
    else:
        items = items_from(pairs, texts)
        out = out_dir / f"{args.source}_{args.model}.jsonl"
    print(f"{args.source}: {len(items)} texts, {sum(len(i.pairs) for i in items)} given targets; prompt "
          f"{PROMPT_VERSION}; output {out}")
    if args.dry_run:
        from .completion import INSTRUCTIONS, render
        print(INSTRUCTIONS, "\n---\n", render(items[:args.batch_size]))
        return
    stats = TargetCompleter(make_client(), model=args.model, reasoning_effort=args.reasoning_effort).run(
        items, out, batch_size=args.batch_size, workers=args.workers)
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}, indent=1))
    for e in stats["errors"]:
        print("  error:", e, file=sys.stderr)
    if args.validate:
        recs = latest_completions(out)
        recs = recs[recs["text_id"].isin(texts["text_id"])]
        gold_all = pairs.groupby("text_id")["target"].agg(list).to_dict()
        res = score_validation(recs, hidden, gold_all)
        print(json.dumps(res, indent=1))
        (out.with_suffix(".score.json")).write_text(json.dumps(res, indent=1) + "\n")
        _write_completion_report(res, args, len(texts), sum(len(v) for v in given.values()))


def _write_completion_report(res: dict, args, n_texts: int, n_given: int) -> None:
    from .completion import FINENTITY_SCOPE, PROMPT_VERSION
    from .profile import md_table

    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    report = "\n".join([
        "# Validation of LLM target completion on FinEntity\n",
        "_Generated by `python -m absa_data complete-targets --source finentity --validate`._\n",
        "FinEntity annotates every financial entity, so it can check the completion used on SEntFiN and FiQA. "
        f"On FinEntity {', '.join(args.splits or ['all'])} ({n_texts} sentences), each distinct gold target was "
        f"kept with probability {args.keep_prob} ({n_given} given) and the rest hidden; the model "
        f"(`{args.model}`, prompt `{PROMPT_VERSION}`) had to find the hidden ones.\n",
        md_table([
            {"metric": "hidden gold targets", "value": res["hidden"]},
            {"metric": "recall, exact match", "value": pct(res["recall_exact"])},
            {"metric": "recall, exact or partial match", "value": pct(res["recall_partial"])},
            {"metric": "additions", "value": res["added"]},
            {"metric": "precision, all additions", "value": pct(res["precision"])},
            {"metric": f"precision, additions typed {' / '.join(FINENTITY_SCOPE)} ({res['added_in_scope']})",
             "value": pct(res["precision_in_finentity_scope"])},
            {"metric": f"additions on the {res['empty_texts']} sentences without any gold target",
             "value": res["additions_on_empty_texts"]},
        ]),
        "\n**Reading the precision.** FinEntity does not annotate currencies, most commodities, sectors, "
        "countries or generic market references ('Wall Street', 'Japanese shares'), while SEntFiN does. The "
        "completion follows SEntFiN's convention, so additions of those types count as errors here although "
        "they are intended. Precision within FinEntity's own scope (companies, indices, regulators) is the "
        "meaningful number. References that cannot be tied to an entity ('the market', 'the index') are "
        "removed by a rule in code.\n",
    ])
    Path("reports/target_completion_validation.md").write_text(report, encoding="utf-8")


def eda(args) -> None:
    from .eda import run

    out = run(Path(args.processed), Path(args.out), Path(args.target_domain) if args.target_domain else None,
              Path(args.annotations_dir) / "calibration" / "fiqa_thresholds.json")
    print(f"report: {out}")


def _data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sentfin-csv", default="SEntFiN.csv")
    p.add_argument("--entity-list", default="entity_list_comprehensive.csv")
    p.add_argument("--split-manifest", default="splits/sentfin_split_v1.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fiqa-dir", help="directory with {train,valid,test}*.parquet (skips download)")
    p.add_argument("--fiqa-revision", default=fiqa_mod.FIQA_REVISION)
    p.add_argument("--no-fiqa", action="store_true")
    p.add_argument("--require-fiqa", action="store_true", help="fail instead of skipping FiQA")
    p.add_argument("--finentity-path", help="local FinEntity.json (skips download)")
    p.add_argument("--no-finentity", action="store_true")
    p.add_argument("--finentity-split-manifest", default="splits/finentity_split_v1.csv")
    p.add_argument("--annotations-dir", default="annotations")


def main(argv=None) -> None:
    from .annotate import DEFAULT_MODEL

    ap = argparse.ArgumentParser(prog="python -m absa_data", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="clean, split, attach aspects, export and profile the datasets")
    _data_args(b)
    b.add_argument("--resplit", action="store_true", help="ignore the manifest and draw a new SEntFiN split")
    b.add_argument("--fiqa-thresholds", choices=["calibrated", "finben"], default="calibrated",
                   help="how FiQA scores become classes in `polarity`: thresholds fitted to SEntFiN's semantics "
                        "(annotations/calibration/fiqa_thresholds.json) or FinBen's ±0.1")
    b.add_argument("--fiqa-sigma", type=float, help="score noise for FiQA soft labels (default: calibration file)")
    b.add_argument("--sentfin-flip-rate", type=float,
                   help="adjacent-class flip rate for SEntFiN soft labels (default: calibration file; 0 = one-hot)")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--report", default="reports/data_profile.md")

    a = sub.add_parser("annotate", help="label aspect categories with an LLM")
    _data_args(a)
    a.add_argument("--source", choices=["sentfin", "fiqa", "finentity"], required=True)
    a.add_argument("--task", choices=["aspect", "sentiment"], default="aspect",
                   help="sentiment = bridge labels for `calibrate` (stored under annotations/calibration/)")
    a.add_argument("--splits", nargs="+", choices=["train", "valid", "test"])
    a.add_argument("--llm-targets", action="store_true", help="only targets added by complete-targets")
    a.add_argument("--text-ids", help="CSV/XLSX with a text_id column (e.g. the review sheet) to restrict to")
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--reasoning-effort", default="low", help="passed to the API; '' to use the model default")
    a.add_argument("--batch-size", type=int, default=10, help="texts per request")
    a.add_argument("--workers", type=int, default=4, help="parallel requests")
    a.add_argument("--limit", type=int, help="label at most this many (not yet labelled) texts")
    a.add_argument("--out", help="default: <annotations-dir>/llm/<source>_<model>.jsonl "
                                 "(sentiment: <annotations-dir>/calibration/<source>_<model>_sentiment.jsonl)")
    a.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")

    r = sub.add_parser("review-sheet", help="write a blind review spreadsheet of SEntFiN test pairs")
    _data_args(r)
    r.add_argument("--n", type=int, default=400, help="number of test headlines")
    r.add_argument("--sample-seed", type=int, default=13)
    r.add_argument("--out", default=str(REVIEW_SHEET_PATH))
    r.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("review-eval", help="score the LLM against a filled review sheet")
    e.add_argument("--sheet", default=str(REVIEW_SHEET_PATH))
    e.add_argument("--annotations-dir", default="annotations")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--llm-labels", help="default: <annotations-dir>/llm/sentfin_<model>.jsonl")

    t = sub.add_parser("complete-targets", help="LLM: add the targets a source's annotators missed")
    _data_args(t)
    t.add_argument("--source", choices=["sentfin", "fiqa", "finentity"], required=True)
    t.add_argument("--splits", nargs="+", choices=["train", "valid", "test"])
    t.add_argument("--validate", action="store_true", help="FinEntity only: hide part of the gold, score recovery")
    t.add_argument("--keep-prob", type=float, default=0.5)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--reasoning-effort", default="low")
    t.add_argument("--batch-size", type=int, default=10)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--limit", type=int)
    t.add_argument("--dry-run", action="store_true")

    x = sub.add_parser("eda", help="exploratory data analysis of data/processed -> reports/eda/")
    x.add_argument("--processed", default="data/processed")
    x.add_argument("--out", default="reports/eda")
    x.add_argument("--target-domain", default="../minhhai205/aspect-based-fsa/data/exports",
                   help="crawler exports (<day>/<doc>.json) used as the deployment domain")
    x.add_argument("--annotations-dir", default="annotations")

    c = sub.add_parser("calibrate", help="fit FiQA score thresholds to SEntFiN's label semantics")
    _data_args(c)
    c.add_argument("--model", default=DEFAULT_MODEL)

    h = sub.add_parser("push-hub", help="publish data/processed to a Hugging Face dataset repo")
    h.add_argument("--repo", required=True, help="<user>/<name>")
    h.add_argument("--processed", default="data/processed")
    h.add_argument("--staging", default="data/hub_staging")
    h.add_argument("--annotations-dir", default="annotations")
    h.add_argument("--gated", choices=["auto", "manual", "none"], default="auto")
    h.add_argument("--private", action="store_true", help="keep the repo private")
    h.add_argument("--message", help="commit message on the Hub")
    h.add_argument("--dry-run", action="store_true", help="stage files and card only; upload nothing")

    args = ap.parse_args(argv)
    if getattr(args, "reasoning_effort", None) == "":
        args.reasoning_effort = None
    {"build": build, "annotate": annotate, "review-sheet": review_sheet, "review-eval": review_eval,
     "calibrate": calibrate, "push-hub": push_hub, "complete-targets": complete_targets, "eda": eda}[args.cmd](args)


if __name__ == "__main__":
    main()
