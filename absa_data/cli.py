"""Command line entry point: ``python -m absa_data <command>``.

Commands:
  build         clean, split, attach aspect labels, export and profile the datasets
  annotate      label aspect categories with an LLM (OpenAI), resumable
  review-sheet  write a blind spreadsheet of SEntFiN test pairs for human aspect labels
  review-eval   compare filled review sheet with the LLM labels; store human labels as gold
  calibrate     fit FiQA score thresholds to SEntFiN's label semantics (bridge LLM)
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
from .splits import (apply_manifest, assign_groups, check_no_leakage, mark_entity_seen,
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


def load_pairs(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load, clean and split both sources. Returns (pairs, issues, meta)."""
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
    frames, issues = [sf], [sf_issues]

    fq = _load_fiqa(args)
    if fq is not None:
        fq_pairs, fq_issues = fq
        fq_pairs["group_id"] = assign_groups(fq_pairs)
        frames.append(fq_pairs)
        issues.append(fq_issues)

    pairs = pd.concat(frames, ignore_index=True)
    pairs["entity_seen_in_train"] = mark_entity_seen(pairs)
    check_no_leakage(pairs)
    meta = {
        "SEntFiN csv sha256": _sha(sentfin_csv),
        "SEntFiN split": split_note,
        "FiQA": f"{fiqa_mod.FIQA_REPO}@{args.fiqa_revision[:12]}" if fq is not None else "not loaded",
        "FiQA score -> label": f"negative < {fiqa_mod.NEG_BELOW} <= neutral < {fiqa_mod.POS_FROM} <= positive",
    }
    return pairs, pd.concat(issues, ignore_index=True), meta


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
                              load_calibration, relabel_fiqa)

    pairs, issues, meta = load_pairs(args)

    # Sentiment: one meaning of negative/neutral/positive across sources + soft labels.
    cal = load_calibration(Path(args.annotations_dir) / "calibration" / "fiqa_thresholds.json")
    if args.fiqa_thresholds == "calibrated" and cal is None:
        print("[warn] no annotations/calibration/fiqa_thresholds.json; using FinBen thresholds", file=sys.stderr)
    use_cal = args.fiqa_thresholds == "calibrated" and cal is not None
    neg_below, pos_from = (cal["neg_below"], cal["pos_from"]) if use_cal else FINBEN_THRESHOLDS
    sigma = args.fiqa_sigma if args.fiqa_sigma is not None else (cal or {}).get("fiqa_score_sigma", DEFAULT_FIQA_SIGMA)
    flip = (args.sentfin_flip_rate if args.sentfin_flip_rate is not None
            else (cal or {}).get("sentfin_flip_rate", DEFAULT_SENTFIN_FLIP_RATE))
    pairs = add_soft_labels(relabel_fiqa(pairs, neg_below, pos_from), sigma, flip, neg_below, pos_from)
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
    has_fiqa = (pairs["source"] == "fiqa").any()
    summary = {"sentfin": export_dataset(pairs[pairs["source"] == "sentfin"], root / "sentfin", e2e=True)}
    if has_fiqa:
        summary["fiqa"] = export_dataset(pairs[pairs["source"] == "fiqa"], root / "fiqa", e2e=False)
        summary["combined"] = export_dataset(pairs, root / "combined", e2e=False)
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
    if args.task == "sentiment":  # bridge labels for calibration only, never training labels
        return Path(f"{args.annotations_dir}/calibration/{source}_{args.model}_sentiment.jsonl")
    return Path(f"{args.annotations_dir}/llm/{source}_{args.model}.jsonl")


def annotate(args) -> None:
    from .annotate import Annotator, make_client, make_items, render_batch

    pairs, _, _ = load_pairs(args)
    pairs = pairs[pairs["source"] == args.source]
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
    pairs, _, _ = load_pairs(args)
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
    pairs, _, _ = load_pairs(args)
    v = SENTIMENT_TASK.version

    def joined(source):
        lab = latest_labels(_label_path(args, source), v)
        if lab.empty:
            sys.exit(f"no sentiment labels for {source}: run `annotate --task sentiment --source {source}` first")
        lab = lab[lab["prompt_version"] == v]
        return pairs[pairs["source"] == source].merge(lab[["uid", "sentiment", "confidence"]], on="uid")

    sf, fq = joined("sentfin"), joined("fiqa")
    sf_eval = sf[sf["split"] != "test"]  # never touch SEntFiN test
    dist = lambda x: {l: f"{(x == l).mean():.1%}" for l in LABELS}  # noqa: E731
    finben = apply_thresholds(fq["score"], -0.1, 0.1)
    fit = fit_thresholds(fq["score"], fq["sentiment"])
    ci = bootstrap_thresholds(fq["score"], fq["sentiment"])
    chosen = choose_thresholds(fit, ci)
    calibrated = apply_thresholds(fq["score"], chosen["neg_below"], chosen["pos_from"])
    chosen_acc = float((calibrated == fq["sentiment"]).mean())
    sigma = estimate_score_sigma(pairs[pairs["source"] == "fiqa"])
    dup = sentfin_duplicate_agreement(args.sentfin_csv)

    result = {"neg_below": chosen["neg_below"], "pos_from": chosen["pos_from"],
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
            {"labels": "FiQA, FinBen thresholds ±0.1", **dist(finben)},
            {"labels": f"FiQA, calibrated [{chosen['neg_below']}, {chosen['pos_from']})", **dist(calibrated)},
            {"labels": "bridge LLM on FiQA", **dist(fq["sentiment"])}]
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
        "The build uses the chosen thresholds for `polarity` / `label` and the soft labels.\n",
        "## 5. Label noise estimates (for soft labels)\n",
        f"* FiQA score noise σ ≈ **{sigma['sigma']}** from {sigma['groups']} (text, target) pairs scored more than "
        f"once ({sigma['differences']} differences: {sigma['scores']}). Very small sample.",
        f"* SEntFiN: {dup['compared']} repeated judgements in duplicated headlines, {dup['conflicts']} disagree "
        f"({dup['by_pair']}) → per-label flip rate ≈ **{dup['flip_rate']}** (half the disagreement rate).",
        f"\nSaved to `{out}`.\n",
    ])
    CALIBRATION_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


def _data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sentfin-csv", default="SEntFiN.csv")
    p.add_argument("--entity-list", default="entity_list_comprehensive.csv")
    p.add_argument("--split-manifest", default="splits/sentfin_split_v1.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fiqa-dir", help="directory with {train,valid,test}*.parquet (skips download)")
    p.add_argument("--fiqa-revision", default=fiqa_mod.FIQA_REVISION)
    p.add_argument("--no-fiqa", action="store_true")
    p.add_argument("--require-fiqa", action="store_true", help="fail instead of skipping FiQA")
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
    a.add_argument("--source", choices=["sentfin", "fiqa"], required=True)
    a.add_argument("--task", choices=["aspect", "sentiment"], default="aspect",
                   help="sentiment = bridge labels for `calibrate` (stored under annotations/calibration/)")
    a.add_argument("--splits", nargs="+", choices=["train", "valid", "test"])
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

    c = sub.add_parser("calibrate", help="fit FiQA score thresholds to SEntFiN's label semantics")
    _data_args(c)
    c.add_argument("--model", default=DEFAULT_MODEL)

    args = ap.parse_args(argv)
    if getattr(args, "reasoning_effort", None) == "":
        args.reasoning_effort = None
    {"build": build, "annotate": annotate, "review-sheet": review_sheet, "review-eval": review_eval,
     "calibrate": calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
