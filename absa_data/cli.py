"""Command line entry point: ``python -m absa_data <command>``.

Commands:
  build         clean, split, attach aspect labels, export and profile the datasets
  annotate      label aspect categories with an LLM (OpenAI), resumable
  review-sheet  write a blind spreadsheet of SEntFiN test pairs for human aspect labels
  review-eval   compare filled review sheet with the LLM labels; store human labels as gold
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

    pairs, issues, meta = load_pairs(args)
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


def annotate(args) -> None:
    from .annotate import PROMPT_VERSION, Annotator, INSTRUCTIONS, make_client, make_items, render_batch

    pairs, _, _ = load_pairs(args)
    pairs = pairs[pairs["source"] == args.source]
    if args.splits:
        pairs = pairs[pairs["split"].isin(args.splits)]
    if args.text_ids:
        pairs = pairs[pairs["text_id"].isin(_select_text_ids(args.text_ids))]
    if pairs.empty:
        sys.exit(f"no {args.source} pairs selected (is the source loaded?)")
    out = Path(args.out or f"{args.annotations_dir}/llm/{args.source}_{args.model}.jsonl")
    print(f"{args.source}: {pairs['text_id'].nunique()} texts / {len(pairs)} pairs selected; "
          f"model {args.model}, prompt {PROMPT_VERSION}, output {out}")
    if args.dry_run:
        prompt, _ = render_batch(make_items(pairs)[: args.batch_size])
        print(f"--- instructions ({len(INSTRUCTIONS)} chars) ---\n{INSTRUCTIONS}\n--- first batch input ---\n{prompt}")
        return
    annotator = Annotator(make_client(), model=args.model, reasoning_effort=args.reasoning_effort)
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
    b.add_argument("--out", default="data/processed")
    b.add_argument("--report", default="reports/data_profile.md")

    a = sub.add_parser("annotate", help="label aspect categories with an LLM")
    _data_args(a)
    a.add_argument("--source", choices=["sentfin", "fiqa"], required=True)
    a.add_argument("--splits", nargs="+", choices=["train", "valid", "test"])
    a.add_argument("--text-ids", help="CSV/XLSX with a text_id column (e.g. the review sheet) to restrict to")
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--reasoning-effort", default="low", help="passed to the API; '' to use the model default")
    a.add_argument("--batch-size", type=int, default=10, help="texts per request")
    a.add_argument("--workers", type=int, default=4, help="parallel requests")
    a.add_argument("--limit", type=int, help="label at most this many (not yet labelled) texts")
    a.add_argument("--out", help="default: <annotations-dir>/llm/<source>_<model>.jsonl")
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

    args = ap.parse_args(argv)
    if getattr(args, "reasoning_effort", None) == "":
        args.reasoning_effort = None
    {"build": build, "annotate": annotate, "review-sheet": review_sheet, "review-eval": review_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
