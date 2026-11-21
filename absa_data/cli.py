"""Command line entry point: ``python -m absa_data build``."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from . import fiqa as fiqa_mod
from .entities import load_alias_table
from .export import export_dataset
from .profile import build_report
from .schema import PAIR_COLUMNS
from .sentfin import load_sentfin
from .splits import (apply_manifest, assign_groups, check_no_leakage, mark_entity_seen,
                     save_manifest, stratified_group_split)


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


def build(args) -> None:
    root = Path(args.out)
    sentfin_csv = Path(args.sentfin_csv)
    aliases = load_alias_table(args.entity_list) if Path(args.entity_list).exists() else None

    sf, sf_issues = load_sentfin(sentfin_csv, aliases)
    sf["group_id"] = assign_groups(sf)
    manifest = Path(args.split_manifest)
    if manifest.exists() and not args.resplit:
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
    pairs = pairs[PAIR_COLUMNS]
    issues = pd.concat(issues, ignore_index=True)

    root.mkdir(parents=True, exist_ok=True)
    summary = {"sentfin": export_dataset(pairs[pairs["source"] == "sentfin"], root / "sentfin", e2e=True)}
    if fq is not None:
        summary["fiqa"] = export_dataset(pairs[pairs["source"] == "fiqa"], root / "fiqa", e2e=False)
        summary["combined"] = export_dataset(pairs, root / "combined", e2e=False)
    issues.to_csv(root / "issues.csv", index=False)

    meta = {
        "SEntFiN csv sha256": _sha(sentfin_csv),
        "SEntFiN split": split_note,
        "FiQA": f"{fiqa_mod.FIQA_REPO}@{args.fiqa_revision[:12]}" if fq is not None else "not loaded",
        "FiQA score -> label": f"negative < {fiqa_mod.NEG_BELOW} <= neutral < {fiqa_mod.POS_FROM} <= positive",
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(pairs, issues, meta), encoding="utf-8")

    for name, counts in summary.items():
        print(f"{name:9s} " + "  ".join(f"{k}={v}" for k, v in counts.items() if k.startswith(("pairs_", "sentences_"))))
    print(f"issues logged: {len(issues)} -> {root / 'issues.csv'}")
    print(f"report: {report}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m absa_data", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="clean, split, export and profile the datasets")
    b.add_argument("--sentfin-csv", default="SEntFiN.csv")
    b.add_argument("--entity-list", default="entity_list_comprehensive.csv")
    b.add_argument("--split-manifest", default="splits/sentfin_split_v1.csv")
    b.add_argument("--resplit", action="store_true", help="ignore the manifest and draw a new SEntFiN split")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--fiqa-dir", help="directory with {train,valid,test}*.parquet (skips download)")
    b.add_argument("--fiqa-revision", default=fiqa_mod.FIQA_REVISION)
    b.add_argument("--no-fiqa", action="store_true")
    b.add_argument("--require-fiqa", action="store_true", help="fail instead of skipping FiQA")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--report", default="reports/data_profile.md")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args)


if __name__ == "__main__":
    main()
