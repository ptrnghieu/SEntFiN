"""End-to-end evaluation: (target, aspect, sentiment) triplets extracted from raw text.

Any system is scored the same way: write one JSON line per text,

    {"text_id": "...", "triplets": [{"start": 0, "end": 8, "sentiment": "positive",
                                      "aspect": "Stock/Price Movement"}, ...]}

(``start``/``end`` may be omitted when ``target`` is given; the target is then located in the
text, which suits LLM outputs), and run

    python -m absa_models.triplets --pred predictions.jsonl [--sources finentity sentfin fiqa]

Gold: every test text whose targets are complete (FinEntity: exhaustive human annotation;
SEntFiN / FiQA / vnnews: gold + LLM-completed targets), with all their (span, sentiment, aspect).
A match needs the exact character span. Scores at five levels, micro P/R/F1 per source:

    span                     target extraction only
    span+sentiment           the (target, sentiment) pairs, FinEntity's task
    span+aspect              (target, aspect)
    triplet                  (target, aspect, sentiment), the project's task
    triplet (aspect level 1) the same with the 5 top-level aspect groups

Label provenance differs by source, so read the numbers accordingly: FinEntity targets and
sentiment are human; aspects are LLM labels everywhere; vnnews is all LLM until reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from absa_data.text import align_targets

from .common import ALL_SOURCES, load_table, span_metrics, write_results

LEVELS = ("span", "span+sentiment", "span+aspect", "triplet", "triplet (aspect level 1)")
COMPLETE = ("exhaustive", "llm_completed")


def _key(t: dict, level: str):
    s, e = int(t["start"]), int(t["end"])
    asp = t.get("aspect")
    return {"span": (s, e), "span+sentiment": (s, e, t.get("sentiment")), "span+aspect": (s, e, asp),
            "triplet": (s, e, t.get("sentiment"), asp),
            "triplet (aspect level 1)": (s, e, t.get("sentiment"), asp.split("/")[0] if isinstance(asp, str) else asp)
            }[level]


def load_gold(processed: Path, sources, split: str = "test") -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """(texts, gold triplets per text_id) for texts with complete target annotation."""
    texts = load_table(processed, "texts", sources)
    texts = texts[(texts["split"] == split) & texts["target_annotation"].isin(COMPLETE)].reset_index(drop=True)
    pairs = load_table(processed, "pairs", sources)
    pairs = pairs[pairs["text_id"].isin(set(texts["text_id"]))]
    gold: dict[str, list[dict]] = {t: [] for t in texts["text_id"]}
    for r in pairs.itertuples(index=False):
        if pd.notna(r.start):
            gold[r.text_id].append({"start": int(r.start), "end": int(r.end), "target": r.target,
                                    "sentiment": r.polarity, "aspect": r.aspect})
    return texts, gold


def read_predictions(path: str | Path, texts: pd.DataFrame) -> dict[str, list[dict]]:
    """Predictions per text_id; triplets without offsets are located by their target string."""
    by_text = dict(zip(texts["text_id"], texts["text"]))
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tid = rec["text_id"]
        if tid not in by_text:
            continue
        trips = [dict(t) for t in rec.get("triplets", [])]
        need = [t for t in trips if t.get("start") is None]
        if need:
            spans = align_targets(by_text[tid], [t.get("target", "") for t in need], fuzzy=False)
            for t, sp in zip(need, spans):
                t["start"], t["end"] = sp.start, sp.end
        out[tid] = [t for t in trips if t.get("start") is not None]
    return out


def score(texts: pd.DataFrame, gold: dict[str, list[dict]], pred: dict[str, list[dict]]) -> dict:
    res = {}
    groups = [("all", texts)] + [(s, texts[texts["source"] == s]) for s in sorted(texts["source"].unique())]
    for name, t in groups:
        ids = list(t["text_id"])
        for level in LEVELS:
            g = [{_key(x, level) for x in gold.get(i, [])} for i in ids]
            p = [{_key(x, level) for x in pred.get(i, [])} for i in ids]
            res[f"{name} {level}"] = {**span_metrics(g, p), "texts": len(ids)}
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="JSONL predictions, one line per text")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--sources", nargs="+", default=["finentity", "sentfin", "fiqa"], choices=ALL_SOURCES)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", help="directory for metrics.json / metrics.md (default: next to --pred)")
    args = ap.parse_args(argv)
    texts, gold = load_gold(Path(args.processed), args.sources, args.split)
    pred = read_predictions(args.pred, texts)
    res = score(texts, gold, pred)
    out = Path(args.out or Path(args.pred).parent)
    write_results(out, {"config": vars(args), args.split: res}, "Triplet extraction (end to end)")
    for k in ("finentity span+sentiment", "finentity triplet", "all triplet"):
        if k in res:
            print(f"{k}: {res[k]}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
