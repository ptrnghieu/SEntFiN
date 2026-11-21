"""Write the unified tables in the formats the usual fine-tuning setups expect.

For each dataset directory (``sentfin/``, ``fiqa/``, ``combined/``):

* ``pairs.parquet`` / ``pairs.csv``: the full unified table, all splits.
* ``pairs_{split}.jsonl``: one (text, target) per line, for encoder models
  (BERT/FinBERT/DeBERTa sentence-pair or target-marked classification), with
  the hard ``label`` and the ``soft_label`` [P(neg), P(neu), P(pos)].
* ``sentences_{split}.jsonl``: one text per line with all its targets and
  character spans, for span/joint models (target extraction + polarity).
* ``instruct_atsc_{split}.jsonl``: chat-format examples for LLM fine-tuning,
  targets given, model returns a polarity per target.
* ``instruct_e2e_{split}.jsonl`` (SEntFiN only): model must find the entities
  as well. Not produced for FiQA, whose annotations do not cover every entity
  in a sentence.
* ``instruct_aspect_{split}.jsonl``: targets given, model returns aspect
  category + sentiment per target (aspect-category sentiment analysis).
* ``instruct_e2e_aspect_{split}.jsonl`` (SEntFiN only): full (target, aspect,
  sentiment) triplet extraction.

The two aspect tasks only contain texts whose targets all have an aspect label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .schema import PAIR_COLUMNS, SPLITS
from .taxonomy import LABELS as ASPECT_LABELS

SYSTEM_PROMPT = "You are a financial analyst who labels the sentiment of financial news and posts."
_KIND = {"headline": "headline", "post": "social media post"}


def _clean(v):
    if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    return v


def _write_jsonl(records, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _records(df: pd.DataFrame) -> list[dict]:
    cols = list(df.columns)
    return [dict(zip(cols, map(_clean, row))) for row in df.itertuples(index=False, name=None)]


def _grouped(df: pd.DataFrame):
    """Yield (text_id, pairs) in order of first appearance; pairs ordered by position in
    the text, targets without a span last (stable, so annotation order breaks ties)."""
    groups: dict[str, list[dict]] = {}
    for r in _records(df):
        groups.setdefault(r["text_id"], []).append(r)
    for text_id, rs in groups.items():
        rs.sort(key=lambda r: (r["start"] is None, r["start"] or 0))
        yield text_id, rs


_PAIR_FIELDS = ("uid", "source", "text_id", "text_type", "text", "target", "start", "end",
                "polarity", "label", "label_type", "p_negative", "p_neutral", "p_positive", "label_confidence",
                "score", "polarity_finben", "aspect", "aspect_l1", "aspect_source", "aspect_confidence",
                "fiqa_aspect", "entity_seen_in_train", "quality_flags")
_TARGET_FIELDS = ("target", "start", "end", "polarity", "label_type", "p_negative", "p_neutral", "p_positive",
                  "aspect", "aspect_source", "score", "polarity_finben", "fiqa_aspect")


def _pack_soft(r: dict) -> dict:
    """p_negative/p_neutral/p_positive -> soft_label: [neg, neu, pos] (same order as `label`)."""
    if "p_negative" in r:
        r["soft_label"] = [r.pop("p_negative"), r.pop("p_neutral"), r.pop("p_positive")]
    return r
_ASPECT_LIST = "\n".join(f"- {a}" for a in ASPECT_LABELS)


def pair_records(df: pd.DataFrame):
    for r in _records(df[[c for c in _PAIR_FIELDS if c in df]]):
        yield _pack_soft(r)


def sentence_records(df: pd.DataFrame):
    for text_id, rs in _grouped(df):
        yield {
            "text_id": text_id,
            "source": rs[0]["source"],
            "text_type": rs[0]["text_type"],
            "text": rs[0]["text"],
            "targets": [_pack_soft({k: r[k] for k in _TARGET_FIELDS if k in r}) for r in rs],
        }


def instruct_records(df: pd.DataFrame, task: str):
    with_aspect = task in ("aspect", "e2e_aspect")
    for text_id, rs in _grouped(df):
        if with_aspect and any(r["aspect"] is None for r in rs):
            continue
        first = rs[0]
        kind = _KIND.get(first["text_type"], "text")
        text = f"Text: {first['text']}\n"
        targets = f"Targets: {json.dumps([r['target'] for r in rs], ensure_ascii=False)}\n\n"
        if with_aspect:
            answer = [{"target": r["target"], "aspect": r["aspect"], "sentiment": r["polarity"]} for r in rs]
        else:
            answer = [{"target": r["target"], "sentiment": r["polarity"]} for r in rs]
        if task == "atsc":
            user = (f"Classify the sentiment expressed towards each target in the financial {kind} below "
                    "as positive, negative or neutral.\n\n" + text + targets +
                    'Answer with a JSON list of objects with keys "target" and "sentiment".')
        elif task == "e2e":
            user = ("List every financial entity (company, index, commodity, currency, sector, regulator, ...) "
                    f"mentioned in the {kind} below and the sentiment expressed towards it "
                    "(positive, negative or neutral).\n\n" + text + "\n" +
                    'Answer with a JSON list of objects with keys "target" and "sentiment", in order of appearance.')
        elif task == "aspect":
            user = (f"For each target in the financial {kind} below, give the aspect category the text "
                    "discusses about it and the sentiment expressed towards it (positive, negative or "
                    f"neutral).\n\nAspect categories:\n{_ASPECT_LIST}\n\n" + text + targets +
                    'Answer with a JSON list of objects with keys "target", "aspect" and "sentiment".')
        elif task == "e2e_aspect":
            user = ("List every financial entity (company, index, commodity, currency, sector, regulator, ...) "
                    f"mentioned in the {kind} below, the aspect category the text discusses about it and the "
                    "sentiment expressed towards it (positive, negative or neutral).\n\n"
                    f"Aspect categories:\n{_ASPECT_LIST}\n\n" + text + "\n" +
                    'Answer with a JSON list of objects with keys "target", "aspect" and "sentiment", '
                    "in order of appearance.")
        else:
            raise ValueError(task)
        conf = [r.get("label_confidence") for r in rs if r.get("label_confidence") is not None]
        yield {
            "id": text_id,
            # Lowest sentiment-label confidence among the targets: SFT trainers ignore it; use it to
            # weight or filter ambiguous (mostly near-threshold FiQA) examples.
            "label_confidence": min(conf) if conf else None,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
        }


def export_dataset(pairs: pd.DataFrame, out_dir: str | Path, e2e: bool) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs[PAIR_COLUMNS]
    pairs.to_parquet(out / "pairs.parquet", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    counts = {}
    for split in SPLITS:
        part = pairs[pairs["split"] == split]
        counts[f"pairs_{split}"] = _write_jsonl(pair_records(part), out / f"pairs_{split}.jsonl")
        counts[f"sentences_{split}"] = _write_jsonl(sentence_records(part), out / f"sentences_{split}.jsonl")
        counts[f"instruct_atsc_{split}"] = _write_jsonl(instruct_records(part, "atsc"), out / f"instruct_atsc_{split}.jsonl")
        tasks = ["e2e"] if e2e else []
        if pairs["aspect"].notna().any():
            tasks += ["aspect"] + (["e2e_aspect"] if e2e else [])
        for task in tasks:
            counts[f"instruct_{task}_{split}"] = _write_jsonl(instruct_records(part, task), out / f"instruct_{task}_{split}.jsonl")
    return counts
