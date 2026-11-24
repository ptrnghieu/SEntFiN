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


def _grouped(df: pd.DataFrame, texts: pd.DataFrame | None = None):
    """Yield (text record, pairs) per text; pairs ordered by position in the text (stable).

    With `texts`, every text is yielded, including those without any target (FinEntity
    sentences without an entity: true negatives for extraction)."""
    groups: dict[str, list[dict]] = {}
    for r in _records(df):
        groups.setdefault(r["text_id"], []).append(r)
    for rs in groups.values():
        rs.sort(key=lambda r: (r["start"] is None, r["start"] or 0))
    if texts is None:
        for text_id, rs in groups.items():
            yield {"text_id": text_id, "source": rs[0]["source"], "text_type": rs[0]["text_type"],
                   "text": rs[0]["text"], "target_annotation": rs[0].get("target_annotation")}, rs
    else:
        for t in _records(texts):
            yield t, groups.get(t["text_id"], [])


_PAIR_FIELDS = ("uid", "source", "text_id", "text_type", "text", "target", "start", "end",
                "polarity", "label", "label_type", "p_negative", "p_neutral", "p_positive", "label_confidence",
                "score", "polarity_finben", "aspect", "aspect_l1", "aspect_source", "aspect_confidence",
                "fiqa_aspect", "target_source", "sentiment_source", "target_type", "target_annotation",
                "paragraph_id", "entity_seen_in_train", "quality_flags")
_TARGET_FIELDS = ("target", "start", "end", "polarity", "label_type", "p_negative", "p_neutral", "p_positive",
                  "aspect", "aspect_source", "target_source", "sentiment_source", "target_type", "score",
                  "polarity_finben", "fiqa_aspect")


def _pack_soft(r: dict) -> dict:
    """p_negative/p_neutral/p_positive -> soft_label: [neg, neu, pos] (same order as `label`)."""
    if "p_negative" in r:
        r["soft_label"] = [r.pop("p_negative"), r.pop("p_neutral"), r.pop("p_positive")]
    return r
_ASPECT_LIST = "\n".join(f"- {a}" for a in ASPECT_LABELS)


def pair_records(df: pd.DataFrame):
    for r in _records(df[[c for c in _PAIR_FIELDS if c in df]]):
        yield _pack_soft(r)


def sentence_records(df: pd.DataFrame, texts: pd.DataFrame | None = None):
    for t, rs in _grouped(df, texts):
        yield {
            "text_id": t["text_id"],
            "source": t["source"],
            "text_type": t["text_type"],
            "text": t["text"],
            "paragraph_id": t.get("paragraph_id"),
            "sent_idx": t.get("sent_idx"),
            "target_annotation": t.get("target_annotation"),
            "targets": [_pack_soft({k: r[k] for k in _TARGET_FIELDS if k in r}) for r in rs],
        }


def instruct_records(df: pd.DataFrame, task: str, texts: pd.DataFrame | None = None):
    """End-to-end tasks (e2e, e2e_aspect) only use texts whose targets are exhaustive or LLM-completed:
    with partial gold the answer would teach the model to skip real targets."""
    with_aspect = task in ("aspect", "e2e_aspect")
    e2e = task.startswith("e2e")
    for t, rs in _grouped(df, texts if e2e else None):
        text_id = t["text_id"]
        if e2e and t.get("target_annotation") not in ("exhaustive", "llm_completed"):
            continue
        if not e2e and not rs:
            continue
        if with_aspect and any(r["aspect"] is None for r in rs):
            continue
        kind = _KIND.get(t["text_type"], "text")
        text = f"Text: {t['text']}\n"
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


def export_dataset(pairs: pd.DataFrame, out_dir: str | Path, texts: pd.DataFrame | None = None) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs[[c for c in PAIR_COLUMNS if c in pairs]]
    pairs.to_parquet(out / "pairs.parquet", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    if texts is not None:
        texts.to_parquet(out / "texts.parquet", index=False)
    complete = texts is not None and texts["target_annotation"].isin(["exhaustive", "llm_completed"]).any()
    has_aspect = pairs["aspect"].notna().any()
    tasks = ["atsc"] + (["e2e"] if complete else []) + (["aspect"] if has_aspect else []) + (
        ["e2e_aspect"] if complete and has_aspect else [])
    counts = {}
    for split in SPLITS:
        part = pairs[pairs["split"] == split]
        tpart = texts[texts["split"] == split] if texts is not None else None
        counts[f"pairs_{split}"] = _write_jsonl(pair_records(part), out / f"pairs_{split}.jsonl")
        counts[f"sentences_{split}"] = _write_jsonl(sentence_records(part, tpart), out / f"sentences_{split}.jsonl")
        for task in tasks:
            counts[f"instruct_{task}_{split}"] = _write_jsonl(instruct_records(part, task, tpart),
                                                              out / f"instruct_{task}_{split}.jsonl")
    return counts
