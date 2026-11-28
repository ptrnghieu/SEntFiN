"""Pipeline baseline for triplets: 1a target extraction -> 1b sentiment -> 1c aspect.

Chains three trained runs (each a directory with ``metrics.json`` and ``best.pt``) on raw test
texts and scores the triplets with ``absa_models.triplets``:

    python -m absa_models.pipeline \\
        --extractor runs/target_extraction/finbert-pretrain/bio-crf-finentity+sentfin+fiqa \\
        --sentiment runs/sentiment/finbert-pretrain/sentfin+fiqa+finentity-soft-cw \\
        --aspect    runs/aspect/finbert-pretrain/sentfin+fiqa+finentity \\
        --out runs/pipeline/finbert-pretrain

Without ``--sentiment``, a ``bio-sentiment`` extractor's own polarity is used. Errors compound:
a target the extractor misses or cuts wrongly cannot be right at the triplet level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import lstm
from .common import ALL_SOURCES, device, load_tokenizer, write_results
from .pair_classify import ASPECTS, MARK_CLOSE, MARK_OPEN, SENTIMENTS, PairClassifier
from .pair_classify import encode as encode_pairs
from .pair_classify import predict as predict_pairs
from .target_extraction import Tagger, evaluate, tag_set
from .target_extraction import encode as encode_tags
from .triplets import load_gold, score


def _config(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text())["config"]


def _state(run: Path):
    return torch.load(run / "best.pt", map_location="cpu")


def load_extractor(run: Path):
    cfg = _config(run)
    tags = tag_set(cfg["scheme"])
    if cfg["model"] == lstm.MODEL_NAME:
        tok, enc = lstm.from_saved(run)
    else:
        tok, enc = load_tokenizer(cfg["model"]), None
    model = Tagger(cfg["model"], len(tags), cfg["crf"], encoder=enc)
    model.load_state_dict(_state(run))
    return model.to(device()).eval(), tok, tags, cfg


def load_classifier(run: Path):
    cfg = _config(run)
    labels = SENTIMENTS if cfg["task"] == "sentiment" else ASPECTS
    if cfg["model"] == lstm.MODEL_NAME:
        tok, enc = lstm.from_saved(run)
    else:
        tok, enc = load_tokenizer(cfg["model"]), None
        tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    model = PairClassifier(cfg["model"], len(labels), len(tok), encoder=enc)
    model.load_state_dict(_state(run))
    return model.to(device()).eval(), tok, labels, cfg


def extract(texts: pd.DataFrame, extractor_run: Path, batch_size: int = 32) -> pd.DataFrame:
    model, tok, tags, cfg = load_extractor(extractor_run)
    examples = [{"text_id": r.text_id, "source": r.source, "text": r.text, "spans": []}
                for r in texts.itertuples(index=False)]
    items = encode_tags(tok, examples, tags, cfg["scheme"], cfg.get("max_length", 128))
    _, preds = evaluate(model, items, tags, batch_size, tok.pad_token_id, device())
    text = dict(zip(texts["text_id"], texts["text"]))
    rows = [{"text_id": p["text_id"], "text": text[p["text_id"]], "start": s, "end": e,
             "target": text[p["text_id"]][s:e], "tagger_sentiment": pol}
            for p in preds for s, e, pol in p["pred"]]
    return pd.DataFrame(rows, columns=["text_id", "text", "start", "end", "target", "tagger_sentiment"])


def classify(spans: pd.DataFrame, run: Path, batch_size: int = 64) -> tuple[list[str], np.ndarray]:
    model, tok, labels, cfg = load_classifier(run)
    col = "polarity" if cfg["task"] == "sentiment" else "aspect"
    df = spans.assign(**{col: labels[0]})  # placeholder label: encode() needs one
    data = encode_pairs(tok, df, cfg["task"], False, cfg.get("max_length", 128))
    probs = predict_pairs(model, data, np.arange(len(df)), batch_size, tok.pad_token_id)
    return [labels[i] for i in probs.argmax(1)], probs.max(1)


def run_pipeline(texts: pd.DataFrame, extractor: Path, sentiment: Path | None, aspect: Path | None) -> dict:
    spans = extract(texts, extractor)
    if len(spans) and sentiment is not None:
        spans["sentiment"], spans["sentiment_confidence"] = classify(spans, sentiment)
    else:
        spans["sentiment"] = spans["tagger_sentiment"]
    if len(spans) and aspect is not None:
        spans["aspect"], spans["aspect_confidence"] = classify(spans, aspect)
    else:
        spans["aspect"] = None
    out: dict[str, list[dict]] = {t: [] for t in texts["text_id"]}
    keep = [c for c in ("target", "start", "end", "sentiment", "aspect", "sentiment_confidence", "aspect_confidence")
            if c in spans]
    for r in spans[["text_id"] + keep].to_dict("records"):
        tid = r.pop("text_id")
        out[tid].append({k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                         for k, v in r.items()})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", required=True, type=Path)
    ap.add_argument("--sentiment", type=Path)
    ap.add_argument("--aspect", type=Path)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--sources", nargs="+", default=["finentity", "sentfin", "fiqa"], choices=ALL_SOURCES)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    texts, gold = load_gold(Path(args.processed), args.sources, args.split)
    pred = run_pipeline(texts, args.extractor, args.sentiment, args.aspect)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for tid, trips in pred.items():
            f.write(json.dumps({"text_id": tid, "triplets": trips}, ensure_ascii=False) + "\n")
    res = score(texts, gold, pred)
    cfg = {k: str(v) for k, v in vars(args).items()}
    write_results(args.out, {"config": cfg, args.split: res}, "Pipeline triplets (1a -> 1b -> 1c)")
    for k in ("finentity span", "finentity span+sentiment", "finentity triplet", "all triplet"):
        print(f"{k}: {res[k]}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
