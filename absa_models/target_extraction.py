"""Step 1a: target extraction as token classification (FinBERT [+ CRF]).

Schemes:
  bio            B/I/O over target spans (the extraction head of the triplet model)
  bio-sentiment  B/I-{positive,neutral,negative}/O: FinEntity's own setup, to reproduce its
                 published FinBERT-CRF result (entity-level macro-F1 0.85, Tang et al. 2023)

Training texts must have complete targets: FinEntity (exhaustive) and the LLM-completed
SEntFiN / FiQA texts. Evaluate extraction on FinEntity: the other sources' gold is partly
LLM-generated.

    python -m absa_models.target_extraction --train-sources finentity --crf --scheme bio-sentiment
    python -m absa_models.target_extraction --train-sources finentity sentfin fiqa --crf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import SOURCES, load_encoder, load_tokenizer, load_table, set_seed, span_metrics, train_loop, write_results

POLARITIES = ("negative", "neutral", "positive")


def tag_set(scheme: str) -> list[str]:
    if scheme == "bio":
        return ["O", "B", "I"]
    return ["O"] + [f"{p}-{s}" for s in POLARITIES for p in ("B", "I")]


def build_examples(texts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """One example per text with its spans; only texts whose targets are complete."""
    ok = texts["target_annotation"].isin(["exhaustive", "llm_completed"])
    spans = (pairs.dropna(subset=["start"]).groupby("text_id")
             .apply(lambda g: [(int(s), int(e), p) for s, e, p in zip(g["start"], g["end"], g["polarity"])],
                    include_groups=False).to_dict())
    return [{"text_id": r.text_id, "source": r.source, "text": r.text, "spans": spans.get(r.text_id, [])}
            for r in texts[ok].itertuples(index=False)]


def encode(tokenizer, examples: list[dict], tags: list[str], scheme: str, max_length: int) -> list[dict]:
    tid = {t: i for i, t in enumerate(tags)}
    out = []
    for ex in examples:
        enc = tokenizer(ex["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        labels = []
        for (a, b), special in zip(enc["offset_mapping"], _special_mask(enc, tokenizer)):
            if special or a == b:
                labels.append(-100)
                continue
            lab = "O"
            for s, e, pol in ex["spans"]:
                if a < e and s < b:  # token overlaps the span
                    prefix = "B" if a <= s else "I"
                    lab = prefix if scheme == "bio" else f"{prefix}-{pol}"
                    break
            labels.append(tid[lab])
        out.append({**ex, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                    "offsets": enc["offset_mapping"], "labels": labels})
    return out


def _special_mask(enc, tokenizer):
    ids = enc["input_ids"]
    special = set(tokenizer.all_special_ids)
    return [i in special for i in ids]


def batches(items: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int = 0):
    order = np.random.default_rng(seed).permutation(len(items)) if shuffle else np.arange(len(items))
    for i in range(0, len(items), batch_size):
        chunk = [items[j] for j in order[i:i + batch_size]]
        n = max(len(x["input_ids"]) for x in chunk)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        yield {"input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in chunk]),
               "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in chunk]),
               "labels": torch.tensor([pad(x["labels"], -100) for x in chunk]), "items": chunk}


class Tagger(nn.Module):
    def __init__(self, model_name: str, n_tags: int, crf: bool, dropout: float = 0.1):
        super().__init__()
        self.encoder = load_encoder(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_tags)
        self.crf = None
        if crf:
            from torchcrf import CRF
            self.crf = CRF(n_tags, batch_first=True)

    def emissions(self, batch):
        h = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        return self.head(self.dropout(h))

    def loss(self, batch):
        em = self.emissions(batch)
        labels = batch["labels"]
        if self.crf is None:
            return nn.functional.cross_entropy(em.reshape(-1, em.size(-1)), labels.reshape(-1), ignore_index=-100)
        # CRF runs over the real word-piece positions only ([CLS] stays as an 'O' anchor at position 0).
        mask = labels != -100
        mask[:, 0] = True
        tags = labels.clamp(min=0)
        return -self.crf(em.float(), tags, mask=mask, reduction="mean")

    @torch.no_grad()
    def predict(self, batch) -> list[list[int]]:
        em = self.emissions(batch)
        if self.crf is None:
            return em.argmax(-1).tolist()
        mask = batch["labels"] != -100
        mask[:, 0] = True
        paths = self.crf.decode(em.float(), mask=mask)
        out = []
        for path, m in zip(paths, mask.tolist()):  # decode() returns only masked positions
            full, it = [], iter(path)
            for keep in m:
                full.append(next(it) if keep else 0)
            out.append(full)
        return out


def decode_spans(item: dict, tag_ids: list[int], tags: list[str]) -> set:
    """Tag sequence -> set of (start, end, polarity|None) character spans."""
    spans, cur = [], None
    for (a, b), lab_id, gold in zip(item["offsets"], tag_ids, item["labels"]):
        if gold == -100:
            continue
        lab = tags[lab_id]
        prefix, _, pol = lab.partition("-")
        if prefix == "B" or (prefix == "I" and (cur is None or (pol or None) != cur[2])):
            if cur:
                spans.append(cur)
            cur = [a, b, pol or None]
        elif prefix == "I":
            cur[1] = b
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return {tuple(s) for s in spans}


def evaluate(model, items, tags, batch_size, pad_id, dev) -> tuple[dict, list[dict]]:
    model.eval()
    preds = []
    for b in batches(items, batch_size, pad_id, shuffle=False):
        bt = {k: v.to(dev) for k, v in b.items() if torch.is_tensor(v)}
        for item, ids in zip(b["items"], model.predict(bt)):
            preds.append({"text_id": item["text_id"], "source": item["source"], "text": item["text"],
                          "gold": sorted((s, e, p) for s, e, p in item["spans"]),
                          "pred": sorted(decode_spans(item, ids, tags), key=lambda x: x[0])})
    res = {}
    for src in ["all", *sorted({p["source"] for p in preds})]:
        sub = [p for p in preds if src == "all" or p["source"] == src]
        gold = [{(s, e) for s, e, _ in p["gold"]} for p in sub]
        pred = [{(s, e) for s, e, _ in p["pred"]} for p in sub]
        res[f"{src} spans"] = span_metrics(gold, pred)
        if tags != ["O", "B", "I"]:
            goldl = [{(s, e, pol) for s, e, pol in p["gold"]} for p in sub]
            predl = [set(p["pred"]) for p in sub]
            res[f"{src} spans+sentiment"] = span_metrics(goldl, predl)
            per = {pol: span_metrics([{x for x in g if x[2] == pol} for g in goldl],
                                     [{x for x in q if x[2] == pol} for q in predl])["f1"] for pol in POLARITIES}
            res[f"{src} spans+sentiment"]["macro_f1"] = round(float(np.mean(list(per.values()))), 4)
    return res, preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--train-sources", nargs="+", default=["finentity"], choices=SOURCES)
    ap.add_argument("--eval-sources", nargs="+", default=["finentity"], choices=SOURCES)
    ap.add_argument("--scheme", choices=["bio", "bio-sentiment"], default="bio")
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, help="LR for the head / CRF (default: --lr)")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use at most N texts per split (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    set_seed(args.seed)

    processed = Path(args.processed)
    srcs = sorted(set(args.train_sources) | set(args.eval_sources))
    texts, pairs = load_table(processed, "texts", srcs), load_table(processed, "pairs", srcs)
    tok = load_tokenizer(args.model)
    tags = tag_set(args.scheme)

    def split(name, sources):
        t = texts[(texts["split"] == name) & texts["source"].isin(sources)]
        ex = build_examples(t, pairs)
        return encode(tok, ex[: args.limit] if args.limit else ex, tags, args.scheme, args.max_length)

    train, valid, test = split("train", args.train_sources), split("valid", args.eval_sources), split("test", args.eval_sources)
    print(f"train {len(train)} / valid {len(valid)} / test {len(test)} texts; tags {tags}")
    model = Tagger(args.model, len(tags), args.crf)
    out = Path(args.out or f"runs/target_extraction/{args.scheme}{'-crf' if args.crf else ''}-{'+'.join(args.train_sources)}")
    from .common import device
    dev = device()
    key = "all spans+sentiment" if args.scheme == "bio-sentiment" else "all spans"
    score_key = "macro_f1" if args.scheme == "bio-sentiment" else "f1"
    epoch = {"n": 0}

    def train_batches():
        epoch["n"] += 1
        return batches(train, args.batch_size, tok.pad_token_id, shuffle=True, seed=args.seed + epoch["n"])

    def val():
        res, _ = evaluate(model, valid, tags, args.batch_size, tok.pad_token_id, dev)
        return res[key][score_key], res

    log = train_loop(model, train_batches, val, epochs=args.epochs, lr=args.lr, head_lr=args.head_lr, patience=args.patience, out=out)
    test_res, preds = evaluate(model, test, tags, args.batch_size, tok.pad_token_id, dev)
    valid_res, _ = evaluate(model, valid, tags, args.batch_size, tok.pad_token_id, dev)
    write_results(out, {"config": {**vars(args), **log}, "valid": valid_res, "test": test_res},
                  f"Target extraction ({args.scheme}{' + CRF' if args.crf else ''})")
    pd.DataFrame(preds).to_json(out / "test_predictions.jsonl", orient="records", lines=True)
    print(f"test {key}: {test_res[key]}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
