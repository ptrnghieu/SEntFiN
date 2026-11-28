"""Direction 1: one multi-task model for (target, aspect, sentiment) triplets.

    text -> shared encoder (--model: a transformer, or `lstm`)
              |-> BIO tagger + CRF                              -> target spans
              '-> per span: [first; last; mean; width] + span-to-context attention
                     |-> sentiment head (3, soft labels with --soft)
                     '-> aspect head (21)

Training: loss = CRF + lambda_s * sentiment + lambda_a * aspect, all three back-propagated into
the shared encoder; the span heads learn on the gold spans (teacher forcing). Prediction: the heads
label the spans the CRF finds. Model selection on valid triplet F1; test scored with
absa_models.triplets, so it compares directly with the pipeline baseline (absa_models.pipeline).

    python -m absa_models.multitask                                # FinBERT, FinEntity + SEntFiN + FiQA
    python -m absa_models.multitask --model roberta-base --soft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import lstm
from .common import (ALL_SOURCES, SOURCES, device, load_encoder, load_table, load_tokenizer, model_tag, set_seed,
                     train_loop, write_results)
from .pair_classify import ASPECTS, SENTIMENTS
from .target_extraction import _special_mask
from .triplets import COMPLETE, score

TAGS = ("O", "B", "I")
POL_COLS = {"negative": "p_negative", "neutral": "p_neutral", "positive": "p_positive"}


# ----------------------------------------------------------------------------- data

def build_examples(texts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """One example per complete-target text, with its gold (start, end, sentiment probs, aspect)."""
    texts = texts[texts["target_annotation"].isin(COMPLETE)]
    pairs = pairs[pairs["start"].notna() & pairs["text_id"].isin(set(texts["text_id"]))]
    spans: dict[str, list] = {}
    for r in pairs.itertuples(index=False):
        probs = [getattr(r, POL_COLS[s]) for s in SENTIMENTS]
        if any(pd.isna(p) for p in probs):
            probs = [1.0 if s == r.polarity else 0.0 for s in SENTIMENTS]
        spans.setdefault(r.text_id, []).append(
            (int(r.start), int(r.end), r.polarity, [float(p) for p in probs],
             ASPECTS.index(r.aspect) if r.aspect in ASPECTS else -100))
    return [{"text_id": r.text_id, "source": r.source, "text": r.text, "spans": spans.get(r.text_id, [])}
            for r in texts.itertuples(index=False)]


def encode(tok, examples: list[dict], max_length: int) -> list[dict]:
    out = []
    for ex in examples:
        enc = tok(ex["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        offs, special = enc["offset_mapping"], _special_mask(enc, tok)
        labels = [-100 if sp or a == b else 0 for (a, b), sp in zip(offs, special)]
        gold = []
        for s, e, pol, probs, asp in sorted(ex["spans"]):
            idx = [i for i, ((a, b), lab) in enumerate(zip(offs, labels)) if lab != -100 and a < e and s < b]
            if not idx:  # truncated away
                continue
            i, j = idx[0], idx[-1]
            if any(labels[k] != 0 for k in range(i, j + 1)):  # overlaps an earlier span: keep the first
                continue
            labels[i] = 1
            for k in range(i + 1, j + 1):
                if labels[k] != -100:
                    labels[k] = 2
            gold.append((i, j, probs, asp))
        out.append({**ex, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "offsets": offs,
                    "labels": labels, "gold": gold})
    return out


def batches(items: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int = 0):
    order = np.random.default_rng(seed).permutation(len(items)) if shuffle else np.arange(len(items))
    for k in range(0, len(items), batch_size):
        chunk = [items[j] for j in order[k:k + batch_size]]
        n = max(len(x["input_ids"]) for x in chunk)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        spans = [(b, i, j) for b, x in enumerate(chunk) for i, j, _, _ in x["gold"]]
        yield {"input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in chunk]),
               "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in chunk]),
               "labels": torch.tensor([pad(x["labels"], -100) for x in chunk]),
               "spans": torch.tensor(spans, dtype=torch.long).reshape(-1, 3),
               "span_sentiment": torch.tensor([p for x in chunk for _, _, p, _ in x["gold"]],
                                              dtype=torch.float32).reshape(-1, len(SENTIMENTS)),
               "span_aspect": torch.tensor([a for x in chunk for _, _, _, a in x["gold"]], dtype=torch.long),
               "items": chunk}


# ----------------------------------------------------------------------------- model

class MultiTask(nn.Module):
    def __init__(self, model_name: str, encoder=None, dropout: float = 0.1, width_dim: int = 32, max_width: int = 32,
                 lambda_sentiment: float = 1.0, lambda_aspect: float = 1.0, soft: bool = False):
        super().__init__()
        from torchcrf import CRF
        self.encoder = encoder if encoder is not None else load_encoder(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(h, len(TAGS))
        self.crf = CRF(len(TAGS), batch_first=True)
        self.max_width = max_width
        self.width = nn.Embedding(max_width + 1, width_dim)
        self.span_proj = nn.Sequential(nn.Linear(3 * h + width_dim, h), nn.Tanh())
        self.query = nn.Linear(h, h, bias=False)
        self.sentiment_head = nn.Linear(2 * h, len(SENTIMENTS))
        self.aspect_head = nn.Linear(2 * h, len(ASPECTS))
        self.lambdas = (lambda_sentiment, lambda_aspect)
        self.soft = soft

    def hidden(self, batch):
        return self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

    def _mask(self, labels):
        mask = labels != -100
        mask[:, 0] = True  # the CRF needs position 0; [CLS] (or the first word) is an 'O' anchor
        return mask

    def span_repr(self, h, mask, spans):
        """spans: (n, 3) rows of (batch index, first token, last token) -> (n, 2h)."""
        h = h.float()
        b, i, j = spans[:, 0], spans[:, 1], spans[:, 2]
        pos = torch.arange(h.size(1), device=h.device)
        inside = (pos[None, :] >= i[:, None]) & (pos[None, :] <= j[:, None])
        mean = (h[b] * inside[..., None]).sum(1) / inside.sum(1, keepdim=True).clamp(min=1)
        width = self.width((j - i).clamp(max=self.max_width))
        span = self.span_proj(torch.cat([h[b, i], h[b, j], mean, width], -1))
        scores = torch.einsum("nd,ntd->nt", self.query(span), h[b]).masked_fill(~mask[b], -1e4)
        ctx = (torch.softmax(scores, -1)[..., None] * h[b]).sum(1)
        return self.dropout(torch.cat([span, ctx], -1))

    def loss(self, batch):
        h = self.dropout(self.hidden(batch))
        labels = batch["labels"]
        mask = self._mask(labels)
        loss = -self.crf(self.tag_head(h).float(), labels.clamp(min=0), mask=mask, reduction="mean")
        if len(batch["spans"]):
            r = self.span_repr(h, batch["attention_mask"].bool(), batch["spans"])
            logp = torch.log_softmax(self.sentiment_head(r), -1)
            target = batch["span_sentiment"]
            if not self.soft:
                target = torch.nn.functional.one_hot(target.argmax(-1), len(SENTIMENTS)).float()
            loss = loss + self.lambdas[0] * -(target * logp).sum(-1).mean()
            asp = batch["span_aspect"]
            if (asp != -100).any():
                loss = loss + self.lambdas[1] * nn.functional.cross_entropy(self.aspect_head(r), asp, ignore_index=-100)
        return loss

    @torch.no_grad()
    def predict(self, batch) -> list[list[dict]]:
        h = self.hidden(batch)
        mask = self._mask(batch["labels"])
        paths = self.crf.decode(self.tag_head(h).float(), mask=mask)
        out, rows = [], []
        for b, (path, m, item) in enumerate(zip(paths, mask.tolist(), batch["items"])):
            full, it = [], iter(path)
            for keep in m:
                full.append(next(it) if keep else 0)
            spans, cur = [], None
            for t, (tag, gold) in enumerate(zip(full, item["labels"])):
                if gold == -100:
                    continue
                if tag == 1 or (tag == 2 and cur is None):
                    if cur:
                        spans.append(cur)
                    cur = [t, t]
                elif tag == 2:
                    cur[1] = t
                else:
                    if cur:
                        spans.append(cur)
                    cur = None
            if cur:
                spans.append(cur)
            out.append([])
            rows += [(b, i, j) for i, j in spans]
            for i, j in spans:
                out[-1].append({"i": i, "j": j})
        if rows:
            r = self.span_repr(h, batch["attention_mask"].bool(), torch.tensor(rows, device=h.device))
            ps, pa = torch.softmax(self.sentiment_head(r), -1), torch.softmax(self.aspect_head(r), -1)
            k = 0
            for b, spans in enumerate(out):
                item = batch["items"][b]
                for sp in spans:
                    (a, _), (_, e) = item["offsets"][sp["i"]], item["offsets"][sp["j"]]
                    sp.update(start=a, end=e, target=item["text"][a:e],
                              sentiment=SENTIMENTS[int(ps[k].argmax())], sentiment_confidence=float(ps[k].max()),
                              aspect=ASPECTS[int(pa[k].argmax())], aspect_confidence=float(pa[k].max()))
                    del sp["i"], sp["j"]
                    k += 1
        return out


def predict_all(model, items, batch_size, pad_id) -> dict[str, list[dict]]:
    dev = device()
    model.eval()
    out = {}
    for b in batches(items, batch_size, pad_id, shuffle=False):
        bt = {k: v.to(dev) for k, v in b.items() if torch.is_tensor(v)}
        bt["items"] = b["items"]
        for item, trips in zip(b["items"], model.predict(bt)):
            out[item["text_id"]] = trips
    return out


def gold_of(items: list[dict]) -> dict[str, list[dict]]:
    return {it["text_id"]: [{"start": s, "end": e, "sentiment": pol,
                             "aspect": ASPECTS[a] if a != -100 else None}
                            for s, e, pol, _, a in it["spans"]] for it in items}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--train-sources", nargs="+", default=list(SOURCES), choices=ALL_SOURCES)
    ap.add_argument("--eval-sources", nargs="+", default=list(SOURCES), choices=ALL_SOURCES)
    ap.add_argument("--soft", action="store_true", help="train the sentiment head on the soft labels")
    ap.add_argument("--lambda-sentiment", type=float, default=1.0)
    ap.add_argument("--lambda-aspect", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, help="default 10 (lstm: 30)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, help="default 3e-5 (lstm: 1e-3)")
    ap.add_argument("--head-lr", type=float, help="LR for tagger, CRF and span heads (default: --lr)")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--patience", type=int, help="default 3 (lstm: 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use at most N texts per split (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    is_lstm = args.model == lstm.MODEL_NAME
    args.lr = args.lr or (lstm.DEFAULTS["lr"] if is_lstm else 3e-5)
    args.epochs = args.epochs or (lstm.DEFAULTS["tagging_epochs"] if is_lstm else 10)
    args.patience = args.patience or (lstm.DEFAULTS["tagging_patience"] if is_lstm else 3)
    set_seed(args.seed)

    processed = Path(args.processed)
    srcs = sorted(set(args.train_sources) | set(args.eval_sources))
    texts, pairs = load_table(processed, "texts", srcs), load_table(processed, "pairs", srcs)

    def split(name, sources):
        t = texts[(texts["split"] == name) & texts["source"].isin(sources)]
        ex = build_examples(t, pairs)
        return ex[: args.limit] if args.limit else ex

    ex_tr, ex_va, ex_te = split("train", args.train_sources), split("valid", args.eval_sources), \
        split("test", args.eval_sources)
    encoder = None
    if is_lstm:
        tok, encoder = lstm.build([e["text"] for e in ex_tr], [e["text"] for e in ex_va + ex_te], seed=args.seed)
    else:
        tok = load_tokenizer(args.model)
    train, valid, test = (encode(tok, e, args.max_length) for e in (ex_tr, ex_va, ex_te))
    print(f"train {len(train)} / valid {len(valid)} / test {len(test)} texts; "
          f"{sum(len(x['gold']) for x in train)} train spans")
    model = MultiTask(args.model, encoder, lambda_sentiment=args.lambda_sentiment,
                      lambda_aspect=args.lambda_aspect, soft=args.soft)
    out = Path(args.out or f"runs/multitask/{model_tag(args.model)}/{'+'.join(args.train_sources)}"
                           f"{'-soft' if args.soft else ''}")
    if is_lstm:
        lstm.save_vocab(tok, out)
    pad = tok.pad_token_id
    va_texts = pd.DataFrame([{"text_id": e["text_id"], "source": e["source"]} for e in ex_va])
    te_texts = pd.DataFrame([{"text_id": e["text_id"], "source": e["source"]} for e in ex_te])
    epoch = {"n": 0}

    def train_batches():
        epoch["n"] += 1
        return batches(train, args.batch_size, pad, shuffle=True, seed=args.seed + epoch["n"])

    def val():
        res = score(va_texts, gold_of(ex_va), predict_all(model, valid, args.batch_size, pad))
        return res["all triplet"]["f1"], res

    log = train_loop(model, train_batches, val, epochs=args.epochs, lr=args.lr, head_lr=args.head_lr,
                     patience=args.patience, out=out)
    pred = predict_all(model, test, args.batch_size, pad)
    test_res = score(te_texts, gold_of(ex_te), pred)
    write_results(out, {"config": {**vars(args), **log}, "valid": val()[1], "test": test_res},
                  "Multi-task triplet model")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for tid, trips in pred.items():
            f.write(json.dumps({"text_id": tid, "triplets": trips}, ensure_ascii=False) + "\n")
    for k in ("finentity span", "finentity span+sentiment", "finentity triplet", "all triplet"):
        if k in test_res:
            print(f"test {k}: {test_res[k]}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
