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

Options (each off by default, for ablations):

  --aspect-head hier      (8) two-level aspect head: P(l2) = P(l1) * P(l2 | l1), trained on both levels
  --sentence-context      (10) the aspect head also sees the sentence vector (<s> / [CLS])
  --aspect-confidence     (11) weight each aspect label by the LLM's confidence (high/medium/low)
  --sentiment-loss ordinal (5) ordinal (cumulative-link) sentiment head: negative < neutral < positive
  --markers               (12) levitated markers (PL-Marker, Ye et al. 2022): a <t> </t> pair per span,
                          appended after the text and sharing the span's first/last position ids; the
                          text cannot attend to markers, so the CRF sees the unchanged text, and each span
                          gets a target-aware representation from one extra encoder pass at prediction

    python -m absa_models.multitask                                # FinBERT, FinEntity + SEntFiN + FiQA
    python -m absa_models.multitask --model roberta-base --soft --markers --aspect-head hier
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
from .pair_classify import ASPECTS, MARK_CLOSE, MARK_OPEN, SENTIMENTS
from .target_extraction import _special_mask
from .triplets import COMPLETE, score

TAGS = ("O", "B", "I")
POL_COLS = {"negative": "p_negative", "neutral": "p_neutral", "positive": "p_positive"}
ASPECT_L1 = tuple(sorted({a.split("/")[0] for a in ASPECTS}))
L1_OF = [ASPECT_L1.index(a.split("/")[0]) for a in ASPECTS]
DEFAULT_CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.5, "low": 0.25}


# ----------------------------------------------------------------------------- data

def build_examples(texts: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    """One example per complete-target text, with its gold
    (start, end, sentiment, sentiment probs, aspect index, aspect confidence)."""
    texts = texts[texts["target_annotation"].isin(COMPLETE)]
    pairs = pairs[pairs["start"].notna() & pairs["text_id"].isin(set(texts["text_id"]))]
    has_conf = "aspect_confidence" in pairs
    spans: dict[str, list] = {}
    for r in pairs.itertuples(index=False):
        probs = [getattr(r, POL_COLS[s]) for s in SENTIMENTS]
        if any(pd.isna(p) for p in probs):
            probs = [1.0 if s == r.polarity else 0.0 for s in SENTIMENTS]
        conf = r.aspect_confidence if has_conf and isinstance(r.aspect_confidence, str) else "high"
        spans.setdefault(r.text_id, []).append(
            (int(r.start), int(r.end), r.polarity, [float(p) for p in probs],
             ASPECTS.index(r.aspect) if r.aspect in ASPECTS else -100, conf))
    return [{"text_id": r.text_id, "source": r.source, "text": r.text, "spans": spans.get(r.text_id, [])}
            for r in texts.itertuples(index=False)]


def encode(tok, examples: list[dict], max_length: int) -> list[dict]:
    out = []
    for ex in examples:
        enc = tok(ex["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        offs, special = enc["offset_mapping"], _special_mask(enc, tok)
        labels = [-100 if sp or a == b else 0 for (a, b), sp in zip(offs, special)]
        gold = []
        for s, e, _pol, probs, asp, *rest in sorted(ex["spans"]):
            conf = rest[0] if rest else "high"
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
            gold.append((i, j, probs, asp, conf))
        out.append({**ex, "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "offsets": offs,
                    "labels": labels, "gold": gold})
    return out


def batches(items: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int = 0,
            confidence_weights: dict | None = None):
    order = np.random.default_rng(seed).permutation(len(items)) if shuffle else np.arange(len(items))
    w = confidence_weights or {}
    for k in range(0, len(items), batch_size):
        chunk = [items[j] for j in order[k:k + batch_size]]
        n = max(len(x["input_ids"]) for x in chunk)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        gold = [(b, g) for b, x in enumerate(chunk) for g in x["gold"]]
        yield {"input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in chunk]),
               "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in chunk]),
               "labels": torch.tensor([pad(x["labels"], -100) for x in chunk]),
               "spans": torch.tensor([(b, g[0], g[1]) for b, g in gold], dtype=torch.long).reshape(-1, 3),
               "span_sentiment": torch.tensor([g[2] for _, g in gold], dtype=torch.float32
                                              ).reshape(-1, len(SENTIMENTS)),
               "span_aspect": torch.tensor([g[3] for _, g in gold], dtype=torch.long),
               "span_aspect_weight": torch.tensor([w.get(g[4], 1.0) if len(g) > 4 else 1.0 for _, g in gold],
                                                  dtype=torch.float32),
               "items": chunk}


# ----------------------------------------------------------------------------- levitated markers

def default_positions(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """The position ids the encoder would use for the text alone (RoBERTa counts from padding_idx + 1)."""
    pad_idx = getattr(getattr(encoder, "embeddings", None), "padding_idx", None)
    if getattr(encoder.config, "model_type", "") in ("roberta", "xlm-roberta", "camembert") and pad_idx is not None:
        m = attention_mask.long()
        return torch.cumsum(m, 1) * m + pad_idx
    return torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)


def levitate(encoder, input_ids, attention_mask, spans, open_id: int, close_id: int, pad_id: int):
    """Append one <t> </t> pair per span. Returns (ids, 4D bool mask, position ids, open/close indices).

    Text tokens attend to text only (so their states equal the plain pass); a marker attends to the text
    and to its partner; marker positions copy the span's first / last token positions."""
    B, N = input_ids.shape
    dev = input_ids.device
    per = [[] for _ in range(B)]
    for r, (b, i, j) in enumerate(spans.tolist()):
        per[b].append((r, i, j))
    K = max((len(p) for p in per), default=0)
    T = N + 2 * K
    ids = torch.full((B, T), pad_id, dtype=input_ids.dtype, device=dev)
    ids[:, :N] = input_ids
    pos_text = default_positions(encoder, input_ids, attention_mask)
    pos = torch.zeros((B, T), dtype=pos_text.dtype, device=dev)
    pos[:, :N] = pos_text
    allow = torch.zeros((B, 1, T, T), dtype=torch.bool, device=dev)
    text = attention_mask.bool()
    allow[:, 0, :N, :N] = text[:, None, :] & text[:, :, None]
    allow[:, 0, :N, :N] |= torch.eye(N, dtype=torch.bool, device=dev)  # padded rows attend to themselves
    allow[:, 0, N:, :N] = text[:, None, :]
    idx_open = torch.zeros(len(spans), dtype=torch.long, device=dev)
    idx_close = torch.zeros(len(spans), dtype=torch.long, device=dev)
    for b, p in enumerate(per):
        for k, (r, i, j) in enumerate(p):
            o, c = N + 2 * k, N + 2 * k + 1
            ids[b, o], ids[b, c] = open_id, close_id
            pos[b, o], pos[b, c] = pos_text[b, i], pos_text[b, j]
            allow[b, 0, o, o] = allow[b, 0, o, c] = allow[b, 0, c, o] = allow[b, 0, c, c] = True
            idx_open[r], idx_close[r] = o, c
        for k in range(len(p), K):  # padding marker slots: attend to themselves only
            for q in (N + 2 * k, N + 2 * k + 1):
                allow[b, 0, q, q] = True
    return ids, allow, pos, idx_open, idx_close


# ----------------------------------------------------------------------------- model

class MultiTask(nn.Module):
    def __init__(self, model_name: str, encoder=None, dropout: float = 0.1, width_dim: int = 32, max_width: int = 32,
                 lambda_sentiment: float = 1.0, lambda_aspect: float = 1.0, soft: bool = False,
                 aspect_head: str = "flat", sentence_context: bool = False, sentiment_loss: str = "ce",
                 markers: bool = False, marker_ids: tuple[int, int] | None = None, pad_id: int = 0,
                 vocab_size: int | None = None):
        super().__init__()
        from torchcrf import CRF
        self.encoder = encoder if encoder is not None else load_encoder(model_name)
        if markers:
            if encoder is not None or marker_ids is None:
                raise ValueError("--markers needs a transformer encoder and the marker token ids")
            self.encoder.resize_token_embeddings(vocab_size)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(h, len(TAGS))
        self.crf = CRF(len(TAGS), batch_first=True)
        self.max_width = max_width
        self.width = nn.Embedding(max_width + 1, width_dim)
        self.span_proj = nn.Sequential(nn.Linear(3 * h + width_dim, h), nn.Tanh())
        self.query = nn.Linear(h, h, bias=False)
        rep = 2 * h
        asp_in = rep + (h if sentence_context else 0)
        if sentiment_loss == "ordinal":
            self.sentiment_head = nn.Linear(rep, 1)
            self.theta = nn.Parameter(torch.tensor([-0.5]))   # first cut point
            self.gap = nn.Parameter(torch.tensor([0.5]))      # second = first + softplus(gap)
        else:
            self.sentiment_head = nn.Linear(rep, len(SENTIMENTS))
        self.aspect_head = nn.Linear(asp_in, len(ASPECTS))
        if aspect_head == "hier":
            self.aspect_l1_head = nn.Linear(asp_in, len(ASPECT_L1))
            self.register_buffer("l1_of", torch.tensor(L1_OF))
            self.register_buffer("l1_mask", torch.tensor([[L1_OF[a] == g for a in range(len(ASPECTS))]
                                                          for g in range(len(ASPECT_L1))]))
        self.lambdas = (lambda_sentiment, lambda_aspect)
        self.soft, self.aspect_mode, self.sentence_context = soft, aspect_head, sentence_context
        self.sentiment_mode, self.markers = sentiment_loss, markers
        self.marker_ids, self.pad_id = marker_ids, pad_id

    # -- encoder passes
    def hidden(self, batch):
        return self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

    def hidden_with_markers(self, batch, spans):
        ids, allow, pos, io, ic = levitate(self.encoder, batch["input_ids"], batch["attention_mask"], spans,
                                           *self.marker_ids, self.pad_id)
        h = self.encoder(input_ids=ids, attention_mask=allow, position_ids=pos).last_hidden_state
        return h, io, ic

    def _mask(self, labels):
        mask = labels != -100
        mask[:, 0] = True  # the CRF needs position 0; [CLS] (or the first word) is an 'O' anchor
        return mask

    # -- span representation and heads
    def span_repr(self, h, mask, spans, ends=None):
        """spans: (n, 3) rows of (batch index, first token, last token) -> (n, 2h).
        `ends`: (states at the open marker, states at the close marker) to use instead of the
        first / last token states (levitated markers)."""
        h = h.float()
        b, i, j = spans[:, 0], spans[:, 1], spans[:, 2]
        pos = torch.arange(h.size(1), device=h.device)
        inside = (pos[None, :] >= i[:, None]) & (pos[None, :] <= j[:, None])
        mean = (h[b] * inside[..., None]).sum(1) / inside.sum(1, keepdim=True).clamp(min=1)
        width = self.width((j - i).clamp(max=self.max_width))
        first, last = (h[b, i], h[b, j]) if ends is None else (ends[0].float(), ends[1].float())
        span = self.span_proj(torch.cat([first, last, mean, width], -1))
        scores = torch.einsum("nd,ntd->nt", self.query(span), h[b]).masked_fill(~mask[b], -1e4)
        ctx = (torch.softmax(scores, -1)[..., None] * h[b]).sum(1)
        return self.dropout(torch.cat([span, ctx], -1))

    def sentiment_logprobs(self, r):
        if self.sentiment_mode != "ordinal":
            return torch.log_softmax(self.sentiment_head(r), -1)
        s = self.sentiment_head(r).squeeze(-1)
        t1 = self.theta[0]
        t2 = t1 + nn.functional.softplus(self.gap[0]) + 1e-3
        ge_neutral, ge_positive = torch.sigmoid(s - t1), torch.sigmoid(s - t2)  # P(y >= neu) >= P(y >= pos)
        p = torch.stack([1 - ge_neutral, ge_neutral - ge_positive, ge_positive], -1)
        return torch.log(p.clamp(min=1e-6))

    def sentiment_loss(self, r, target):
        if self.sentiment_mode != "ordinal":
            return -(target * self.sentiment_logprobs(r)).sum(-1).mean()
        s = self.sentiment_head(r).squeeze(-1)
        t1 = self.theta[0]
        t2 = t1 + nn.functional.softplus(self.gap[0]) + 1e-3
        bce = nn.functional.binary_cross_entropy_with_logits
        # cumulative targets: P(y >= neutral) = p_neu + p_pos, P(y >= positive) = p_pos
        return bce(s - t1, target[:, 1] + target[:, 2]) + bce(s - t2, target[:, 2])

    def aspect_input(self, r, h, spans):
        if not self.sentence_context:
            return r
        return torch.cat([r, self.dropout(h.float()[spans[:, 0], 0])], -1)

    def aspect_logprobs(self, a):
        logits = self.aspect_head(a)
        if self.aspect_mode != "hier":
            return torch.log_softmax(logits, -1)
        # log P(l2) = log P(l1) + log P(l2 | l1), the conditional normalised within the l1 group
        masked = logits[:, None, :].masked_fill(~self.l1_mask[None], float("-inf"))
        group_lse = torch.logsumexp(masked, -1)                                 # (n, n_l1)
        cond = logits - group_lse[:, self.l1_of]
        return torch.log_softmax(self.aspect_l1_head(a), -1)[:, self.l1_of] + cond

    def heads_input(self, batch, h_text, spans, marked=None):
        """Span representations for `spans`: from the text pass, or from a levitated-marker pass
        (`marked` = (states, open idx, close idx) when already computed)."""
        mask = batch["attention_mask"].bool()
        if not self.markers:
            return self.span_repr(h_text, mask, spans)
        hm, io, ic = marked if marked is not None else self.hidden_with_markers(batch, spans)
        hm = self.dropout(hm)
        b = spans[:, 0]
        return self.span_repr(hm[:, :h_text.size(1)], mask, spans, ends=(hm[b, io], hm[b, ic]))

    def loss(self, batch):
        spans = batch["spans"]
        marked = None
        if self.markers:
            # One pass: the text cannot attend to the markers, so its states equal the plain pass.
            marked = self.hidden_with_markers(batch, spans)
            h = self.dropout(marked[0][:, :batch["input_ids"].size(1)])
        else:
            h = self.dropout(self.hidden(batch))
        labels = batch["labels"]
        mask = self._mask(labels)
        loss = -self.crf(self.tag_head(h).float(), labels.clamp(min=0), mask=mask, reduction="mean")
        if len(spans):
            r = self.heads_input(batch, h, spans, marked)
            target = batch["span_sentiment"]
            if not self.soft:
                target = torch.nn.functional.one_hot(target.argmax(-1), len(SENTIMENTS)).float()
            loss = loss + self.lambdas[0] * self.sentiment_loss(r, target)
            asp = batch["span_aspect"]
            keep = asp != -100
            if keep.any():
                logp = self.aspect_logprobs(self.aspect_input(r, h, spans))
                nll = -logp[keep].gather(1, asp[keep][:, None]).squeeze(1)
                w = batch.get("span_aspect_weight")
                w = torch.ones_like(nll) if w is None else w[keep]
                loss = loss + self.lambdas[1] * (w * nll).sum() / w.sum().clamp(min=1e-6)
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
            spans_t = torch.tensor(rows, device=h.device)
            r = self.heads_input(batch, h, spans_t)
            ps = self.sentiment_logprobs(r).exp()
            pa = self.aspect_logprobs(self.aspect_input(r, h, spans_t)).exp()
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
                            for s, e, pol, _, a, *_ in it["spans"]] for it in items}


def parse_weights(text: str | None) -> dict | None:
    """'high=1,medium=0.5,low=0.25' -> dict; 'default' -> DEFAULT_CONFIDENCE_WEIGHTS."""
    if not text:
        return None
    if text == "default":
        return dict(DEFAULT_CONFIDENCE_WEIGHTS)
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in text.split(","))}


def run_name(args) -> str:
    flags = [("soft", args.soft), ("hier", args.aspect_head == "hier"), ("sent", args.sentence_context),
             ("conf", bool(args.aspect_confidence)), ("ord", args.sentiment_loss == "ordinal"),
             ("mark", args.markers)]
    return "+".join(args.train_sources) + "".join(f"-{n}" for n, on in flags if on)


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--train-sources", nargs="+", default=list(SOURCES), choices=ALL_SOURCES)
    ap.add_argument("--eval-sources", nargs="+", default=list(SOURCES), choices=ALL_SOURCES)
    ap.add_argument("--soft", action="store_true", help="train the sentiment head on the soft labels")
    ap.add_argument("--aspect-head", choices=["flat", "hier"], default="flat", help="(8) hier = P(l1) * P(l2|l1)")
    ap.add_argument("--sentence-context", action="store_true", help="(10) aspect head sees the sentence vector")
    ap.add_argument("--aspect-confidence", nargs="?", const="default",
                    help="(11) weight aspect labels by LLM confidence; 'high=1,medium=0.5,low=0.25' (default)")
    ap.add_argument("--sentiment-loss", choices=["ce", "ordinal"], default="ce", help="(5) ordinal sentiment head")
    ap.add_argument("--markers", action="store_true", help="(12) levitated markers for the span heads")
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
    if is_lstm and args.markers:
        ap.error("--markers needs a transformer encoder")
    args.lr = args.lr or (lstm.DEFAULTS["lr"] if is_lstm else 3e-5)
    args.epochs = args.epochs or (lstm.DEFAULTS["tagging_epochs"] if is_lstm else 10)
    args.patience = args.patience or (lstm.DEFAULTS["tagging_patience"] if is_lstm else 3)
    conf_weights = parse_weights(args.aspect_confidence)
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
    marker_ids = None
    if args.markers:
        tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
        marker_ids = tuple(tok.convert_tokens_to_ids([MARK_OPEN, MARK_CLOSE]))
    train, valid, test = (encode(tok, e, args.max_length) for e in (ex_tr, ex_va, ex_te))
    print(f"train {len(train)} / valid {len(valid)} / test {len(test)} texts; "
          f"{sum(len(x['gold']) for x in train)} train spans")
    model = MultiTask(args.model, encoder, lambda_sentiment=args.lambda_sentiment, lambda_aspect=args.lambda_aspect,
                      soft=args.soft, aspect_head=args.aspect_head, sentence_context=args.sentence_context,
                      sentiment_loss=args.sentiment_loss, markers=args.markers, marker_ids=marker_ids,
                      pad_id=tok.pad_token_id, vocab_size=len(tok))
    out = Path(args.out or f"runs/multitask/{model_tag(args.model)}/{run_name(args)}")
    if is_lstm:
        lstm.save_vocab(tok, out)
    pad = tok.pad_token_id
    va_texts = pd.DataFrame([{"text_id": e["text_id"], "source": e["source"]} for e in ex_va])
    te_texts = pd.DataFrame([{"text_id": e["text_id"], "source": e["source"]} for e in ex_te])
    epoch = {"n": 0}

    def train_batches():
        epoch["n"] += 1
        return batches(train, args.batch_size, pad, shuffle=True, seed=args.seed + epoch["n"],
                       confidence_weights=conf_weights)

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
