"""Steps 1b / 1c: classify a given (text, target) pair with FinBERT.

  --task sentiment   negative / neutral / positive (soft labels with --soft)
  --task aspect      21 unified aspect categories (silver labels)

Input: the text with the target marked as ``<t> target </t>`` (when its span is
known) as segment A and the target string as segment B.

Reported on test, per source:
  * sentiment: all pairs, gold-sentiment pairs only, unseen entities (SEntFiN shortcut
    check), FiQA under FinBen's thresholds (``polarity_finben``) for comparability;
  * aspect: level-2 and level-1 macro-F1 against the silver labels, and against FiQA's
    human gold aspects where they map unambiguously.

    python -m absa_models.pair_classify --task sentiment --soft
    python -m absa_models.pair_classify --task sentiment --train-sources sentfin   # single-source run
    python -m absa_models.pair_classify --task aspect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from absa_data.taxonomy import LABELS as ASPECTS
from absa_data.taxonomy import map_fiqa_aspect

from .common import SOURCES, load_encoder, load_tokenizer, cls_metrics, device, load_table, set_seed, train_loop, write_results

SENTIMENTS = ("negative", "neutral", "positive")
MARK_OPEN, MARK_CLOSE = "<t>", "</t>"


def mark(text: str, start, end) -> str:
    if pd.isna(start) or pd.isna(end):
        return text
    s, e = int(start), int(end)
    return f"{text[:s]}{MARK_OPEN} {text[s:e]} {MARK_CLOSE}{text[e:]}"


def encode(tok, df: pd.DataFrame, task: str, soft: bool, max_length: int) -> dict:
    a = [mark(t, s, e) for t, s, e in zip(df["text"], df["start"], df["end"])]
    enc = tok(a, df["target"].tolist(), truncation="only_first", max_length=max_length)
    labels = SENTIMENTS if task == "sentiment" else ASPECTS
    col = "polarity" if task == "sentiment" else "aspect"
    y = df[col].map({l: i for i, l in enumerate(labels)}).to_numpy()
    if task == "sentiment" and soft:
        target = df[["p_negative", "p_neutral", "p_positive"]].to_numpy(dtype=np.float32)
    else:
        target = np.eye(len(labels), dtype=np.float32)[y]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "y": y, "target": target}


def batches(data: dict, idx, batch_size: int, pad_id: int, shuffle: bool, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(idx) if shuffle else np.asarray(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        n = max(len(data["input_ids"][j]) for j in b)
        yield {"input_ids": torch.tensor([data["input_ids"][j] + [pad_id] * (n - len(data["input_ids"][j])) for j in b]),
               "attention_mask": torch.tensor([data["attention_mask"][j] + [0] * (n - len(data["attention_mask"][j]))
                                               for j in b]),
               "target": torch.tensor(data["target"][b])}


class PairClassifier(nn.Module):
    def __init__(self, model_name: str, n_labels: int, vocab_size: int, class_weights=None, dropout: float = 0.1):
        super().__init__()
        self.encoder = load_encoder(model_name)
        self.encoder.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_labels)
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None
                             else torch.ones(n_labels))

    def logits(self, batch):
        h = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0]
        return self.head(self.dropout(h))

    def loss(self, batch):
        # Soft-target cross-entropy (equal to ordinary CE for one-hot targets), class-weighted.
        logp = torch.log_softmax(self.logits(batch).float(), -1)
        per = -(batch["target"] * logp * self.w).sum(-1)
        return per.mean()


@torch.no_grad()
def predict(model, data, idx, batch_size, pad_id) -> np.ndarray:
    dev = device()
    model.eval()
    out = []
    for b in batches(data, idx, batch_size, pad_id, shuffle=False):
        out.append(torch.softmax(model.logits({k: v.to(dev) for k, v in b.items()}).float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def report(df: pd.DataFrame, probs: np.ndarray, task: str) -> dict:
    labels = SENTIMENTS if task == "sentiment" else ASPECTS
    df = df.assign(pred=[labels[i] for i in probs.argmax(1)])
    gold_col = "polarity" if task == "sentiment" else "aspect"
    subsets = {"all": df}
    for s in sorted(df["source"].unique()):
        subsets[s] = df[df["source"] == s]
    res = {}
    if task == "sentiment":
        human = df[df["sentiment_source"] == "gold"]
        subsets["all, gold sentiment only"] = human
        for s in sorted(human["source"].unique()):
            subsets[f"{s}, gold only"] = human[human["source"] == s]
            sub = human[(human["source"] == s) & ~human["entity_seen_in_train"].astype(bool)]
            if len(sub):
                subsets[f"{s}, gold, unseen entities"] = sub
        for name, d in subsets.items():
            res[name] = cls_metrics(d[gold_col], d["pred"], labels)
        fq = human[(human["source"] == "fiqa") & human["polarity_finben"].notna()]
        if len(fq):
            res["fiqa vs FinBen classes (polarity_finben)"] = cls_metrics(fq["polarity_finben"], fq["pred"], labels)
    else:
        for name, d in subsets.items():
            res[name] = cls_metrics(d[gold_col], d["pred"], labels)
            l1 = sorted({a.split("/")[0] for a in labels})
            res[f"{name}, level 1"] = cls_metrics(d[gold_col].str.split("/").str[0], d["pred"].str.split("/").str[0], l1)
        fq = df[df["source"] == "fiqa"].assign(gold=lambda d: d["fiqa_aspect"].map(map_fiqa_aspect)).dropna(subset=["gold"])
        if len(fq):
            res["fiqa vs human gold aspects (mapped)"] = cls_metrics(fq["gold"], fq["pred"], labels)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["sentiment", "aspect"], required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model", default="yiyanghkust/finbert-pretrain")
    ap.add_argument("--train-sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--eval-sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--gold-only", action="store_true",
                    help="sentiment: train only on pairs with human sentiment (drop LLM-added targets)")
    ap.add_argument("--soft", action="store_true", help="sentiment: train on the soft labels")
    ap.add_argument("--class-weights", action="store_true", help="inverse-frequency class weights")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use at most N pairs per split (smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    set_seed(args.seed)

    srcs = sorted(set(args.train_sources) | set(args.eval_sources))
    pairs = load_table(Path(args.processed), "pairs", srcs)
    if args.task == "aspect":
        pairs = pairs[pairs["aspect"].notna()]
    tr = pairs[(pairs["split"] == "train") & pairs["source"].isin(args.train_sources)]
    if args.gold_only:
        tr = tr[tr["sentiment_source"] == "gold"]
    va = pairs[(pairs["split"] == "valid") & pairs["source"].isin(args.eval_sources)]
    te = pairs[(pairs["split"] == "test") & pairs["source"].isin(args.eval_sources)]
    if args.limit:
        tr, va, te = tr.head(args.limit), va.head(args.limit), te.head(args.limit)
    tr, va, te = (d.reset_index(drop=True) for d in (tr, va, te))
    print(f"{args.task}: train {len(tr)} / valid {len(va)} / test {len(te)} pairs from {args.train_sources}")

    tok = load_tokenizer(args.model)
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    labels = SENTIMENTS if args.task == "sentiment" else ASPECTS
    weights = None
    if args.class_weights:
        col = "polarity" if args.task == "sentiment" else "aspect"
        counts = tr[col].value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
        weights = (len(tr) / (len(labels) * np.maximum(counts, 1))).tolist()
    model = PairClassifier(args.model, len(labels), len(tok), weights)
    data = {name: encode(tok, d, args.task, args.soft, args.max_length) for name, d in (("tr", tr), ("va", va), ("te", te))}
    pad = tok.pad_token_id
    out = Path(args.out or f"runs/{args.task}/{'+'.join(args.train_sources)}"
               f"{'-soft' if args.soft else ''}{'-gold' if args.gold_only else ''}{'-cw' if args.class_weights else ''}")
    ep = {"n": 0}

    def train_batches():
        ep["n"] += 1
        return batches(data["tr"], np.arange(len(tr)), args.batch_size, pad, shuffle=True, seed=args.seed + ep["n"])

    def val():
        res = report(va, predict(model, data["va"], np.arange(len(va)), args.batch_size, pad), args.task)
        key = "all, gold sentiment only" if args.task == "sentiment" else "all"
        return res[key].get("macro_f1", 0.0), res

    log = train_loop(model, train_batches, val, epochs=args.epochs, lr=args.lr, patience=args.patience, out=out)
    probs = predict(model, data["te"], np.arange(len(te)), args.batch_size, pad)
    test_res = report(te, probs, args.task)
    valid_res = val()[1]
    write_results(out, {"config": {**vars(args), **log}, "valid": valid_res, "test": test_res},
                  f"{args.task.capitalize()} given the target")
    te.assign(pred=[labels[i] for i in probs.argmax(1)], confidence=probs.max(1))[
        ["uid", "source", "text", "target", "polarity" if args.task == "sentiment" else "aspect", "pred", "confidence"]
    ].to_json(out / "test_predictions.jsonl", orient="records", lines=True)
    key = "all, gold sentiment only" if args.task == "sentiment" else "all"
    print(f"test {key}: acc {test_res[key].get('accuracy')} macro-F1 {test_res[key].get('macro_f1')}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
