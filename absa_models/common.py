"""Shared pieces for the single-task baselines: data access, metrics, training loop.

Data comes from ``python -m absa_data build`` (``data/processed/<source>/``).
Every run writes ``metrics.json``, ``metrics.md`` and test predictions to its
output directory, so runs on Colab / a GPU box can be compared afterwards.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

SOURCES = ("sentfin", "fiqa", "finentity")  # trained together by default
# Selectable with --train-sources / --eval-sources, never part of the default mix:
#   fiqa_finben  FiQA under FinBen's +-0.1 thresholds, for comparison with published numbers
#   vnnews       crawled Vietnam news (in-domain; LLM labels until the test split is reviewed)
EXTRA_SOURCES = ("fiqa_finben", "vnnews")
ALL_SOURCES = SOURCES + EXTRA_SOURCES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_table(processed: Path, name: str, sources) -> pd.DataFrame:
    """`name` is 'pairs' or 'texts'; concatenates the requested sources."""
    frames = []
    for s in sources:
        f = processed / s / f"{name}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run `python -m absa_data build` first")
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def load_tokenizer(name: str):
    """AutoTokenizer, falling back to a BERT WordPiece tokenizer built from vocab.txt.

    yiyanghkust/finbert-pretrain ships only config.json + vocab.txt (no tokenizer_config),
    and a config.json without `model_type`, which transformers 5.x cannot resolve through the Auto classes."""
    from transformers import AutoTokenizer, BertTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(name)
    except (ValueError, OSError):
        return BertTokenizerFast.from_pretrained(name)


def load_encoder(name: str):
    """AutoModel, falling back to BertModel when config.json lacks `model_type` (finbert-pretrain)."""
    from transformers import AutoModel, BertModel
    try:
        return AutoModel.from_pretrained(name)
    except ValueError:
        return BertModel.from_pretrained(name)


# ----------------------------------------------------------------------------- metrics

def cls_metrics(y, yhat, labels) -> dict:
    """Accuracy and macro-F1. Macro-F1 averages over the classes that occur in the gold or the
    predictions of this subset: a class absent from both would otherwise count as F1 = 0 and
    drag the average down (e.g. FiQA's human aspects cover 12 of the 21 categories)."""
    y, yhat = list(y), list(yhat)
    if not y:
        return {"n": 0}
    present = [l for l in labels if l in set(y) | set(yhat)]
    return {"n": len(y), "accuracy": round(accuracy_score(y, yhat), 4),
            "macro_f1": round(f1_score(y, yhat, labels=present, average="macro", zero_division=0), 4),
            "n_classes": len(present),
            "per_class_f1": {l: round(v, 4) for l, v in zip(labels, f1_score(
                y, yhat, labels=list(labels), average=None, zero_division=0)) if l in present}}


def span_metrics(gold: list[set], pred: list[set]) -> dict:
    """Exact-match span P/R/F1; items are (start, end) or (start, end, label) tuples per text."""
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    ng, npred = sum(map(len, gold)), sum(map(len, pred))
    p = tp / npred if npred else 0.0
    r = tp / ng if ng else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"gold": ng, "pred": npred, "tp": tp, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def write_results(out: Path, metrics: dict, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    lines = [f"# {title}\n", f"```json\n{json.dumps(metrics.get('config', {}), indent=1)}\n```\n"]
    for split_name, block in metrics.items():
        if split_name == "config" or not isinstance(block, dict):
            continue
        lines.append(f"## {split_name}\n")
        lines.append("| subset | n | " + " | ".join(k for k in _metric_keys(block)) + " |")
        lines.append("|---|---|" + "---|" * len(_metric_keys(block)))
        for subset, m in block.items():
            if isinstance(m, dict):
                lines.append(f"| {subset} | {m.get('n', m.get('gold', ''))} | " +
                             " | ".join(str(m.get(k, "")) for k in _metric_keys(block)) + " |")
        lines.append("")
    (out / "metrics.md").write_text("\n".join(lines))


def _metric_keys(block: dict) -> list[str]:
    keys = []
    for m in block.values():
        if isinstance(m, dict):
            for k, v in m.items():
                if k not in keys and k not in ("n", "gold", "per_class_f1") and not isinstance(v, dict):
                    keys.append(k)
    return keys


# ----------------------------------------------------------------------------- training loop

def train_loop(model, train_batches, evaluate, *, epochs: int, lr: float, head_lr: float | None = None,
               weight_decay: float = 0.01, warmup: float = 0.1, patience: int = 2, grad_clip: float = 1.0,
               out: Path, log=print) -> dict:
    """Generic loop: `train_batches()` yields dicts the model's `loss(batch)` accepts;
    `evaluate()` returns (score, details) on the validation set, higher is better.
    `head_lr` (default: `lr`) applies to everything outside `model.encoder`, i.e. the
    classification head and CRF transitions, which learn far too slowly at BERT's 2e-5.
    Keeps the best state in memory and on disk (`best.pt`)."""
    dev = device()
    model.to(dev)
    steps_per_epoch = sum(1 for _ in train_batches())
    total = max(1, steps_per_epoch * epochs)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = []
    for is_enc, group_lr in ((True, lr), (False, head_lr or lr)):
        for decay in (True, False):
            ps = [p for n, p in model.named_parameters()
                  if n.startswith("encoder.") == is_enc and any(k in n for k in no_decay) != decay]
            if ps:
                params.append({"params": ps, "lr": group_lr, "weight_decay": weight_decay if decay else 0.0})
    opt = torch.optim.AdamW(params, lr=lr)
    warm = int(warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad, history = -math.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in train_batches():
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(dev.type, enabled=use_amp):
                loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            score, details = evaluate()
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "valid_score": round(score, 4),
                        "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch}: loss {np.mean(losses):.4f}  valid {score:.4f}  ({time.time() - t0:.0f}s)")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            out.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop after epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_valid_score": round(best, 4), "history": history}
