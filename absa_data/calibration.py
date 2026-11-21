"""Put SEntFiN's hard labels and FiQA's continuous scores on one footing.

The two datasets disagree in kind and in meaning:

* SEntFiN gives a hard class per (headline, entity). Duplicate headlines show
  these are fairly stable: 73 of 80 repeated judgements agree, and all 7
  disagreements are between adjacent classes (negative <-> neutral).
* FiQA gives a score in [-1, 1]; the usual 3-class labels come from fixed
  thresholds (FinBen: neg < -0.1 <= neu < 0.1 <= pos). Repeated tweets were
  scored with noise of about the same size as that neutral band, so labels
  near a threshold are close to coin flips.
* "Neutral" does not mean the same thing: 38 % of SEntFiN is neutral, but only
  7.8 % of FiQA under the FinBen thresholds.

This module provides

1. soft labels (a distribution over negative / neutral / positive) for both
   sources from an explicit noise model, so the two can be trained with one
   loss, and
2. a calibration of FiQA's thresholds to SEntFiN's notion of neutral, using an
   LLM taught SEntFiN's conventions as a bridge annotator: its agreement with
   SEntFiN gold is measured on SEntFiN *valid*, then the FiQA score
   thresholds that best reproduce its FiQA judgements are fitted. FiQA's labels
   are still computed from the human scores; the LLM only locates the
   thresholds.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .annotate import make_task
from .entities import surface_key
from .schema import LABEL2ID
from .schema import LABELS as POLARITIES
from .text import dedup_key

# ----------------------------------------------------------------------------- soft labels

def soft_from_score(score: float, sigma: float, neg_below: float, pos_from: float) -> tuple[float, float, float]:
    """P(negative), P(neutral), P(positive) if the true score is Normal(score, sigma).

    sigma = 0 reproduces the hard thresholding (neg < neg_below <= neu < pos_from <= pos).
    """
    if sigma <= 0:
        if score < neg_below:
            return 1.0, 0.0, 0.0
        if score >= pos_from:
            return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0
    cdf = lambda x: 0.5 * (1 + math.erf((x - score) / (sigma * math.sqrt(2))))  # noqa: E731
    p_neg = cdf(neg_below)
    p_pos = 1 - cdf(pos_from)
    return p_neg, max(0.0, 1 - p_neg - p_pos), p_pos


def soft_from_hard(polarity: str, flip_rate: float) -> tuple[float, float, float]:
    """Ordinal label smoothing: move `flip_rate` of the mass to the adjacent class(es) only
    (observed SEntFiN disagreements are never positive <-> negative)."""
    e = flip_rate
    return {"negative": (1 - e, e, 0.0), "neutral": (e / 2, 1 - e, e / 2), "positive": (0.0, e, 1 - e)}[polarity]


FINBEN_THRESHOLDS = (-0.1, 0.1)
DEFAULT_FIQA_SIGMA = 0.08        # used when no calibration file exists
DEFAULT_SENTFIN_FLIP_RATE = 0.044


def load_calibration(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def relabel_fiqa(pairs: pd.DataFrame, neg_below: float, pos_from: float) -> pd.DataFrame:
    """Re-cut FiQA scores at the given thresholds; keep the FinBen class in `polarity_finben`."""
    pairs = pairs.copy()
    f = pairs["source"] == "fiqa"
    pairs["polarity_finben"] = None
    pairs.loc[f, "polarity_finben"] = apply_thresholds(pairs.loc[f, "score"], *FINBEN_THRESHOLDS)
    pairs.loc[f, "polarity"] = apply_thresholds(pairs.loc[f, "score"], neg_below, pos_from)
    pairs.loc[f, "label"] = pairs.loc[f, "polarity"].map(LABEL2ID)
    pairs.loc[f, "is_mixed"] = pairs[f].groupby("text_id")["polarity"].transform("nunique") > 1
    return pairs


def add_soft_labels(pairs: pd.DataFrame, fiqa_sigma: float, sentfin_flip_rate: float,
                    neg_below: float, pos_from: float) -> pd.DataFrame:
    """Add label_type, p_negative, p_neutral, p_positive, label_confidence (= max p)."""
    pairs = pairs.copy()
    probs = [
        soft_from_score(s, fiqa_sigma, neg_below, pos_from) if src == "fiqa"
        else soft_from_hard(pol, sentfin_flip_rate)
        for src, s, pol in zip(pairs["source"], pairs["score"], pairs["polarity"])
    ]
    arr = np.round(np.array(probs, dtype=float), 4)
    pairs["label_type"] = np.where(pairs["source"] == "fiqa", "soft", "hard")
    pairs["p_negative"], pairs["p_neutral"], pairs["p_positive"] = arr[:, 0], arr[:, 1], arr[:, 2]
    pairs["label_confidence"] = arr.max(axis=1)
    return pairs


# ----------------------------------------------------------------------------- noise estimates

def estimate_score_sigma(fiqa_pairs: pd.DataFrame) -> dict:
    """Annotation noise of FiQA scores from (text, target) pairs that occur more than once
    (re-posted tweets differing only in their link). sd(difference) = sqrt(2) * sigma."""
    key = fiqa_pairs["text"].map(dedup_key) + "|" + fiqa_pairs["target"].map(surface_key)
    groups = fiqa_pairs.groupby(key)["score"].agg(list)
    groups = groups[groups.map(len) > 1]
    diffs = [a - b for v in groups for a, b in zip(v[:-1], v[1:])]
    sigma = float(np.sqrt(np.mean(np.square(diffs)) / 2)) if diffs else float("nan")
    return {"sigma": round(sigma, 3), "groups": len(groups), "differences": len(diffs),
            "scores": [list(map(float, v)) for v in groups]}


def sentfin_duplicate_agreement(csv_path: str | Path) -> dict:
    """Agreement of SEntFiN labels for the same entity in duplicated headlines.
    Disagreement between two noisy copies is ~2x the per-label flip rate."""
    from .sentfin import parse_decisions, repair_entity
    from .text import clean_text

    raw = pd.read_csv(csv_path)
    copies: dict[str, list[dict]] = {}
    for title_raw, dec in zip(raw["Title"], raw["Decisions"]):
        title = clean_text(title_raw)
        mapping, _ = parse_decisions(dec)
        copies.setdefault(title, []).append({repair_entity(" ".join(e.split()), title): l for e, l in mapping.items()})
    compared, conflicts = 0, Counter()
    for cs in copies.values():
        if len(cs) < 2:
            continue
        for ent, n in Counter(e for c in cs for e in c).items():
            if n < 2:
                continue
            labels = {c[ent] for c in cs if ent in c}
            compared += 1
            if len(labels) > 1:
                conflicts[" <-> ".join(sorted(labels))] += 1
    disagreement = sum(conflicts.values()) / compared if compared else float("nan")
    return {"compared": compared, "conflicts": sum(conflicts.values()), "by_pair": dict(conflicts),
            "disagreement": round(disagreement, 4), "flip_rate": round(disagreement / 2, 4)}


# ----------------------------------------------------------------------------- bridge annotator

# Random stratified sample of SEntFiN *train* headlines (6 per polarity + 3 with mixed
# polarity, seed 7), used verbatim with their gold labels to show SEntFiN's conventions.
SENTIMENT_EXAMPLES: tuple[tuple[str, dict[str, str]], ...] = (
    ("Brent drops below $87 as Fed optimism lifts dollar", {"Brent": "negative"}),
    ("Currency headwinds weigh on IT stocks", {"IT stocks": "negative"}),
    ("Gold's fall threatens to squash run of mining M&As just as momentum was picking up", {"Gold": "negative"}),
    ("Are you prepared for a bear market? Find out", {"market": "negative"}),
    ("Vishal Retail Q4 loss widens by 12 pc to Rs 129 cr", {"Vishal Retail": "negative"}),
    ("Silver extends weakness in futures trade on overseas cues", {"Silver": "negative"}),
    ("MFs get Sebi warning for violating investor norms", {"Sebi": "neutral"}),
    ("Hold Financial Technologies at current level: G Chokkalingam", {"Financial Technologies": "neutral"}),
    ("Hewlett-Packard Co to resume share repurchase program", {"Hewlett-Packard Co": "neutral"}),
    ("India produces 14.3 million tonne sugar till mid-February", {"sugar": "neutral"}),
    ("StanChart IDR holders can convert 25% of holding into shares", {"StanChart": "neutral"}),
    ("Coffee Day to list shares on Monday", {"Coffee Day": "neutral"}),
    ("Aarti Drugs Q1 net up 92 per cent at Rs 8.88 crore", {"Aarti Drugs": "positive"}),
    ("TCS shares gain over 1 per cent post earnings", {"TCS": "positive"}),
    ("Traders build bullish bets on gold", {"gold": "positive"}),
    ("Tata Steel Q4 show beats estimates", {"Tata Steel": "positive"}),
    ("Railway stocks on a roll ahead of Budget", {"Railway stocks": "positive"}),
    ("Castrol Q4 net up 7% to Rs 126 crore", {"Castrol": "positive"}),
    ("Cummins India shares gain after Deutsche upgrades to 'buy'",
     {"Cummins India": "positive", "Deutsche": "neutral"}),
    ("Iraq fears hit European shares, German bond yields fall",
     {"German bond": "positive", "European shares": "negative"}),
    ("Ranbaxy stock up 50% post deal with Sun Pharma; co grabs 13% market share after Diovans launch",
     {"Ranbaxy": "positive", "Sun Pharma": "neutral"}),
)


def build_sentiment_instructions() -> str:
    lines = [
        "You label the sentiment that financial news headlines and social-media posts express towards a "
        "given target, following the conventions of the annotators of the SEntFiN dataset shown in the "
        "examples below.",
        "For every (text, target) pair choose positive, negative or neutral: positive if the text is "
        "favourable for the target, negative if it is unfavourable, neutral if it is neither. Judge each "
        "target separately; targets in the same text can differ.",
        "",
        "Examples (SEntFiN annotations):",
    ]
    for text, labels in SENTIMENT_EXAMPLES:
        lines.append(f"Text: {json.dumps(text, ensure_ascii=False)}")
        lines += [f"  target {json.dumps(t, ensure_ascii=False)} -> {s}" for t, s in labels.items()]
    lines += [
        "",
        "Input: numbered texts, each followed by its targets. Each target has a key T<text>.<target> "
        "(T3.2 = text 3, target 2). Return exactly one label for every key, in input order. Copy the key "
        "and the target string exactly.",
    ]
    return "\n".join(lines)


SENTIMENT_TASK = make_task("sentiment", "sentiment", POLARITIES, build_sentiment_instructions(), "s1")


# ----------------------------------------------------------------------------- threshold fit

def apply_thresholds(scores: pd.Series, neg_below: float, pos_from: float) -> pd.Series:
    return pd.Series(np.where(scores < neg_below, "negative", np.where(scores >= pos_from, "positive", "neutral")),
                     index=scores.index)


def fit_thresholds(scores: pd.Series, labels: pd.Series, step: float = 0.01, max_abs: float = 0.6) -> dict:
    """Thresholds on `scores` that best reproduce `labels` (accuracy; ties -> narrower band)."""
    grid = np.round(np.arange(0, max_abs + 1e-9, step), 4)
    s, y = scores.to_numpy(), labels.to_numpy()
    best = None
    for tn in grid:
        neg = s < -tn
        for tp in grid:
            pred = np.where(neg, "negative", np.where(s >= tp, "positive", "neutral"))
            acc = float((pred == y).mean())
            key = (acc, -(tn + tp))
            if best is None or key > best[0]:
                best = (key, float(-tn), float(tp))
    (acc, _), neg_below, pos_from = best
    return {"neg_below": neg_below, "pos_from": pos_from, "accuracy": round(acc, 4), "n": int(len(s))}


def choose_thresholds(fit: dict, ci: dict, default: tuple[float, float] = (-0.1, 0.1)) -> dict:
    """Keep each default (FinBen) threshold unless its bootstrap 95 % interval excludes it.

    The agreement surface is often a flat plateau (e.g. any negative threshold in
    [-0.12, 0] fits equally well), so moving a threshold needs evidence, not just a
    marginally higher point estimate.
    """
    out = {}
    for key, fb in zip(("neg_below", "pos_from"), default):
        lo, hi = ci[f"{key}_95ci"]
        out[key] = fb if lo <= fb <= hi else fit[key]
    return out


def bootstrap_thresholds(scores: pd.Series, labels: pd.Series, n_boot: int = 200, seed: int = 0) -> dict:
    """Percentile intervals of the fitted thresholds (resampling pairs)."""
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        fits.append(fit_thresholds(scores.iloc[idx], labels.iloc[idx], step=0.02))
    q = lambda k: [round(float(np.percentile([f[k] for f in fits], p)), 3) for p in (2.5, 97.5)]  # noqa: E731
    return {"neg_below_95ci": q("neg_below"), "pos_from_95ci": q("pos_from")}
