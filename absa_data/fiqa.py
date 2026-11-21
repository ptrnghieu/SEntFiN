"""Loader and cleaner for FiQA-2018 Task 1 as packaged by TheFinAI (FinBen).

Hub dataset: https://huggingface.co/datasets/TheFinAI/en-fiqa-absa (gated:
accept the access form once, then authenticate with ``huggingface-cli login``
or the ``HF_TOKEN`` environment variable).

Raw columns: ``_id, sentence, target, aspect, score, type`` with official
train / valid / test splits (822 / 117 / 234 rows).

Problems found and how they are handled (each is logged):

* 17 sentences contain mojibake (``Â£100m``, ``BÃ¶rse``) -> repaired with ftfy.
* 20 headlines carry a scraped section label glued to the first word
  (``CompaniesTesco ...``, ``News FeedSchroders ...``) -> label removed.
* ~10 targets are misspelled w.r.t. the sentence ("Perssimon", "Tailor Wimpey",
  "AB InBrev", "SAB Miller") -> replaced by the fuzzy-matched surface form,
  original kept in ``target_raw``.
* ~10 targets do not occur in the sentence at all: aliases ("Standard
  Chartered" for "StanChart"), implicit ("Berkshire Hathaway Inc." for
  "Buffett's Company") or plainly wrong ("Morrissons" for a Smiths Group
  headline) -> kept but flagged ``target_not_in_text``.
* Aspect labels have typos / inconsistent spelling (``Market/Volatilit``,
  ``Behaviour`` vs ``Behavior``, ``downgrade``, trailing ``/``, ``/?``) and one
  value is a stringified list -> normalized, first aspect kept.
* 3 tweets occur in two official splits, differing only in their t.co link
  -> flagged ``near_duplicate_across_splits`` (official splits kept as-is).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .entities import entity_id
from .schema import LABEL2ID, SPLITS
from .text import align_targets, clean_text, dedup_key

FIQA_REPO = "TheFinAI/en-fiqa-absa"
# Pinned so the data cannot silently change under you; bump deliberately.
FIQA_REVISION = "2484314781c82018dc97057c3d9165b390815d30"

# FinBen's mapping from FiQA's continuous score to 3 classes. Recovered
# empirically: it reproduces 100% of ChanceFocus/flare-fiqasa labels on the
# 1,054 single-target sentences, including the boundary cases -0.1 -> neutral
# and 0.1 -> positive.
NEG_BELOW = -0.1
POS_FROM = 0.1

_SECTION_PREFIX = re.compile(r"^(?:Companies|Markets|Industry News|News Feed)(?=[A-Z])")
_ASPECT_PART_FIXES = {
    "Volatilit": "Volatility",
    "Bearish Behaviour": "Bearish Behavior",
    "Bullish Behaviour": "Bullish Behavior",
    "downgrade": "Downgrade",
    "Exisiting bankruptcy": "Existing bankruptcy",
}


def score_to_polarity(score: float, neg_below: float = NEG_BELOW, pos_from: float = POS_FROM) -> str:
    if score < neg_below:
        return "negative"
    if score >= pos_from:
        return "positive"
    return "neutral"


def normalize_aspect(aspect: str) -> tuple[str, list[str]]:
    flags = []
    if "', '" in aspect:  # "Stock/Price Action/Bullish', 'Corporate/Financial/..."
        aspect = aspect.split("', '")[0]
        flags.append("aspect_multiple_kept_first")
    parts = [p.strip() for p in aspect.split("/")]
    parts = [_ASPECT_PART_FIXES.get(p, p) for p in parts if p and p != "?"]
    fixed = "/".join(parts)
    if fixed != aspect and not flags:
        flags.append("aspect_normalized")
    return fixed, flags


def download_fiqa(cache_dir: str | Path | None = None, revision: str = FIQA_REVISION) -> Path:
    """Download the parquet files (needs HF auth for the gated repo)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        FIQA_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["data/*.parquet"],
        cache_dir=cache_dir,
    )
    return Path(path) / "data"


def read_raw_fiqa(data_dir: str | Path) -> pd.DataFrame:
    """Read ``{split}*.parquet`` files from a directory into one frame with a `split` column."""
    data_dir = Path(data_dir)
    frames = []
    for split in SPLITS:
        files = sorted(data_dir.glob(f"{split}*.parquet"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one {split}*.parquet in {data_dir}, found {files}")
        frames.append(pd.read_parquet(files[0]).assign(split=split))
    return pd.concat(frames, ignore_index=True)


def load_fiqa(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pairs, issues); `split` holds the official split."""
    raw = read_raw_fiqa(data_dir)
    issues: list[dict] = []

    def log(rid, kind, detail=""):
        issues.append({"source": "fiqa", "raw_id": str(rid), "issue": kind, "detail": detail})

    if raw.groupby("_id")["sentence"].nunique().max() != 1:
        raise ValueError("FiQA _id no longer identifies a single sentence; loader needs updating")

    out = []
    for rid, g in raw.groupby("_id", sort=False):
        sentence_raw = g["sentence"].iloc[0]
        text_flags = []
        text = clean_text(sentence_raw)
        if text != " ".join(sentence_raw.split()):
            text_flags.append("mojibake_repaired")
            log(rid, "mojibake_repaired", text)
        stripped = _SECTION_PREFIX.sub("", text)
        if stripped != text:
            text_flags.append("section_prefix_removed")
            log(rid, "section_prefix_removed", text[: len(text) - len(stripped)])
            text = stripped

        targets_raw = g["target"].tolist()
        targets = [clean_text(t) for t in targets_raw]
        spans = align_targets(text, targets, fuzzy=True)
        polarities = [score_to_polarity(s) for s in g["score"]]
        for k, (row, tgt, span, pol) in enumerate(zip(g.itertuples(index=False), targets, spans, polarities)):
            flags = list(text_flags)
            if span.match == "fuzzy":
                surface = text[span.start : span.end]
                log(rid, "target_misspelling_corrected", f"{tgt!r} -> {surface!r}")
                flags.append("target_corrected")
                tgt = surface
            elif not span.found:
                log(rid, "target_not_in_text", tgt)
                flags.append("target_not_in_text")
            aspect, aflags = normalize_aspect(row.aspect)
            for f in aflags:
                log(rid, f, f"{row.aspect!r} -> {aspect!r}")
            flags += aflags
            parts = aspect.split("/")
            out.append({
                "uid": f"fiqa-{rid}#{k}",
                "source": "fiqa",
                "text_id": f"fiqa-{rid}",
                "split": row.split,
                "text_type": row.type,
                "text": text,
                "text_raw": sentence_raw,
                "target": tgt,
                "target_raw": row.target,
                "start": span.start,
                "end": span.end,
                "span_match": span.match,
                "polarity": pol,
                "label": LABEL2ID[pol],
                "score": float(row.score),
                "fiqa_aspect": aspect,
                "fiqa_aspect_l1": parts[0],
                "fiqa_aspect_l2": "/".join(parts[:2]),
                "entity_id": entity_id(tgt),
                "n_targets": len(g),
                "is_mixed": len(set(polarities)) > 1,
                "quality_flags": ";".join(flags),
            })
    pairs = pd.DataFrame(out)
    pairs["start"] = pairs["start"].astype("Int64")
    pairs["end"] = pairs["end"].astype("Int64")

    # Near duplicates that straddle official splits (same tweet, different t.co link).
    keys = pairs["text"].map(dedup_key)
    straddle = pairs.groupby(keys)["split"].transform("nunique") > 1
    for tid in pairs.loc[straddle, "text_id"].unique():
        log(tid.removeprefix("fiqa-"), "near_duplicate_across_splits")
    pairs.loc[straddle, "quality_flags"] = pairs.loc[straddle, "quality_flags"].map(
        lambda f: ";".join(filter(None, [f, "near_duplicate_across_splits"]))
    )
    return pairs, pd.DataFrame(issues, columns=["source", "raw_id", "issue", "detail"])
