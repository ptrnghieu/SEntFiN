"""Leakage-safe splitting.

SEntFiN has no official split. We build one that

* keeps every target of a headline in the same split (pairs are grouped by
  text), and also keeps near-duplicate headlines together (``Infosys' Q2`` vs
  ``Infosys Q2``), so the test set never contains a headline seen in training;
* stratifies on polarity x single/multi-target, so test has the same share of
  multi-entity headlines (the interesting ABSA cases) as train;
* is frozen to a manifest CSV so it does not drift with library versions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .text import dedup_key


def assign_groups(pairs: pd.DataFrame) -> pd.Series:
    """Near-duplicate cluster id (same key after lower-casing, URL/punctuation removal)."""
    keys = pairs["source"] + "|" + pairs["text"].map(dedup_key)
    return keys.map(lambda k: "g-" + hashlib.sha1(k.encode()).hexdigest()[:12])


def stratified_group_split(pairs: pd.DataFrame, seed: int = 42, n_folds: int = 10) -> pd.Series:
    """~80/10/10 split: fold 0 -> test, fold 1 -> valid, rest -> train. Returns split per row."""
    strata = pairs["polarity"] + "|" + np.where(pairs["n_targets"] > 1, "multi", "single")
    folds = np.full(len(pairs), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(sgkf.split(pairs, strata, groups=pairs["group_id"])):
        folds[idx] = fold
    split = np.where(folds == 0, "test", np.where(folds == 1, "valid", "train"))
    return pd.Series(split, index=pairs.index, name="split")


def save_manifest(pairs: pd.DataFrame, path: str | Path) -> None:
    m = pairs[["text_id", "group_id", "split"]].drop_duplicates("text_id").sort_values("text_id")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)


def apply_manifest(pairs: pd.DataFrame, path: str | Path) -> pd.Series:
    m = pd.read_csv(path).set_index("text_id")["split"]
    split = pairs["text_id"].map(m)
    missing = pairs.loc[split.isna(), "text_id"].unique()
    if len(missing):
        raise ValueError(
            f"{len(missing)} texts are not in split manifest {path} (e.g. {missing[:3].tolist()}). "
            "The source data changed: rebuild with --resplit and commit the new manifest."
        )
    return split.rename("split")


def check_no_leakage(pairs: pd.DataFrame) -> None:
    """Raise if a text or near-duplicate group occurs in more than one split of a source."""
    for col in ("text_id", "group_id"):
        n = pairs.groupby(["source", col])["split"].nunique()
        bad = n[n > 1]
        if col == "group_id":
            # FiQA's official splits contain known near-duplicates; they are flagged, not fatal.
            bad = bad[bad.index.get_level_values("source") != "fiqa"]
        if len(bad):
            raise AssertionError(f"{len(bad)} {col}s straddle splits: {bad.index[:5].tolist()}")


def mark_entity_seen(pairs: pd.DataFrame) -> pd.Series:
    """True if the pair's entity_id occurs in the train split of the same source."""
    seen = pairs[pairs["split"] == "train"].groupby("source")["entity_id"].agg(set)
    return pd.Series(
        [e in seen.get(s, set()) for s, e in zip(pairs["source"], pairs["entity_id"])],
        index=pairs.index,
        name="entity_seen_in_train",
    )
