"""Attach unified aspect labels (LLM silver + human gold) to the pairs table.

Label files are keyed by ``uid`` and also store the target string; a label is
only attached when the target still matches, so a change in the loaders that
shifts uids can never silently put labels on the wrong pair.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .annotate import PROMPT_VERSION, latest_labels
from .taxonomy import level1


def attach_aspects(pairs: pd.DataFrame, llm_files: list[Path], human_files: list[Path]) -> tuple[pd.DataFrame, dict]:
    pairs = pairs.copy()
    pairs["aspect"] = None
    pairs["aspect_source"] = None
    pairs["aspect_confidence"] = None
    info = {"llm_labels": 0, "llm_stale_prompt": 0, "human_labels": 0, "target_mismatch": 0}
    idx = pd.Series(pairs.index, index=pairs["uid"])

    def apply(df: pd.DataFrame, source_of) -> int:
        df = df[df["uid"].isin(idx.index)]
        rows = idx.loc[df["uid"]].to_numpy()
        ok = pairs.loc[rows, "target"].to_numpy() == df["target"].to_numpy()
        info["target_mismatch"] += int((~ok).sum())
        df, rows = df[ok], rows[ok]
        pairs.loc[rows, "aspect"] = df["aspect"].to_numpy()
        pairs.loc[rows, "aspect_source"] = [source_of(r) for r in df.itertuples(index=False)]
        pairs.loc[rows, "aspect_confidence"] = df["confidence"].to_numpy() if "confidence" in df else None
        return len(df)

    for f in llm_files:
        lab = latest_labels(f)
        if lab.empty:
            continue
        info["llm_stale_prompt"] += int((lab["prompt_version"] != PROMPT_VERSION).sum())
        info["llm_labels"] += apply(lab, lambda r: f"llm:{r.model}@{r.prompt_version}")
    for f in human_files:
        hum = pd.read_csv(f).drop_duplicates("uid", keep="last").assign(confidence=None)
        info["human_labels"] += apply(hum, lambda r: "human")
    pairs["aspect_l1"] = pairs["aspect"].map(level1)
    return pairs, info
