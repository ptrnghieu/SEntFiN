"""Canonical entity keys.

SEntFiN ships ``entity_list_comprehensive.csv`` mapping ~1,000 NSE symbols /
sector tags to their surface aliases ("Bharat Petroleum Corporation", "BPCL",
...). We use it to collapse aliases onto one key so that "seen in train" really
means "the model has seen this company before", not just this spelling.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[^\w$]+|[^\w]+$")


def surface_key(name: str) -> str:
    """Case/whitespace/edge-punctuation-insensitive form of an entity name."""
    return _EDGE_PUNCT.sub("", _WS.sub(" ", name.strip().lower()))


def load_alias_table(path: str | Path) -> dict[str, str]:
    """Return {surface_key(alias): SYMBOL}; aliases claimed by >1 symbol are dropped."""
    df = pd.read_csv(path)
    owners: dict[str, set[str]] = {}
    for symbol, aliases in zip(df["Symbol"], df["Entity"]):
        for alias in str(aliases).split(","):
            key = surface_key(alias)
            if key:
                owners.setdefault(key, set()).add(str(symbol).strip())
    return {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}


def entity_id(name: str, aliases: dict[str, str] | None = None) -> str:
    key = surface_key(name)
    if aliases and key in aliases:
        return f"sym:{aliases[key]}"
    return f"txt:{key}"
