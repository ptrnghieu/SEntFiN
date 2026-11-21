"""Loader and cleaner for SEntFiN 1.0 (Sinha et al., 2022).

Raw format: ``S No., Title, Decisions, Words`` where ``Decisions`` is a Python
dict literal ``{'entity': 'positive', ...}``.

Problems found in the raw file and how they are handled (each is logged):

* 13 ``Decisions`` strings are not valid Python because an apostrophe inside an
  entity is not escaped (``'Moody's'``) -> parsed with a tolerant regex.
* 19 entities were corrupted by a global find/replace of trading-call words
  ("Hold" -> "neutral", "sell" -> "negative"): ``'Bajaj neutralings'`` is
  "Bajaj Holdings", ``'FTSE Rusnegative'`` is "FTSE Russell" -> reversed when the
  repaired name is found in the title.
* 6 entities are truncated ("Alibab", "Karur Vysya Ban", "Bhart", "Australi",
  "B" for Brent, "G" for Gold) -> extended to the full word of the title.
  Plural/possessive endings ("NBFCs", "Sebis") are left alone.
* 4 entities have leading/trailing spaces; 86 titles have double spaces.
* 67 titles occur twice. Copies are merged; an entity labelled differently in
  two copies (12 titles) is dropped as an annotation conflict.
* ``Words`` is a whitespace token count that disagrees with the title when it
  contains double spaces -> dropped (trivially recomputable).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from .entities import entity_id
from .schema import LABEL2ID, LABELS
from .text import Span, align_targets, clean_text

_LABEL_ALT = "|".join(LABELS)
_PAIR = re.compile(r"""(?:'(.+?)'|"(.+?)")\s*:\s*'(""" + _LABEL_ALT + r""")'\s*(?=,|\})""")
# Reverse of the label-normalisation replace that leaked into entity names.
_REPAIRS = (("neutral", "Hold"), ("negative", "sell"), ("positive", "Buy"))


def parse_decisions(raw: str) -> tuple[dict[str, str], bool]:
    """Parse a Decisions cell. Returns (mapping, used_fallback)."""
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, dict) and all(v in LABEL2ID for v in value.values()):
            return {str(k): v for k, v in value.items()}, False
    except (ValueError, SyntaxError):
        pass
    pairs = {(a or b): lab for a, b, lab in _PAIR.findall(raw)}
    if not pairs:
        raise ValueError(f"cannot parse Decisions: {raw!r}")
    return pairs, True


def repair_entity(entity: str, title: str) -> str:
    """Undo the 'Hold'->'neutral' / 'sell'->'negative' corruption when it fits the title."""
    if entity.lower() in title.lower():
        return entity
    fixed = entity
    for bad, good in _REPAIRS:
        fixed = fixed.replace(bad, good)
    return fixed if fixed != entity and fixed.lower() in title.lower() else entity


def extend_truncated(text: str, span: Span) -> Span | None:
    """If a substring match stops mid-word ("Alibab" in "Alibaba"), extend it to the word end.

    Returns None when the match does not start at a word boundary ("IHFL" in
    "IIHFL") or the rest of the word is only a plural/possessive "s"/"es".
    """
    s, e = span.start, span.end
    if s > 0 and text[s - 1].isalnum():
        return None
    j = e
    while j < len(text) and text[j].isalpha():
        j += 1
    if j == e or text[e:j].lower() in ("s", "es"):
        return None
    return Span(s, j, "exact")


def load_sentfin(csv_path: str | Path, aliases: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pairs, issues). `pairs` follows schema.PAIR_COLUMNS minus split columns."""
    raw = pd.read_csv(csv_path)
    issues: list[dict] = []

    def log(sno, kind, detail=""):
        issues.append({"source": "sentfin", "raw_id": str(sno), "issue": kind, "detail": detail})

    # 1) parse + per-row cleaning ------------------------------------------------
    records = []  # (sno, title_raw, title_clean, [(ent_raw, ent_clean, label, flags)])
    for sno, title_raw, dec in zip(raw["S No."], raw["Title"], raw["Decisions"]):
        mapping, fallback = parse_decisions(dec)
        if fallback:
            log(sno, "decisions_unescaped_quote", dec)
        title = clean_text(title_raw)
        if title != title_raw:
            log(sno, "title_whitespace_normalized")
        ents = []
        for ent_raw, label in mapping.items():
            flags = []
            ent = " ".join(ent_raw.split())
            if ent != ent_raw:
                flags.append("entity_whitespace_stripped")
            fixed = repair_entity(ent, title)
            if fixed != ent:
                log(sno, "entity_label_word_corruption_repaired", f"{ent!r} -> {fixed!r}")
                flags.append("entity_repaired")
                ent = fixed
            ents.append((ent_raw, ent, label, flags))
        records.append((sno, title_raw, title, ents))

    # 2) merge exact duplicate titles ---------------------------------------------
    by_title: dict[str, list] = {}
    for rec in records:
        by_title.setdefault(rec[2], []).append(rec)

    rows = []
    for title, recs in by_title.items():
        snos = [r[0] for r in recs]
        text_id = f"sentfin-{min(snos)}"
        merged: dict[str, dict] = {}
        conflicted: set[str] = set()
        for _, _, _, ents in recs:
            for ent_raw, ent, label, flags in ents:
                if ent in merged and merged[ent]["polarity"] != label:
                    conflicted.add(ent)
                merged.setdefault(ent, {"target_raw": ent_raw, "polarity": label, "flags": list(flags)})
        if len(recs) > 1:
            for s in snos[1:]:
                log(s, "duplicate_title_merged", f"merged into {text_id}")
        for ent in conflicted:
            log(text_id, "duplicate_title_label_conflict_dropped", ent)
            merged.pop(ent)
        if not merged:
            continue
        rows.append((text_id, recs[0][1], title, merged, len(recs) > 1))

    # 3) spans + flat pair rows ---------------------------------------------------
    out = []
    for text_id, title_raw, title, merged, was_dup in rows:
        targets = list(merged)
        spans = align_targets(title, targets, fuzzy=False)
        polarities = {m["polarity"] for m in merged.values()}
        for k, (tgt, span) in enumerate(zip(targets, spans)):
            m = merged[tgt]
            flags = list(m["flags"]) + (["duplicate_merged"] if was_dup else [])
            if span.match == "substring":
                ext = extend_truncated(title, span)
                if ext and title[ext.start : ext.end] not in merged:
                    new = title[ext.start : ext.end]
                    log(text_id, "entity_truncation_repaired", f"{tgt!r} -> {new!r}")
                    flags.append("entity_repaired")
                    tgt, span = new, ext
            if not span.found:
                flags.append("target_not_in_text")
                log(text_id, "target_not_in_text", tgt)
            out.append({
                "uid": f"{text_id}#{k}",
                "source": "sentfin",
                "text_id": text_id,
                "text_type": "headline",
                "text": title,
                "text_raw": title_raw,
                "target": tgt,
                "target_raw": m["target_raw"],
                "start": span.start,
                "end": span.end,
                "span_match": span.match,
                "polarity": m["polarity"],
                "label": LABEL2ID[m["polarity"]],
                "score": float("nan"),
                "aspect": None,
                "aspect_l1": None,
                "aspect_l2": None,
                "entity_id": entity_id(tgt, aliases),
                "n_targets": len(targets),
                "is_mixed": len(polarities) > 1,
                "quality_flags": ";".join(flags),
            })
    pairs = pd.DataFrame(out)
    pairs["start"] = pairs["start"].astype("Int64")
    pairs["end"] = pairs["end"].astype("Int64")
    return pairs, pd.DataFrame(issues, columns=["source", "raw_id", "issue", "detail"])
