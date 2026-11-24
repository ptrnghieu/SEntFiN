"""Loader and cleaner for FinEntity (Tang et al., EMNLP 2023).

Hub dataset: https://huggingface.co/datasets/yixuantt/FinEntity (ODC-BY).
979 Reuters paragraphs; every financial entity is annotated with its span and
sentiment, so this is the one source whose targets are *exhaustive* (a token
not inside a span really is not a target).

Problems found and how they are handled:

* 70 of 2,131 spans are shifted by 1-5 characters (218 U+FFFD characters in 50
  paragraphs moved the offsets) -> re-anchored on the nearest exact occurrence.
* Reuters codes are mixed into the text (RICs such as ``<JNJ.N>`` in 394
  paragraphs, story codes ``[nL4N2Z028U]`` in 164). Deployment text has none,
  and a ticker in brackets is a shortcut a model would learn -> removed.
* Paragraphs (median 2 sentences) are split into sentences with the crawler's
  own splitter (``sentsplit``), after the crawler's ``normalize``, so training
  text looks like deployment text. A boundary never cuts a gold span.
* Sentences without any entity are kept: with exhaustive annotation they are
  true negatives for target extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .entities import entity_id
from .schema import LABEL2ID
from .sentsplit import normalize, split_sentences

FINENTITY_REPO = "yixuantt/FinEntity"
FINENTITY_REVISION = "e86e5451a1fce4731fe07c710649a229d92e5cab"

_CODES = re.compile(r"\s*<[^<>\s]{1,25}>|\s*\[(?:n[A-Z0-9]{6,}|IRO[^\]\s]*)\]")


def download_finentity(revision: str = FINENTITY_REVISION) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(FINENTITY_REPO, "FinEntity.json", repo_type="dataset", revision=revision))


def _occurrences(text: str, value: str) -> list[int]:
    return [m.start() for m in re.finditer(re.escape(value), text)]


def _anchor(content: str, ann: dict) -> tuple[int, int, bool]:
    """Return (start, end, repaired) with content[start:end] == ann['value']."""
    s, value = ann["start"], ann["value"]
    if content[s:ann["end"]] == value:
        return s, ann["end"], False
    occ = _occurrences(content, value)
    if not occ:
        raise ValueError(f"entity {value!r} not found in paragraph")
    s = min(occ, key=lambda p: abs(p - ann["start"]))
    return s, s + len(value), True


def clean_paragraph(content: str, anns: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Repair spans, strip Reuters codes, normalize like the crawler; return text and
    entities with offsets on the cleaned text."""
    flags = []
    ents = []
    codes = [(m.start(), m.end()) for m in _CODES.finditer(content)]
    for a in anns:
        s, e, repaired = _anchor(content, a)
        # Count only occurrences that survive code removal (a ticker also appears in its RIC <URBN.O>).
        occ = [o for o in _occurrences(content, a["value"])
               if not any(cs < o + len(a["value"]) and o < ce for cs, ce in codes)]
        ents.append({"value": a["value"], "tag": a["tag"], "k": occ.index(s), "repaired": repaired})
    text = _CODES.sub("", content)
    if text != content:
        flags.append("reuters_codes_removed")
    text = normalize(text)
    if "�" in text:
        flags.append("replacement_char_in_text")
    out = []
    for ent in ents:
        value = normalize(ent["value"])
        occ = _occurrences(text, value)
        if ent["k"] >= len(occ):
            raise ValueError(f"lost occurrence {ent['k']} of {value!r} after cleaning")
        s = occ[ent["k"]]
        out.append({"target": value, "target_raw": ent["value"], "start": s, "end": s + len(value),
                    "polarity": ent["tag"].lower(), "repaired": ent["repaired"]})
    return text, out, flags


def sentence_spans(text: str, ents: list[dict]) -> list[tuple[int, int]]:
    """Crawler sentence spans, merged so that no entity crosses or falls outside a span."""
    spans = split_sentences(text) or [(0, len(text))]
    changed = True
    while changed:
        changed = False
        for e in ents:
            hit = [i for i, (s, t) in enumerate(spans) if s < e["end"] and e["start"] < t]
            inside = len(hit) == 1 and spans[hit[0]][0] <= e["start"] and e["end"] <= spans[hit[0]][1]
            if inside:
                continue
            if not hit:  # in a segment the splitter dropped (too short): attach to the nearest span
                i = min(range(len(spans)), key=lambda j: min(abs(spans[j][0] - e["start"]), abs(spans[j][1] - e["end"])))
                hit = [i]
            lo, hi = min(hit), max(hit)
            merged = (min(spans[lo][0], e["start"]), max(spans[hi][1], e["end"]))
            spans = spans[:lo] + [merged] + spans[hi + 1:]
            changed = True
            break
    return spans


def load_finentity(path: str | Path, aliases: dict[str, str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (pairs, texts, issues). `texts` has every sentence, including those without targets."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs, texts, issues = [], [], []

    def log(rid, kind, detail=""):
        issues.append({"source": "finentity", "raw_id": str(rid), "issue": kind, "detail": detail})

    for p, rec in enumerate(data):
        text, ents, flags = clean_paragraph(rec["content"], rec["annotations"])
        for f in flags:
            log(p, f)
        for e in ents:
            if e["repaired"]:
                log(p, "span_offset_repaired", e["target"])
        spans = sentence_spans(text, ents)
        n_split = len(split_sentences(text))
        if len(spans) < n_split:
            log(p, "sentences_merged_to_keep_span", f"{n_split} -> {len(spans)}")
        for j, (s, t) in enumerate(spans):
            text_id = f"finentity-{p}-{j}"
            sent = text[s:t]
            inside = sorted((e for e in ents if s <= e["start"] and e["end"] <= t), key=lambda e: e["start"])
            texts.append({"text_id": text_id, "source": "finentity", "paragraph_id": f"finentity-{p}",
                          "sent_idx": j, "text": sent, "text_type": "news", "target_annotation": "exhaustive",
                          "quality_flags": ";".join(flags)})
            polarities = {e["polarity"] for e in inside}
            for k, e in enumerate(inside):
                eflags = list(flags) + (["entity_repaired"] if e["repaired"] else [])
                pairs.append({
                    "uid": f"{text_id}#{k}", "source": "finentity", "text_id": text_id,
                    "text_type": "news", "text": sent, "text_raw": rec["content"],
                    "target": e["target"], "target_raw": e["target_raw"],
                    "start": e["start"] - s, "end": e["end"] - s, "span_match": "exact",
                    "polarity": e["polarity"], "label": LABEL2ID[e["polarity"]], "score": float("nan"),
                    "fiqa_aspect": None, "fiqa_aspect_l1": None, "fiqa_aspect_l2": None,
                    "entity_id": entity_id(e["target"], aliases), "n_targets": len(inside),
                    "is_mixed": len(polarities) > 1, "quality_flags": ";".join(eflags),
                    "paragraph_id": f"finentity-{p}", "target_annotation": "exhaustive",
                })
    pairs = pd.DataFrame(pairs)
    pairs["start"] = pairs["start"].astype("Int64")
    pairs["end"] = pairs["end"].astype("Int64")
    return pairs, pd.DataFrame(texts), pd.DataFrame(issues, columns=["source", "raw_id", "issue", "detail"])
