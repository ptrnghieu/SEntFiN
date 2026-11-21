"""Text cleaning and target-span alignment shared by all sources."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

import ftfy

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_TOKEN = re.compile(r"\w+")


def clean_text(s: str) -> str:
    """Repair mojibake (``Â£`` -> ``£``), NFC-normalize and collapse whitespace."""
    s = unicodedata.normalize("NFC", ftfy.fix_encoding(s))
    return _WS.sub(" ", s).strip()


def dedup_key(s: str) -> str:
    """Loose key for near-duplicate detection.

    Lower-cases, drops URLs (tweets re-posted with a different t.co link) and
    all punctuation (``Infosys' Q2`` == ``Infosys Q2``).
    """
    s = _URL.sub(" ", s.lower())
    return _WS.sub(" ", _NON_ALNUM.sub(" ", s)).strip()


@dataclass(frozen=True)
class Span:
    start: int | None
    end: int | None
    match: str  # exact | case_insensitive | substring | fuzzy | not_found

    @property
    def found(self) -> bool:
        return self.start is not None


NOT_FOUND = Span(None, None, "not_found")

# (name, ignore_case, word_bounded) in order of preference.
_STRATEGIES = (
    ("exact", False, True),
    ("case_insensitive", True, True),
    ("substring", False, False),
    ("substring", True, False),
)


def _occurrences(text: str, target: str, ignore_case: bool, bounded: bool) -> list[tuple[int, int]]:
    pat = re.escape(target)
    if bounded:
        # Word boundaries stop "IT" matching inside "Nitin" or "Jet" inside "SpiceJet".
        pat = rf"(?<!\w){pat}(?!\w)"
    flags = re.IGNORECASE if ignore_case else 0
    return [(m.start(), m.end()) for m in re.finditer(pat, text, flags)]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _fuzzy(text: str, target: str, min_ratio: float) -> tuple[int, int] | None:
    """Best window of whole word tokens whose similarity to `target` >= min_ratio.

    Handles annotation typos such as "Perssimon" (Persimmon), "Tailor Wimpey"
    (Taylor Wimpey) or "SAB Miller" (SABMiller).
    """
    toks = list(_TOKEN.finditer(text))
    n = max(1, len(_TOKEN.findall(target)))
    t = target.lower()
    best_ratio, best = 0.0, None
    for size in range(max(1, n - 1), n + 2):
        for i in range(len(toks) - size + 1):
            s, e = toks[i].start(), toks[i + size - 1].end()
            r = difflib.SequenceMatcher(None, t, text[s:e].lower()).ratio()
            if r > best_ratio:
                best_ratio, best = r, (s, e)
    return best if best_ratio >= min_ratio else None


def align_targets(
    text: str,
    targets: list[str],
    *,
    fuzzy: bool = True,
    min_ratio: float = 0.85,
    min_fuzzy_len: int = 4,
) -> list[Span]:
    """Find a character span for every target of one text.

    Targets are aligned longest first and prefer occurrences that do not overlap
    an already-assigned span, so in "Add IT, pharma names ...; Aurobindo Pharma top
    bets" the target "pharma" lands on the stand-alone word, not inside
    "Aurobindo Pharma". Nested spans are only used when nothing else matches.
    """
    out: list[Span] = [NOT_FOUND] * len(targets)
    taken: list[tuple[int, int]] = []
    for i in sorted(range(len(targets)), key=lambda j: -len(targets[j])):
        span = _align_one(text, targets[i], taken, fuzzy, min_ratio, min_fuzzy_len)
        out[i] = span
        if span.found:
            taken.append((span.start, span.end))
    return out


def _align_one(text, target, taken, fuzzy, min_ratio, min_fuzzy_len) -> Span:
    if not target:
        return NOT_FOUND
    for allow_overlap in (False, True):
        for name, ignore_case, bounded in _STRATEGIES:
            for occ in _occurrences(text, target, ignore_case, bounded):
                if allow_overlap or not any(_overlaps(occ, t) for t in taken):
                    return Span(occ[0], occ[1], name)
    if fuzzy and len(target) >= min_fuzzy_len:
        hit = _fuzzy(text, target, min_ratio)
        if hit:
            return Span(hit[0], hit[1], "fuzzy")
    return NOT_FOUND
