"""Vendored from the crawler repo so training text is normalized and split exactly like
deployment text: https://github.com/minhhai205/aspect-based-FSA, commit
5b5d4042e18cf04c1f49a29eea6c7c666c75bd08, src/absa_fsa/textnorm.py (normalize, split_sentences).
Do not edit here; re-vendor if the crawler changes (tests/test_finentity.py pins behaviour).

Original docstring: "Chuẩn hóa văn bản và tách câu. QUAN TRỌNG: normalize() chạy MỘT LẦN, TRƯỚC khi
tính bất kỳ offset nào. Chuẩn hóa lại chuỗi đã gắn span là cách chắc chắn nhất làm hỏng nhãn mà
không ai phát hiện."
"""
from __future__ import annotations

import html
import re
import unicodedata

# Ký tự vô hình hay lẫn trong HTML báo điện tử
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Dấu nháy / gạch ngang kiểu sách -> ASCII, để khớp từ điển ổn định
_PUNCT_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x2032: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x2033: '"',
    0x2013: "-", 0x2014: "-", 0x2212: "-",
    0x00A0: " ", 0x202F: " ", 0x2009: " ", 0x2007: " ",
    0x2026: "...",
}

_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{2,}")


def normalize(text: str) -> str:
    """Giải HTML entity + NFC + bỏ ký tự vô hình + gộp khoảng trắng.

    Giải entity hai lần vì vài feed (Thanh Niên) mã hóa lồng: `&amp;ocirc;`.
    """
    if not text:
        return ""
    prev = None
    for _ in range(3):
        if text == prev:
            break
        prev = text
        text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE).translate(_PUNCT_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _NL.sub("\n", text)
    return "\n".join(ln.strip() for ln in text.split("\n")).strip()


# --------------------------------------------------------------------------- #
# Tách câu
# --------------------------------------------------------------------------- #

_ABBREV = {
    # tiếng Anh
    "mr", "mrs", "ms", "dr", "prof", "inc", "ltd", "co", "corp", "jsc", "plc",
    "vs", "etc", "no", "vol", "est", "approx", "ca", "cf", "al", "fig", "dept",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # tiếng Việt
    "tp", "tt", "ts", "ths", "gs", "pgs", "ks", "bs", "ông", "bà", "cty", "ctcp",
    "tnhh", "hđqt", "tgđ", "gđ", "nxb", "kcn", "kdc", "ubnd", "hcm",
}

_SENT_END = re.compile(
    r"""
    (?<=[.!?…])          # ngay sau dấu kết câu
    ["')\]”’]*      # có thể kèm dấu đóng
    \s+
    (?=["'(\[“‘]*[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ0-9])
    """,
    re.VERBOSE,
)

# dấu chấm/phẩy trong số: 10,5 tỷ | 1.234.567 | 3.5%
_NUMERIC_DOT = re.compile(r"(?<=\d)[.,](?=\d)")
# chuỗi chữ cái đầu: j.p | u.s | a.b.c
_INITIALISM = re.compile(r"(?:[a-z]\.)+[a-z]?")


def _is_false_break(text: str, pos: int) -> bool:
    """pos = vị trí bắt đầu câu kế tiếp. True nếu đây là ngắt câu giả."""
    left = text[:pos].rstrip()
    if not left or left[-1] not in ".!?…":
        return not left
    token = re.split(r"[\s(\[\"']", left[:-1])[-1].lower().strip(".,;:")
    if token in _ABBREV:
        return True
    if len(token) == 1 and token.isalpha():
        return True
    if _INITIALISM.fullmatch(token):
        return True
    if token.isdigit() and len(token) <= 2:
        return True
    return False


def split_sentences(text: str, min_len: int = 10) -> list[tuple[int, int]]:
    """Trả span (start, end) của từng câu TRÊN CHÍNH `text`.

    Trả span thay vì chuỗi để text[start:end] luôn truy ngược được về bài gốc.
    """
    if not text:
        return []

    masked = _NUMERIC_DOT.sub("\u0000", text)  # che dấu chấm trong số, giữ nguyên độ dài

    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(masked):
        if _is_false_break(masked, m.end()):
            continue
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))

    # xuống dòng cũng là ranh giới câu
    out: list[tuple[int, int]] = []
    for s, e in spans:
        seg = s
        for i in range(s, e):
            if text[i] == "\n":
                if i > seg:
                    out.append((seg, i))
                seg = i + 1
        if e > seg:
            out.append((seg, e))

    final: list[tuple[int, int]] = []
    for s, e in out:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= min_len:
            final.append((s, e))
    return final
