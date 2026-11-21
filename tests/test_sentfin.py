from pathlib import Path

import pandas as pd
import pytest

from absa_data.sentfin import extend_truncated, load_sentfin, parse_decisions, repair_entity
from absa_data.text import align_targets

REPO = Path(__file__).resolve().parents[1]


def test_parse_decisions_valid_and_broken_quotes():
    assert parse_decisions("{'Gold': 'positive', 'Silver': 'negative'}") == (
        {"Gold": "positive", "Silver": "negative"}, False)
    # Unescaped apostrophe -> not valid Python, regex fallback.
    assert parse_decisions("{'Sebi': 'neutral', 'Osian's Art Fund': 'neutral'}") == (
        {"Sebi": "neutral", "Osian's Art Fund": "neutral"}, True)
    # Double-quoted keys are valid Python and must survive.
    assert parse_decisions("""{'L&T': 'neutral', "Dr Reddy's": 'neutral'}""")[0] == {
        "L&T": "neutral", "Dr Reddy's": "neutral"}
    with pytest.raises(ValueError):
        parse_decisions("not a dict")


def test_repair_label_word_corruption():
    assert repair_entity("Bajaj neutralings", "Bajaj Holdings has corrected quite a bit") == "Bajaj Holdings"
    assert repair_entity("FTSE Rusnegative", "FTSE Russell to launch indexes") == "FTSE Russell"
    # Untouched when the entity is fine or the repair does not fit the title.
    assert repair_entity("Sebi", "Sebi tightens norms") == "Sebi"
    assert repair_entity("neutral zone", "Nifty in a no-trade zone") == "neutral zone"


@pytest.mark.parametrize("text,target,expected", [
    ("Alibaba's results beat estimates", "Alibab", "Alibaba"),
    ("Brent slips below $40", "B", "Brent"),
    ("NBFCs look attractive", "NBFC", None),      # plural, keep
    ("Sebis reading of the rules", "Sebi", None),  # possessive without apostrophe, keep
    ("IIHFL to raise funds", "IHFL", None),        # does not start at a word boundary
])
def test_extend_truncated(text, target, expected):
    span = align_targets(text, [target])[0]
    assert span.match == "substring"
    ext = extend_truncated(text, span)
    assert (text[ext.start:ext.end] if ext else None) == expected


def test_duplicate_titles_are_merged_and_conflicts_dropped(tmp_path):
    csv = tmp_path / "s.csv"
    pd.DataFrame({
        "S No.": [1, 2, 3, 4, 5],
        "Title": ["Netflix stake sold", "Netflix stake sold", "Nifty  up, rupee down", "Nifty up, rupee down", "Gold flat"],
        "Decisions": ["{'Netflix': 'neutral'}", "{'Netflix': 'negative'}",
                      "{'Nifty': 'positive'}", "{'Nifty': 'positive', 'rupee': 'negative'}", "{'Gold ': 'neutral'}"],
        "Words": [3, 3, 4, 4, 2],
    }).to_csv(csv, index=False)
    pairs, issues = load_sentfin(csv)
    # Netflix conflicts -> whole title dropped; Nifty copies merged (union of targets).
    assert sorted(pairs["text_id"].unique()) == ["sentfin-3", "sentfin-5"]
    nifty = pairs[pairs["text_id"] == "sentfin-3"].set_index("target")
    assert nifty["polarity"].to_dict() == {"Nifty": "positive", "rupee": "negative"}
    assert nifty["is_mixed"].all() and (nifty["n_targets"] == 2).all()
    assert pairs.loc[pairs["text_id"] == "sentfin-5", "target"].item() == "Gold"
    assert set(issues["issue"]) >= {"duplicate_title_label_conflict_dropped", "duplicate_title_merged",
                                    "title_whitespace_normalized"}


@pytest.mark.skipif(not (REPO / "SEntFiN.csv").exists(), reason="SEntFiN.csv not present")
def test_real_sentfin_file():
    pairs, issues = load_sentfin(REPO / "SEntFiN.csv")
    assert pairs["uid"].is_unique
    assert (pairs["span_match"] != "not_found").all()  # every target located after repairs
    assert not pairs["target"].str.contains("neutralings|Rusnegative").any()
    for t, s, e, tgt, m in zip(pairs["text"], pairs["start"], pairs["end"], pairs["target"], pairs["span_match"]):
        if m == "exact":
            assert t[s:e] == tgt
    assert pairs["polarity"].isin(["negative", "neutral", "positive"]).all()
    assert len(pairs) > 14_000 and pairs["text_id"].nunique() > 10_600
