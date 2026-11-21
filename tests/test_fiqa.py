import pandas as pd
import pytest

from absa_data.fiqa import load_fiqa, normalize_aspect, score_to_polarity


@pytest.mark.parametrize("score,label", [
    (-0.938, "negative"), (-0.101, "negative"), (-0.1, "neutral"), (0.0, "neutral"),
    (0.099, "neutral"), (0.1, "positive"), (0.975, "positive"),
])
def test_score_to_polarity_matches_finben_boundaries(score, label):
    assert score_to_polarity(score) == label


@pytest.mark.parametrize("raw,fixed,flag", [
    ("Stock/Price Action/Bullish", "Stock/Price Action/Bullish", None),
    ("Market/Volatilit", "Market/Volatility", "aspect_normalized"),
    ("Corporate/M&A/", "Corporate/M&A", "aspect_normalized"),
    ("Economy/Central Banks/?", "Economy/Central Banks", "aspect_normalized"),
    ("Stock/Price Action/Bearish/Bearish Behaviour", "Stock/Price Action/Bearish/Bearish Behavior", "aspect_normalized"),
    ("Stock/Price Action/Bullish', 'Corporate/Financial/Financial Results/Earnings",
     "Stock/Price Action/Bullish", "aspect_multiple_kept_first"),
])
def test_normalize_aspect(raw, fixed, flag):
    got, flags = normalize_aspect(raw)
    assert got == fixed
    assert flags == ([flag] if flag else [])


@pytest.fixture
def fiqa_dir(tmp_path):
    rows = {
        "train": [
            ("1", "CompaniesTesco off to a bad start over Xmas â€“ Kantar", "Tesco", "Corporate/Sales", -0.4, "headline"),
            ("2", "UK housing market steadies after Brexit dip, Persimmon says", "Perssimon", "Market/Market", 0.3, "headline"),
            ("3", "Juniper tumbles on Q1 warning; Cisco also lower $JNPR $CSCO https://t.co/a", "JNPR", "Stock/Price Action", -0.37, "post"),
            ("3", "Juniper tumbles on Q1 warning; Cisco also lower $JNPR $CSCO https://t.co/a", "CSCO", "Stock/Price Action", 0.05, "post"),
        ],
        "valid": [("4", "StanChart and RBS struggle in stress tests", "Standard Chartered", "Corporate/Risks", -0.5, "headline")],
        "test": [("5", "Juniper tumbles on Q1 warning; Cisco also lower https://t.co/b $JNPR $CSCO", "JNPR", "Stock/Price Action", -0.55, "post")],
    }
    for split, rs in rows.items():
        pd.DataFrame(rs, columns=["_id", "sentence", "target", "aspect", "score", "type"]).to_parquet(
            tmp_path / f"{split}-00000-of-00001-abc.parquet")
    return tmp_path


def test_load_fiqa_cleans_and_flags(fiqa_dir):
    pairs, issues = load_fiqa(fiqa_dir)
    by = pairs.set_index("uid")
    tesco = by.loc["fiqa-1#0"]
    assert tesco["text"] == "Tesco off to a bad start over Xmas – Kantar"
    assert {"mojibake_repaired", "section_prefix_removed"} <= set(tesco["quality_flags"].split(";"))
    assert tesco["text"][tesco["start"]:tesco["end"]] == "Tesco"

    pers = by.loc["fiqa-2#0"]
    assert (pers["target"], pers["target_raw"], pers["span_match"]) == ("Persimmon", "Perssimon", "fuzzy")

    jun = pairs[pairs["text_id"] == "fiqa-3"]
    assert list(jun["polarity"]) == ["negative", "neutral"] and jun["is_mixed"].all()
    assert "near_duplicate_across_splits" in by.loc["fiqa-5#0", "quality_flags"]

    assert by.loc["fiqa-4#0", "span_match"] == "not_found"
    assert list(pairs["split"].unique()) == ["train", "valid", "test"]
    assert {"target_misspelling_corrected", "target_not_in_text"} <= set(issues["issue"])


def test_missing_split_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fiqa(tmp_path)
