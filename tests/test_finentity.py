import json

import pandas as pd

from absa_data.finentity import clean_paragraph, load_finentity, sentence_spans
from absa_data.sentsplit import normalize, split_sentences


def test_vendored_splitter_behaviour_is_pinned():
    # Same behaviour as the crawler (minhhai205/aspect-based-FSA @5b5d404): abbreviations, initialisms
    # and decimals never end a sentence; spans index the normalized text.
    t = normalize("Apple Inc. said on Monday it would buy U.S. firm X. Shares rose 3.5% to $10.2. “Good” — yes.")
    assert t == 'Apple Inc. said on Monday it would buy U.S. firm X. Shares rose 3.5% to $10.2. "Good" - yes.'
    assert [t[s:e] for s, e in split_sentences(t)] == [
        "Apple Inc. said on Monday it would buy U.S. firm X. Shares rose 3.5% to $10.2.", '"Good" - yes.']


def test_clean_paragraph_repairs_offsets_and_strips_codes():
    content = "Urban Outfitters <URBN.O> fell 3%. URBN said sales slowed [nL4N2Z028U]"
    anns = [{"start": 0, "end": 16, "value": "Urban Outfitters", "tag": "Negative"},
            {"start": 33, "end": 37, "value": "URBN", "tag": "Negative"}]  # shifted by 2: real start is 35
    text, ents, flags = clean_paragraph(content, anns)
    assert text == "Urban Outfitters fell 3%. URBN said sales slowed"
    assert [(text[e["start"]:e["end"]], e["repaired"]) for e in ents] == [("Urban Outfitters", False), ("URBN", True)]
    assert "reuters_codes_removed" in flags


def test_sentence_spans_never_cut_an_entity():
    text = "Shares of Foo Holdings Co. Ltd. rose. Bar fell sharply today."
    ents = [{"start": 10, "end": 31}]  # "Foo Holdings Co. Ltd."
    spans = sentence_spans(text, ents)
    assert any(s <= 10 and 31 <= e for s, e in spans)


def test_load_finentity_keeps_empty_sentences(tmp_path):
    data = [{"content": "Tesla rose 2%. Analysts were not surprised by the move.",
             "annotations": [{"start": 0, "end": 5, "value": "Tesla", "tag": "Positive", "label": "Positive"}]}]
    f = tmp_path / "FinEntity.json"
    f.write_text(json.dumps(data))
    pairs, texts, issues = load_finentity(f)
    assert texts["text"].tolist() == ["Tesla rose 2%.", "Analysts were not surprised by the move."]
    assert pairs["text_id"].tolist() == ["finentity-0-0"] and pairs["target_annotation"].eq("exhaustive").all()
    assert (texts["paragraph_id"] == "finentity-0").all()
    assert pairs.loc[0, "polarity"] == "positive" and pairs.loc[0, "start"] == 0 and pairs.loc[0, "end"] == 5
