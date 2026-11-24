import json
import re
from types import SimpleNamespace

import pandas as pd
import pytest

import absa_data.completion as c
from absa_data.annotate import Item


def test_clean_added_enforces_rules():
    text = "Brent drops below $87 as the market eyes Fed; dollar lifts, Brent steady"
    kept, dropped = c.clean_added(text, ["Brent"], [
        {"target": "Fed", "type": "regulator_government", "confidence": "high"},
        {"target": "dollar", "type": "currency", "confidence": "high"},
        {"target": "brent", "type": "commodity", "confidence": "low"},      # repeats gold
        {"target": "$87", "type": "other", "confidence": "low"},            # no letters
        {"target": "the market", "type": "index_market", "confidence": "low"},  # unlinkable
        {"target": "Federal Reserve", "type": "regulator_government", "confidence": "low"},  # not verbatim
        {"target": "Fed", "type": "regulator_government", "confidence": "low"},  # duplicate
    ])
    assert [k["target"] for k in kept] == ["Fed", "dollar"]
    assert len(dropped) == 5


def test_cashtag_becomes_ticker_and_overlap_with_gold_is_dropped():
    kept, _ = c.clean_added("$TSLA and $NIO up; Tesla Motors flat", ["Tesla Motors"],
                            [{"target": "$NIO", "type": "company", "confidence": "high"},
                             {"target": "Tesla", "type": "company", "confidence": "high"}])
    assert [k["target"] for k in kept] == ["NIO"]  # "Tesla" only occurs inside the gold "Tesla Motors"


def _resp(payload):
    return SimpleNamespace(output_text=json.dumps(payload), usage=None)


def fake_client(answer):
    calls = []

    def create(**kw):
        calls.append(kw)
        return answer(kw)
    return SimpleNamespace(responses=SimpleNamespace(create=create), calls=calls)


def answer_all(kw):
    texts = []
    for block in re.split(r"\n(?=T\d+ \()", kw["input"]):
        key = re.match(r"T(\d+)", block).group(0)
        given = [{"key": k, "target": json.loads(t), "type": "company"}
                 for k, t in re.findall(r"(T\d+\.\d+): (\".*\")", block)]
        added = [{"target": "Fed", "type": "regulator_government", "confidence": "high"}] if "Fed" in block else []
        texts.append({"key": key, "given": given, "added": added})
    return _resp({"texts": texts})


def test_run_types_gold_adds_targets_and_resumes(tmp_path):
    items = [Item("t1", "Brent drops as Fed lifts dollar", "headline", [("t1#0", "Brent")]),
             Item("t2", "Nothing here", "headline", [])]
    client = fake_client(answer_all)
    out = tmp_path / "c.jsonl"
    stats = c.TargetCompleter(client, model="m").run(items, out, batch_size=5, workers=1, log=lambda *_: None)
    assert stats["texts_done"] == 2 and stats["added"] == 1
    recs = c.latest_completions(out).set_index("text_id")
    assert recs.loc["t1", "given"] == [{"uid": "t1#0", "target": "Brent", "type": "company"}]
    assert recs.loc["t1", "added"][0]["target"] == "Fed" and recs.loc["t2", "added"] == []
    c.TargetCompleter(client, model="m").run(items, out, log=lambda *_: None)
    assert len(client.calls) == 1  # cached


def test_misaligned_given_target_is_rejected():
    items = [Item("t1", "A and B", "headline", [("t1#0", "A")])]
    bad = json.dumps({"texts": [{"key": "T1", "given": [{"key": "T1.1", "target": "B", "type": "company"}],
                                 "added": []}]})
    with pytest.raises(ValueError, match="misaligned"):
        c.parse(bad, items)


def test_add_completed_targets_creates_llm_rows(tmp_path):
    (tmp_path / "completion").mkdir()
    rec = {"text_id": "sentfin-1", "given": [{"uid": "sentfin-1#0", "target": "Brent", "type": "commodity"}],
           "added": [{"target": "Fed", "type": "regulator_government", "confidence": "high"},
                     {"target": "the market", "type": "index_market", "confidence": "low"}],
           "dropped": [], "model": "m", "prompt_version": c.PROMPT_VERSION, "reasoning_effort": "low"}
    (tmp_path / "completion" / "sentfin_m.jsonl").write_text(json.dumps(rec) + "\n")
    pairs = pd.DataFrame([{"uid": "sentfin-1#0", "source": "sentfin", "text_id": "sentfin-1", "split": "train",
                           "text": "Brent drops as Fed eyes the market", "text_type": "headline", "target": "Brent",
                           "target_source": "gold", "target_annotation": "partial", "target_type": None,
                           "start": 0, "end": 5}])
    texts = pd.DataFrame([{"text_id": "sentfin-1", "target_annotation": "partial"}])
    out, texts, info = c.add_completed_targets(pairs, texts, tmp_path)
    new = out[out["target_source"] == "llm"]
    assert new["target"].tolist() == ["Fed"] and new["uid"].tolist() == ["sentfin-1#x0"]  # unlinkable filtered
    assert (new["start"].iloc[0], new["end"].iloc[0]) == (15, 18)
    assert out.loc[0, "target_type"] == "commodity"
    assert set(out["target_annotation"]) == {"llm_completed"} and texts.loc[0, "target_annotation"] == "llm_completed"


def test_validation_scoring():
    recs = pd.DataFrame([{"text_id": "a", "added": [{"target": "Cineworld Group", "type": "company"},
                                                    {"target": "dollar", "type": "currency"}]},
                         {"text_id": "b", "added": [{"target": "Refinitiv", "type": "company"}]}])
    res = c.score_validation(recs, {"a": ["Cineworld Group", "Spirit"], "b": []}, {"a": ["x"], "b": []})
    assert res["recall_exact"] == 0.5 and res["precision"] == pytest.approx(1 / 3)
    assert res["precision_in_finentity_scope"] == 0.5 and res["additions_on_empty_texts"] == 1
