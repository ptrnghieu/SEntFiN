import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import absa_data.annotate as ann
from absa_data.aspects import attach_aspects
from absa_data.review import evaluate_review, read_review, sample_review_pairs, write_review_sheet
from absa_data.taxonomy import (EXAMPLES, FIQA_PREFIX_MAP, L1_LABELS, LABELS, guidelines_markdown,
                                map_fiqa_aspect)
from absa_data.text import dedup_key

REPO = Path(__file__).resolve().parents[1]


# --- taxonomy ------------------------------------------------------------------

def test_taxonomy_is_consistent():
    assert len(LABELS) == len(set(LABELS))
    assert set(L1_LABELS) == {"Corporate", "Stock", "Market", "Economy", "Other"}
    assert all(lab in LABELS for _, lab in FIQA_PREFIX_MAP if lab)
    assert all(a in LABELS for _, d in EXAMPLES for a in d.values())
    g = guidelines_markdown()
    assert all(lab in g for lab in LABELS)


@pytest.mark.parametrize("path,expected", [
    ("Corporate/M&A/Proposed Merger", "Corporate/M&A & Stakes"),
    ("Corporate/Strategy/Investment", None),
    ("Corporate/Strategy/New Product", "Corporate/Strategy & Operations"),
    ("Stock/Price Action/Volatility/Short Selling", "Stock/Trading Activity & Ownership"),
    ("Stock/Price Action/Current Price/Oversold", "Stock/Outlook & Recommendations"),
    ("Corporate/Sales/Deal", None),
    ("Corporate/Risks", None),
    ("Corporate/Risks/Product Recall", "Corporate/Risks & Reputation"),
    ("Corporate/Rumors/Scoop", None),
    ("Stock/Price Action/Bullish/Bull Position", "Stock/Outlook & Recommendations"),
    ("Stock/Price Action/Bullish", None),
    ("Stock/IPO", "Stock/IPO & Listing"),
    ("Stock/IPOX", None),                            # prefix must end at a path boundary
    (None, None),
    (float("nan"), None),
])
def test_map_fiqa_aspect(path, expected):
    assert map_fiqa_aspect(path) == expected


@pytest.mark.skipif(not (REPO / "SEntFiN.csv").exists(), reason="SEntFiN.csv not present")
def test_few_shot_examples_never_come_from_valid_or_test():
    from absa_data.sentfin import load_sentfin
    from absa_data.splits import apply_manifest

    pairs, _ = load_sentfin(REPO / "SEntFiN.csv")
    pairs["split"] = apply_manifest(pairs, REPO / "splits" / "sentfin_split_v1.csv")
    held_out = set(pairs.loc[pairs["split"] != "train", "text"].map(dedup_key))
    assert not [t for t, _ in EXAMPLES if dedup_key(t) in held_out]


# --- annotator with a fake OpenAI client ------------------------------------------

def _resp(payload):
    usage = SimpleNamespace(input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=40),
                            output_tokens=10, output_tokens_details=SimpleNamespace(reasoning_tokens=2))
    return SimpleNamespace(output_text=json.dumps(payload), usage=usage)


class FakeClient:
    def __init__(self, answer):
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)
        self._answer = answer

    def _create(self, **kw):
        self.calls.append(kw)
        return self._answer(kw)


def label_all(kw, aspect="Other"):
    keys = re.findall(r"\b(T\d+\.\d+): target (\".*\")", kw["input"])
    return _resp({"labels": [{"key": k, "target": json.loads(t), "aspect": aspect, "confidence": "high"}
                             for k, t in keys]})


def _pairs(texts):
    rows = []
    for i, (text, targets) in enumerate(texts):
        for k, t in enumerate(targets):
            rows.append({"uid": f"t{i}#{k}", "text_id": f"t{i}", "text": text, "text_type": "headline", "target": t})
    return pd.DataFrame(rows)


def test_render_and_parse_roundtrip():
    items = ann.make_items(_pairs([("Gold up; Silver dull", ["Gold", "Silver"]), ("Infosys Q2", ["Infosys"])]))
    prompt, keys = ann.render_batch(items)
    assert list(keys) == ["T1.1", "T1.2", "T2.1"] and keys["T2.1"] == ("t1#0", "t1", "Infosys")
    assert 'T1.2: target "Silver"' in prompt
    ok = {"labels": [{"key": k, "target": v[2], "aspect": "Other", "confidence": "low"} for k, v in keys.items()]}
    assert ann.parse_output(json.dumps(ok), keys)["T1.1"] == ("Other", "low")
    shifted = {"labels": [dict(lab, target=t) for lab, t in zip(ok["labels"], ["Silver", "Infosys", "Gold"])]}
    with pytest.raises(ValueError, match="misaligned"):
        ann.parse_output(json.dumps(shifted), keys)
    # Echo comparison tolerates case/whitespace differences only.
    loose = {"labels": [dict(lab, target=" " + lab["target"].upper()) for lab in ok["labels"]]}
    assert len(ann.parse_output(json.dumps(loose), keys)) == 3
    with pytest.raises(ValueError, match="missing"):
        ann.parse_output(json.dumps({"labels": ok["labels"][:2]}), keys)
    with pytest.raises(ValueError, match="duplicate"):
        ann.parse_output(json.dumps({"labels": ok["labels"] + ok["labels"][:1]}), keys)
    with pytest.raises(ValueError, match="invalid"):
        ann.parse_output(json.dumps({"labels": [{"key": "T1.1", "target": "Gold", "aspect": "Weather",
                                                  "confidence": "high"}]}), keys)


def test_request_uses_strict_schema_and_no_storage():
    client = FakeClient(label_all)
    recs, _, failed = ann.Annotator(client, model="m", reasoning_effort="low").label_batch(
        ann.make_items(_pairs([("x", ["a"])])))
    assert len(recs) == 1 and failed == []
    kw = client.calls[0]
    assert kw["store"] is False and kw["reasoning"] == {"effort": "low"} and kw["model"] == "m"
    fmt = kw["text"]["format"]
    assert fmt["strict"] is True
    assert fmt["schema"]["properties"]["labels"]["items"]["properties"]["aspect"]["enum"] == list(LABELS)


def test_run_writes_resumes_and_tracks_usage(tmp_path):
    out = tmp_path / "labels.jsonl"
    pairs = _pairs([(f"text {i}", ["A", "B"] if i % 2 else ["A"]) for i in range(7)])
    client = FakeClient(label_all)
    stats = ann.Annotator(client, model="m").run(pairs, out, batch_size=3, workers=2, log=lambda *_: None)
    assert stats["pairs_labelled"] == len(pairs) == 10 and stats["batches"] == 3
    assert stats["input_tokens"] == 300 and stats["cached_input_tokens"] == 120
    labels = ann.latest_labels(out)
    assert set(labels["uid"]) == set(pairs["uid"]) and (labels["prompt_version"] == ann.PROMPT_VERSION).all()
    # Second run: everything cached, no API call.
    stats2 = ann.Annotator(client, model="m").run(pairs, out, log=lambda *_: None)
    assert stats2["batches"] == 0 and len(client.calls) == 3
    assert len(out.with_suffix(".usage.jsonl").read_text().splitlines()) == 2


def test_reasoning_effort_is_part_of_the_cache_key(tmp_path):
    out = tmp_path / "labels.jsonl"
    pairs = _pairs([("text", ["A"])])
    client = FakeClient(label_all)
    ann.Annotator(client, model="m", reasoning_effort="low").run(pairs, out, log=lambda *_: None)
    ann.Annotator(client, model="m", reasoning_effort="low").run(pairs, out, log=lambda *_: None)
    assert len(client.calls) == 1
    ann.Annotator(client, model="m", reasoning_effort="medium").run(pairs, out, log=lambda *_: None)
    assert len(client.calls) == 2
    assert ann.latest_labels(out)["reasoning_effort"].tolist() == ["medium"]


def test_prompt_change_invalidates_cache(tmp_path):
    from dataclasses import replace

    out = tmp_path / "labels.jsonl"
    pairs = _pairs([("text", ["A"])])
    client = FakeClient(label_all)
    ann.Annotator(client, model="m").run(pairs, out, log=lambda *_: None)
    changed = replace(ann.ASPECT_TASK, version="v1-changed")
    ann.Annotator(client, model="m", task=changed).run(pairs, out, log=lambda *_: None)
    assert len(client.calls) == 2
    assert ann.latest_labels(out, "v1-changed")["prompt_version"].tolist() == ["v1-changed"]
    assert ann.latest_labels(out)["prompt_version"].tolist() == [ann.PROMPT_VERSION]


def test_bad_output_is_isolated_by_splitting_the_batch(tmp_path):
    def answer(kw):
        if "BAD" in kw["input"]:
            return _resp({"labels": []})  # never valid while the bad text is in the batch
        return label_all(kw)

    pairs = _pairs([("ok 1", ["A"]), ("BAD", ["B"]), ("ok 2", ["C"]), ("ok 3", ["D"])])
    stats = ann.Annotator(FakeClient(answer), model="m", max_attempts=2).run(
        pairs, tmp_path / "l.jsonl", batch_size=4, workers=1, log=lambda *_: None)
    assert stats["pairs_labelled"] == 3 and stats["pairs_failed"] == 1
    assert set(ann.latest_labels(tmp_path / "l.jsonl")["text_id"]) == {"t0", "t2", "t3"}


# --- joining labels onto pairs ------------------------------------------------------

def test_attach_aspects_prefers_human_and_skips_mismatched_targets(tmp_path):
    pairs = _pairs([("Gold up; Silver dull", ["Gold", "Silver"]), ("Infosys Q2", ["Infosys"])])
    llm = tmp_path / "llm.jsonl"
    with llm.open("w") as f:
        for uid, tgt, asp in [("t0#0", "Gold", "Market/Commodities"), ("t0#1", "Silver", "Market/Commodities"),
                              ("t1#0", "Wipro", "Corporate/Financial Results")]:  # stale: target changed
            f.write(json.dumps({"uid": uid, "text_id": uid[:2], "target": tgt, "aspect": asp, "confidence": "high",
                                "model": "m", "prompt_version": ann.PROMPT_VERSION}) + "\n")
    human = tmp_path / "human.csv"
    pd.DataFrame([{"uid": "t0#1", "target": "Silver", "aspect": "Other"}]).to_csv(human, index=False)
    out, info = attach_aspects(pairs, [llm], [human])
    by = out.set_index("uid")
    assert by.loc["t0#0", "aspect"] == "Market/Commodities" and by.loc["t0#0", "aspect_l1"] == "Market"
    assert by.loc["t0#0", "aspect_source"] == f"llm:m@{ann.PROMPT_VERSION}"
    assert (by.loc["t0#1", "aspect"], by.loc["t0#1", "aspect_source"]) == ("Other", "human")
    assert by.loc["t1#0", "aspect"] is None and info["target_mismatch"] == 1


# --- review sheet --------------------------------------------------------------------

def test_review_sheet_roundtrip_and_evaluation(tmp_path):
    from openpyxl import load_workbook

    pairs = _pairs([(f"headline {i}", ["A", "B"]) for i in range(6)]).assign(
        source="sentfin", split=["test"] * 8 + ["train"] * 4, polarity="neutral",
        text_id=lambda d: "sentfin-" + d["text_id"].str[1:])
    sample = sample_review_pairs(pairs, n_texts=3, seed=1)
    assert sample["text_id"].nunique() == 3 and (sample["split"] == "test").all()
    path = tmp_path / "review.xlsx"
    write_review_sheet(sample, path)

    wb = load_workbook(path)
    ws = wb["review"]
    assert [c.value for c in ws[1]][:6] == ["uid", "text_id", "text", "target", "sentiment", "aspect"]
    assert {"guidelines", "labels"} <= set(wb.sheetnames)
    for i, row in enumerate(ws.iter_rows(min_row=2), 0):
        row[5].value = "Market/Commodities" if i % 2 == 0 else "Other"
    ws.cell(row=2, column=7, value="y")
    wb.save(path)

    review = read_review(path)
    assert len(review) == len(sample) and review["unsure"].sum() == 1
    llm = sample[["uid", "target"]].assign(aspect="Market/Commodities", confidence="high", model="m",
                                           prompt_version="p")
    report, merged = evaluate_review(review, llm)
    assert "accuracy (level 2)" in report and "0.500" in report
    assert len(merged) == len(sample)


def test_read_review_rejects_unknown_labels(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame([{"uid": "a", "target": "x", "aspect": "Weather", "unsure": None, "note": None}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="outside taxonomy"):
        read_review(p)


@pytest.mark.skipif(not (REPO / "annotations/llm/sentfin_gpt-6-luna.jsonl").exists(), reason="no committed labels")
def test_committed_sentfin_labels_cover_current_pairs():
    """Fails if the loaders or the prompt changed since the committed labels were made:
    re-run `python -m absa_data annotate --source sentfin` (only changed pairs are re-labelled)."""
    from absa_data.sentfin import load_sentfin

    pairs, _ = load_sentfin(REPO / "SEntFiN.csv")
    labels = ann.latest_labels(REPO / "annotations/llm/sentfin_gpt-6-luna.jsonl").set_index("uid")
    assert set(pairs["uid"]) <= set(labels.index)
    assert (labels.loc[pairs["uid"], "target"].to_numpy() == pairs["target"].to_numpy()).all()
    assert (labels["prompt_version"] == ann.PROMPT_VERSION).all()
    assert labels["aspect"].isin(LABELS).all()
