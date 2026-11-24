import json
from pathlib import Path

import pandas as pd
import pytest

from absa_data import cli
from absa_data.export import instruct_records, sentence_records
from absa_data.sentfin import load_sentfin
from absa_data.splits import (apply_manifest, assign_groups, check_no_leakage, mark_entity_seen,
                              save_manifest, stratified_group_split)

REPO = Path(__file__).resolve().parents[1]
needs_csv = pytest.mark.skipif(not (REPO / "SEntFiN.csv").exists(), reason="SEntFiN.csv not present")


@pytest.fixture(scope="module")
def sentfin_split():
    pairs, _ = load_sentfin(REPO / "SEntFiN.csv")
    pairs["group_id"] = assign_groups(pairs)
    pairs["split"] = stratified_group_split(pairs, seed=42)
    return pairs


@needs_csv
def test_split_is_grouped_stratified_and_proportional(sentfin_split):
    p = sentfin_split
    check_no_leakage(p)  # no text or near-duplicate group in two splits
    share = p["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.8) < 0.02 and abs(share["test"] - 0.1) < 0.02
    dist = pd.crosstab(p["split"], p["polarity"], normalize="index")
    assert (dist.max() - dist.min()).max() < 0.02  # same label mix in every split


@needs_csv
def test_split_is_deterministic(sentfin_split):
    again = stratified_group_split(sentfin_split, seed=42)
    assert (again == sentfin_split["split"]).all()


@needs_csv
def test_manifest_roundtrip_and_detects_changed_data(sentfin_split, tmp_path):
    path = tmp_path / "m.csv"
    save_manifest(sentfin_split, path)
    assert (apply_manifest(sentfin_split, path) == sentfin_split["split"]).all()
    extra = pd.concat([sentfin_split, sentfin_split.head(1).assign(text_id="sentfin-new")])
    with pytest.raises(ValueError, match="resplit"):
        apply_manifest(extra, path)


@needs_csv
def test_committed_manifest_matches_current_csv():
    pairs, _ = load_sentfin(REPO / "SEntFiN.csv")
    apply_manifest(pairs, REPO / "splits" / "sentfin_split_v1.csv")  # raises if stale


def test_leakage_check_catches_straddling_group():
    p = pd.DataFrame({"source": ["sentfin"] * 2, "text_id": ["a", "b"], "group_id": ["g", "g"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(p)


def test_entity_seen_in_train():
    p = pd.DataFrame({"source": ["s", "s", "s", "f"], "split": ["train", "test", "test", "test"],
                      "entity_id": ["txt:a", "txt:a", "txt:b", "txt:a"]})
    assert mark_entity_seen(p).tolist() == [True, True, False, False]


def _toy_pairs():
    return pd.DataFrame({
        "uid": ["t1#0", "t1#1"], "source": "sentfin", "text_id": "t1", "text_type": "headline",
        "text": "Gold shines; Silver dull", "target": ["Silver", "Gold"],
        "start": pd.array([13, 0], dtype="Int64"), "end": pd.array([19, 4], dtype="Int64"),
        "polarity": ["negative", "positive"], "score": float("nan"),
        "aspect": ["Market/Commodities", None], "aspect_source": ["human", None], "fiqa_aspect": None,
    })


def test_sentence_and_instruct_records_are_ordered_by_position():
    sent = list(sentence_records(_toy_pairs()))
    assert [t["target"] for t in sent[0]["targets"]] == ["Gold", "Silver"]
    assert sent[0]["targets"][0]["score"] is None  # NaN -> null in JSON
    rec = list(instruct_records(_toy_pairs(), "atsc"))[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert json.loads(rec["messages"][2]["content"]) == [
        {"target": "Gold", "sentiment": "positive"}, {"target": "Silver", "sentiment": "negative"}]
    assert '["Gold", "Silver"]' in rec["messages"][1]["content"]


@needs_csv
def test_cli_build_sentfin_only(tmp_path):
    cli.main(["build", "--no-fiqa", "--no-finentity", "--sentfin-csv", str(REPO / "SEntFiN.csv"),
              "--entity-list", str(REPO / "entity_list_comprehensive.csv"),
              "--split-manifest", str(REPO / "splits" / "sentfin_split_v1.csv"),
              "--out", str(tmp_path / "out"), "--report", str(tmp_path / "report.md")])
    out = tmp_path / "out" / "sentfin"
    for split in ("train", "valid", "test"):
        for kind in ("pairs", "sentences", "instruct_atsc", "instruct_e2e"):
            line = (out / f"{kind}_{split}.jsonl").open().readline()
            assert json.loads(line)
    pq = pd.read_parquet(out / "pairs.parquet")
    assert set(pq["split"]) == {"train", "valid", "test"}
    assert "## SEntFiN 1.0" in (tmp_path / "report.md").read_text()
