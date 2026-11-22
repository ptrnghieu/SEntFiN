import json

import pandas as pd
import yaml

from absa_data.hub import stage, write_staging


def _processed(tmp_path, with_fiqa=True):
    root = tmp_path / "processed"
    for source in ["sentfin"] + (["fiqa", "combined"] if with_fiqa else []):
        d = root / source
        d.mkdir(parents=True)
        rows = []
        for split in ("train", "valid", "test"):
            for k in range(2):
                rows.append({"uid": f"{source}-{split}-{k}#0", "source": "fiqa" if source == "fiqa" else "sentfin",
                             "split": split, "polarity": "neutral", "aspect": "Other", "aspect_l1": "Other",
                             "aspect_source": "llm:m@v", "fiqa_aspect": None})
            for kind in ("sentences", "instruct_atsc"):
                (d / f"{kind}_{split}.jsonl").write_text(json.dumps({"id": 1}) + "\n")
        pd.DataFrame(rows).to_parquet(d / "pairs.parquet")
    (root / "issues.csv").write_text("source,raw_id,issue,detail\n")
    return root


def test_stage_layout_and_configs(tmp_path):
    configs = stage(_processed(tmp_path), tmp_path / "reports", tmp_path / "staging")
    names = [c["config_name"] for c in configs]
    assert names == ["sentfin", "sentfin_sentences", "sentfin_instruct_atsc", "fiqa", "fiqa_sentences",
                     "fiqa_instruct_atsc", "combined", "combined_sentences", "combined_instruct_atsc"]
    assert [d["split"] for d in configs[0]["data_files"]] == ["train", "validation", "test"]
    assert configs[0].get("default") is True
    for c in configs:
        for d in c["data_files"]:
            assert (tmp_path / "staging" / d["path"]).exists()
    assert len(pd.read_parquet(tmp_path / "staging" / "sentfin/pairs/validation.parquet")) == 2
    assert (tmp_path / "staging" / "docs" / "issues.csv").exists()
    # One file format for the whole repo, and one schema per config across its splits.
    import pyarrow.parquet as pq
    for c in configs:
        paths = [tmp_path / "staging" / d["path"] for d in c["data_files"]]
        assert all(p.suffix == ".parquet" for p in paths)
        schemas = [pq.read_schema(p) for p in paths]
        assert all(sc.equals(schemas[0]) for sc in schemas)


def test_jsonl_configs_keep_nested_fields_and_null_columns(tmp_path):
    root = _processed(tmp_path, with_fiqa=False)
    recs = {"train": [{"id": "a", "label_confidence": 0.9, "messages": [{"role": "user", "content": "x"}],
                       "extra": None}],
            "valid": [{"id": "b", "label_confidence": None, "messages": [{"role": "user", "content": "y"}],
                       "extra": "s"}],
            "test": [{"id": "c", "label_confidence": 0.5, "messages": [], "extra": None}]}
    for split, rs in recs.items():
        (root / "sentfin" / f"instruct_atsc_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n")
    stage(root, tmp_path / "reports", tmp_path / "staging")
    import pyarrow.parquet as pq
    t = pq.read_table(tmp_path / "staging/sentfin/instruct_atsc/train.parquet")
    assert t.to_pylist() == recs["train"]
    assert str(t.schema.field("extra").type) == "string"  # typed from all splits, not null


def _front_matter(card: str) -> dict:
    return yaml.safe_load(card.split("---\n")[1])


def test_card_is_gated_when_it_contains_fiqa(tmp_path):
    staging = write_staging(_processed(tmp_path), tmp_path / "reports", tmp_path / "staging", "me/ds",
                            "https://github.com/x/y/tree/abc", tmp_path / "missing.json")
    card = (staging / "README.md").read_text()
    fm = _front_matter(card)
    assert fm["license"] == "mit" and "extra_gated_prompt" in fm and "FinBen" in fm["extra_gated_prompt"]
    assert list(fm["extra_gated_fields"].values()) == ["checkbox"]
    assert [c["config_name"] for c in fm["configs"]][0] == "sentfin"
    assert 'load_dataset("me/ds", "sentfin")' in card and "{repo_id}" not in card
    assert "| `sentfin` | SEntFiN 1.0 |" in card


def test_card_without_fiqa_is_not_gated(tmp_path):
    staging = write_staging(_processed(tmp_path, with_fiqa=False), tmp_path / "reports", tmp_path / "staging",
                            "me/ds", "url", tmp_path / "missing.json")
    fm = _front_matter((staging / "README.md").read_text())
    assert "extra_gated_prompt" not in fm and [c["config_name"] for c in fm["configs"]] == [
        "sentfin", "sentfin_sentences", "sentfin_instruct_atsc"]
