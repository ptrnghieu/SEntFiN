from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from absa_data.calibration import (SENTIMENT_EXAMPLES, SENTIMENT_TASK, add_soft_labels, apply_thresholds,
                                   attach_llm_sentiment, choose_thresholds, estimate_score_sigma, fit_thresholds,
                                   refresh_text_stats, relabel_fiqa, sentfin_duplicate_agreement, soft_from_hard,
                                   soft_from_score)
from absa_data.export import instruct_records, pair_records
from absa_data.schema import LABELS
from absa_data.text import dedup_key

REPO = Path(__file__).resolve().parents[1]
needs_csv = pytest.mark.skipif(not (REPO / "SEntFiN.csv").exists(), reason="SEntFiN.csv not present")


@pytest.mark.parametrize("score", [-0.9, -0.1, -0.05, 0.0, 0.1, 0.239, 0.24, 0.7])
def test_soft_from_score_is_a_distribution_and_sigma0_is_hard(score):
    p = soft_from_score(score, 0.08, -0.1, 0.24)
    assert abs(sum(p) - 1) < 1e-9 and min(p) >= 0
    hard = soft_from_score(score, 0, -0.1, 0.24)
    assert LABELS[int(np.argmax(hard))] == apply_thresholds(pd.Series([score]), -0.1, 0.24).iloc[0]


def test_soft_from_score_shape():
    # Far from thresholds -> near certain; at a threshold -> 50/50 between the two adjacent classes.
    assert soft_from_score(0.9, 0.08, -0.1, 0.1)[2] > 0.999
    neg, neu, pos = soft_from_score(0.1, 0.08, -0.1, 0.1)
    assert abs(neu - 0.5) < 0.02 and abs(pos - 0.5) < 0.02 and neg < 0.01
    ps = [soft_from_score(s, 0.08, -0.1, 0.1)[2] for s in np.linspace(-1, 1, 41)]
    assert all(a <= b for a, b in zip(ps, ps[1:]))  # P(positive) increases with the score


def test_soft_from_hard_only_moves_mass_to_adjacent_classes():
    assert soft_from_hard("negative", 0.1) == (0.9, 0.1, 0.0)
    assert soft_from_hard("positive", 0.1) == (0.0, 0.1, 0.9)
    assert soft_from_hard("neutral", 0.1) == (0.05, 0.9, 0.05)
    assert soft_from_hard("neutral", 0.0) == (0.0, 1.0, 0.0)


def _toy():
    return pd.DataFrame({
        "source": ["sentfin", "sentfin", "fiqa", "fiqa", "fiqa"],
        "text_id": ["s1", "s1", "f1", "f1", "f2"],
        "polarity": ["neutral", "positive", "positive", "positive", "neutral"],
        "label": [1, 2, 2, 2, 1],
        "score": [np.nan, np.nan, 0.15, 0.6, -0.05],
        "is_mixed": [True, True, False, False, False],
        "sentiment_source": "gold", "uid": ["s1#0", "s1#1", "f1#0", "f1#1", "f2#0"],
    })


def test_relabel_fiqa_keeps_finben_and_updates_dependent_columns():
    out = relabel_fiqa(_toy(), -0.1, 0.24)
    assert out["polarity"].tolist() == ["neutral", "positive", "neutral", "positive", "neutral"]
    assert out["label"].tolist() == [1, 2, 1, 2, 1]
    assert out["polarity_finben"].tolist()[2:] == ["positive", "positive", "neutral"]
    assert out["polarity_finben"].iloc[:2].isna().all()
    texts = pd.DataFrame({"text_id": ["s1", "f1", "f2", "e1"]})
    out, texts = refresh_text_stats(out, texts)
    assert out["is_mixed"].tolist() == [True, True, True, True, False]
    assert texts["n_targets"].tolist() == [2, 2, 1, 0]


def test_add_soft_labels():
    out = add_soft_labels(relabel_fiqa(_toy(), -0.1, 0.24), 0.08, 0.04, -0.1, 0.24)
    assert out["label_type"].tolist() == ["hard", "hard", "soft", "soft", "soft"]
    assert out.loc[1, ["p_negative", "p_neutral", "p_positive"]].tolist() == [0.0, 0.04, 0.96]
    probs = out[["p_negative", "p_neutral", "p_positive"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-3)
    assert (out["label_confidence"] == probs.max(axis=1)).all()
    # The hard label is the most likely class of the soft label.
    assert [LABELS[i] for i in probs.argmax(axis=1)] == out["polarity"].tolist()


def test_fit_thresholds_recovers_known_cut_points():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(-1, 1, 3000))
    y = apply_thresholds(s, -0.2, 0.3)
    fit = fit_thresholds(s, y)
    assert abs(fit["neg_below"] + 0.2) <= 0.011 and abs(fit["pos_from"] - 0.3) <= 0.011 and fit["accuracy"] == 1


def test_choose_thresholds_keeps_finben_unless_excluded():
    fit = {"neg_below": -0.01, "pos_from": 0.24}
    ci = {"neg_below_95ci": [-0.14, 0.0], "pos_from_95ci": [0.16, 0.28]}
    assert choose_thresholds(fit, ci) == {"neg_below": -0.1, "pos_from": 0.24}


def test_estimate_score_sigma_from_repeated_pairs():
    df = pd.DataFrame({"text": ["Costco play https://t.co/a", "Costco play https://t.co/b", "other"],
                       "target": ["COST", "COST", "X"], "score": [0.3, 0.5, 0.1]})
    est = estimate_score_sigma(df)
    assert est["groups"] == 1 and est["differences"] == 1 and abs(est["sigma"] - 0.2 / np.sqrt(2)) < 1e-3


@needs_csv
def test_sentfin_duplicate_agreement_on_real_data():
    d = sentfin_duplicate_agreement(REPO / "SEntFiN.csv")
    assert (d["compared"], d["conflicts"]) == (80, 7)
    assert set(d["by_pair"]) == {"negative <-> neutral"}


@needs_csv
def test_sentiment_examples_are_sentfin_train_with_gold_labels():
    from absa_data.sentfin import load_sentfin
    from absa_data.splits import apply_manifest

    pairs, _ = load_sentfin(REPO / "SEntFiN.csv")
    pairs["split"] = apply_manifest(pairs, REPO / "splits" / "sentfin_split_v1.csv")
    by_text = {k: g for k, g in pairs.groupby(pairs["text"].map(dedup_key))}
    for text, labels in SENTIMENT_EXAMPLES:
        g = by_text[dedup_key(text)]
        assert (g["split"] == "train").all(), text
        assert dict(zip(g["target"], g["polarity"])) == labels, text


def test_sentiment_task_schema():
    item = SENTIMENT_TASK.schema["properties"]["labels"]["items"]
    assert item["properties"]["sentiment"]["enum"] == list(LABELS)
    assert item["required"] == ["key", "target", "sentiment", "confidence"]
    assert SENTIMENT_TASK.version.startswith("s1-")


def test_exports_pack_soft_labels_and_confidence():
    df = add_soft_labels(relabel_fiqa(_toy(), -0.1, 0.24), 0.08, 0.04, -0.1, 0.24).assign(
        uid=lambda d: d["text_id"] + "#" + d.groupby("text_id").cumcount().astype(str),
        text_type="headline", text="t", target=["a", "b", "c", "d", "e"], start=pd.array([0] * 5, dtype="Int64"),
        end=pd.array([1] * 5, dtype="Int64"), aspect="Other")
    rec = next(pair_records(df))
    assert rec["soft_label"] == [0.02, 0.96, 0.02] and "p_neutral" not in rec  # SEntFiN neutral, flip 0.04
    ins = {r["id"]: r for r in instruct_records(df, "atsc")}
    assert ins["f1"]["label_confidence"] == df.loc[df["text_id"] == "f1", "label_confidence"].min()


def test_llm_added_targets_get_silver_sentiment(tmp_path):
    import json

    from absa_data.calibration import SENTIMENT_TASK
    pairs = _toy().assign(target=["a", "b", "c", "d", "e"])
    pairs = pd.concat([pairs, pd.DataFrame([
        {"uid": "s1#x0", "source": "sentfin", "text_id": "s1", "polarity": None, "label": pd.NA, "score": np.nan,
         "sentiment_source": "llm", "target": "Fed"},
        {"uid": "s1#x1", "source": "sentfin", "text_id": "s1", "polarity": None, "label": pd.NA, "score": np.nan,
         "sentiment_source": "llm", "target": "dollar"}])], ignore_index=True)
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"uid": "s1#x0", "text_id": "s1", "target": "Fed", "sentiment": "negative",
                             "confidence": "high", "model": "m", "prompt_version": SENTIMENT_TASK.version}) + "\n")
    out, info = attach_llm_sentiment(pairs, [f])
    assert info == {"llm_targets": 2, "with_sentiment": 1, "dropped_unlabelled": 1}
    assert "s1#x1" not in set(out["uid"])
    soft = add_soft_labels(relabel_fiqa(out, -0.1, 0.24), 0.08, 0.04, -0.1, 0.24, {"high": 0.1, "medium": 0.4})
    row = soft.set_index("uid").loc["s1#x0"]
    assert row["label_type"] == "silver" and [row["p_negative"], row["p_neutral"], row["p_positive"]] == [0.9, 0.1, 0.0]
