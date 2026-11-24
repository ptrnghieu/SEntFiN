import json

import pandas as pd

from absa_data import vnnews as vn
from absa_data.annotate import ASPECT_TASK
from absa_data.calibration import SENTIMENT_TASK


def _sentences(tmp_path, n_docs=12, per_doc=5):
    recs = []
    for d in range(n_docs):
        doc = f"vietnamnews_202611{d + 10:02d}_{d:010x}"
        for i in range(per_doc):
            text = f"Vingroup {d} rose in session {i}." if i else f"Headline {d}"
            recs.append({"sent_id": f"{doc}#{i}", "doc_id": doc, "sent_idx": i, "text": text, "lang": "en",
                         "is_title": i == 0})
    recs.append(dict(recs[1], sent_id=recs[1]["doc_id"] + "#99", sent_idx=99))  # duplicate text, same doc
    p = tmp_path / "s.json"
    p.write_text(json.dumps(recs))
    return p


def _fake_annotations(texts, ann_dir):
    paths = vn.ann_paths(ann_dir)
    paths["targets"].parent.mkdir(parents=True)
    with paths["targets"].open("w") as f:
        for t in texts.itertuples():
            added = [{"target": "Vingroup", "type": "company", "confidence": "high"}] if "Vingroup" in t.text else []
            f.write(json.dumps({"text_id": t.text_id, "given": [], "added": added, "dropped": []}) + "\n")
    pairs = vn.target_pairs(texts, paths["targets"])
    for task, field, value in ((SENTIMENT_TASK, "sentiment", "positive"), (ASPECT_TASK, "aspect",
                                                                           "Stock/Price Movement")):
        with paths[task.name].open("w") as f:
            for u in pairs["uid"]:
                f.write(json.dumps({"uid": u, field: value, "confidence": "high", "model": vn.MODEL,
                                    "prompt_version": task.version}) + "\n")


def test_build_tables_split_and_review(tmp_path):
    texts = vn.load_sentences(_sentences(tmp_path))
    _fake_annotations(texts, tmp_path / "ann")
    manifest = tmp_path / "split.csv"
    pairs, out_texts = vn.build_tables(texts, tmp_path / "ann", manifest, n_test=15)
    # whole articles per split, ~15 test sentences
    assert out_texts.groupby("paragraph_id")["split"].nunique().max() == 1
    assert 15 <= (out_texts["split"] == "test").sum() <= 21
    assert set(out_texts["split"]) == {"train", "valid", "test"}
    # spans index the original text; labels and soft labels attached
    row = pairs.iloc[0]
    assert row["text"][int(row["start"]):int(row["end"])] == row["target"]
    assert pairs["polarity"].eq("positive").all() and pairs["aspect_l1"].eq("Stock").all()
    assert (pairs["p_positive"] > 0.9).all() and pairs["label_type"].eq("silver").all()
    assert out_texts.loc[out_texts["text_type"] == "headline", "n_targets"].eq(0).all()
    # the frozen manifest reproduces the split
    _, again = vn.build_tables(texts, tmp_path / "ann", manifest, n_test=999)
    assert again["split"].tolist() == out_texts["split"].tolist()

    asg = vn.review_assignment(out_texts, n_reviewers=3, n_overlap=5)
    overlap = set(asg.pop("overlap"))
    test_ids = set(out_texts.loc[out_texts["split"] == "test", "text_id"])
    assert overlap and all(overlap <= set(ids) for ids in asg.values())
    assert set().union(*map(set, asg.values())) == test_ids
    res = vn.write_review_set(pairs, out_texts, tmp_path / "rev", n_reviewers=3, n_overlap=5)
    sheet = pd.read_excel(res["R1"]["file"], sheet_name="review")
    assert list(sheet.columns) == vn.REVIEW_COLUMNS
    assert set(sheet["sent_id"]) == set(asg["R1"])
