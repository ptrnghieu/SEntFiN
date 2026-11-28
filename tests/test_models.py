"""Correctness checks for the baselines with a tiny random BERT (CPU, seconds)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchcrf")

TINY = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # offline
        pytest.skip(f"tiny model unavailable: {e}")


EXAMPLES = [
    {"text_id": "a", "source": "finentity", "text": "Tesla rose while Ford fell sharply.",
     "spans": [(0, 5, "positive"), (17, 21, "negative")]},
    {"text_id": "b", "source": "finentity", "text": "Goldman Sachs cut its outlook for Apple Inc.",
     "spans": [(0, 13, "neutral"), (34, 43, "neutral")]},
    {"text_id": "c", "source": "finentity", "text": "Analysts were not surprised by the move.", "spans": []},
]


@pytest.mark.parametrize("scheme", ["bio", "bio-sentiment"])
def test_tags_round_trip_to_gold_spans(tok, scheme):
    from absa_models.target_extraction import decode_spans, encode, tag_set
    tags = tag_set(scheme)
    for item in encode(tok, EXAMPLES, tags, scheme, 64):
        ids = [max(l, 0) for l in item["labels"]]
        got = decode_spans(item, ids, tags)
        want = {(s, e, p if scheme != "bio" else None) for s, e, p in item["spans"]}
        assert got == want, (item["text"], got)


@pytest.mark.parametrize("crf", [False, True])
def test_tagger_can_overfit(tok, crf):
    from absa_models.common import train_loop
    from absa_models.target_extraction import Tagger, batches, encode, evaluate, tag_set
    torch.manual_seed(0)
    tags = tag_set("bio")
    items = encode(tok, EXAMPLES * 4, tags, "bio", 64)
    model = Tagger(TINY, len(tags), crf)
    dev = torch.device("cpu")
    val = lambda: (evaluate(model, items, tags, 8, tok.pad_token_id, dev)[0]["all spans"]["f1"], {})  # noqa: E731
    log = train_loop(model, lambda: batches(items, 4, tok.pad_token_id, True), val, epochs=60, lr=3e-3,
                     patience=60, out=__import__("pathlib").Path("/tmp/claude-overfit-tagger"), log=lambda *_: None)
    assert log["best_valid_score"] >= 0.9


def test_pair_classifier_can_overfit_soft_targets(tok, tmp_path):
    from absa_models.common import train_loop
    from absa_models.pair_classify import MARK_CLOSE, MARK_OPEN, PairClassifier, batches, encode, predict
    tok.add_special_tokens({"additional_special_tokens": [MARK_OPEN, MARK_CLOSE]})
    df = pd.DataFrame({"text": ["Tesla rose sharply", "Ford fell hard", "Sebi issued a circular"] * 4,
                       "target": ["Tesla", "Ford", "Sebi"] * 4, "start": [0, 0, 0] * 4, "end": [5, 4, 4] * 4,
                       "polarity": ["positive", "negative", "neutral"] * 4,
                       "p_negative": [0.0, 0.9, 0.02] * 4, "p_neutral": [0.1, 0.1, 0.96] * 4,
                       "p_positive": [0.9, 0.0, 0.02] * 4})
    data = encode(tok, df, "sentiment", True, 32)
    assert "<t>" in tok.decode(data["input_ids"][0])
    torch.manual_seed(0)
    model = PairClassifier(TINY, 3, len(tok))
    idx = np.arange(len(df))
    acc = lambda: (float((predict(model, data, idx, 8, tok.pad_token_id).argmax(1) == data["y"]).mean()), {})  # noqa: E731
    log = train_loop(model, lambda: batches(data, idx, 4, tok.pad_token_id, True), acc, epochs=40, lr=3e-3,
                     patience=40, out=tmp_path, log=lambda *_: None)
    assert log["best_valid_score"] == 1.0


def test_span_metrics():
    from absa_models.common import span_metrics
    m = span_metrics([{(0, 5), (6, 9)}, set()], [{(0, 5)}, {(1, 2)}])
    assert (m["tp"], m["precision"], m["recall"]) == (1, 0.5, 0.5)


def test_cls_metrics_macro_ignores_absent_classes():
    from absa_models.common import cls_metrics
    m = cls_metrics(["a", "a", "b"], ["a", "a", "b"], ["a", "b", "c", "d"])
    assert m["macro_f1"] == 1.0 and m["n_classes"] == 2


def test_lstm_tokenizer_offsets_pairs_and_markers():
    from absa_models.lstm import WordTokenizer
    tok = WordTokenizer(["vingroup", "rose", "3.5", "%", "shares"])
    enc = tok("Vingroup rose 3.5%", return_offsets_mapping=True)
    assert enc["offset_mapping"] == [(0, 8), (9, 13), (14, 17), (17, 18)]
    assert tok.unk_id not in enc["input_ids"]
    batch = tok(["<t> Vingroup </t> rose"], ["Vingroup"], max_length=16)
    ids = batch["input_ids"][0]
    assert ids[0] == tok.stoi["<t>"] and ids[2] == tok.stoi["</t>"] and tok.sep_id in ids
    assert tok.pad_token_id in tok.all_special_ids and tok.unk_id not in tok.all_special_ids


def test_lstm_tagger_and_classifier_train(tmp_path):
    import numpy as np
    import torch
    from absa_models import lstm
    from absa_models.pair_classify import PairClassifier
    from absa_models.target_extraction import Tagger, batches, encode, tag_set
    texts = ["Vingroup rose", "FPT fell sharply", "Gold rose on demand"] * 4
    tok, enc = lstm.build(texts, [], glove=([], np.zeros((0, 300), dtype=np.float32)), log=lambda *a: None)
    examples = [{"text_id": str(i), "source": "x", "text": t, "spans": [(0, t.index(" "), "positive")]}
                for i, t in enumerate(texts)]
    items = encode(tok, examples, tag_set("bio"), "bio", 32)
    tagger = Tagger("lstm", 3, crf=True, encoder=enc)
    opt = torch.optim.Adam(tagger.parameters(), lr=1e-2)
    b = next(batches(items, 12, tok.pad_token_id, shuffle=False))
    b = {k: v for k, v in b.items() if torch.is_tensor(v)}
    first = tagger.loss(b).item()
    for _ in range(30):
        opt.zero_grad()
        tagger.loss(b).backward()
        opt.step()
    assert tagger.loss(b).item() < first / 3
    clf = PairClassifier("lstm", 3, len(tok), encoder=enc)
    x = tok(["<t> Vingroup </t> rose"], ["Vingroup"], max_length=16)
    logits = clf.logits({"input_ids": torch.tensor(x["input_ids"]), "attention_mask": torch.tensor(x["attention_mask"])})
    assert logits.shape == (1, 3)
