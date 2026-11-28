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


def test_pipeline_reloads_runs_and_writes_triplets(tmp_path):
    import json

    import numpy as np
    import pandas as pd
    import torch
    from absa_models import lstm
    from absa_models.pair_classify import ASPECTS, SENTIMENTS, PairClassifier
    from absa_models.pipeline import run_pipeline
    from absa_models.target_extraction import Tagger, tag_set
    texts = pd.DataFrame({"text_id": ["a", "b"], "source": "finentity",
                          "text": ["Vingroup rose on Monday", "FPT fell"]})
    tok, enc = lstm.build(texts["text"], [], glove=([], np.zeros((0, 300), dtype=np.float32)), log=lambda *a: None)
    runs = {}
    for name, model, cfg in [
        ("ext", Tagger("lstm", 3, True, encoder=enc), {"scheme": "bio", "crf": True}),
        ("sent", PairClassifier("lstm", len(SENTIMENTS), len(tok), encoder=lstm.LSTMEncoder(
            torch.zeros(len(tok), 300))), {"task": "sentiment"}),
        ("asp", PairClassifier("lstm", len(ASPECTS), len(tok), encoder=lstm.LSTMEncoder(
            torch.zeros(len(tok), 300))), {"task": "aspect"}),
    ]:
        d = tmp_path / name
        lstm.save_vocab(tok, d)
        torch.save(model.state_dict(), d / "best.pt")
        (d / "metrics.json").write_text(json.dumps({"config": {"model": "lstm", "max_length": 32, **cfg}}))
        runs[name] = d
    out = run_pipeline(texts, runs["ext"], runs["sent"], runs["asp"])
    assert set(out) == {"a", "b"}
    for tid, trips in out.items():
        text = texts.set_index("text_id").at[tid, "text"]
        for t in trips:
            assert text[t["start"]:t["end"]] == t["target"]
            assert t["sentiment"] in SENTIMENTS and t["aspect"] in ASPECTS


def test_multitask_learns_and_predicts_triplets(tok):
    import torch
    from absa_models.multitask import MultiTask, batches, encode, predict_all
    from absa_models.pair_classify import ASPECTS
    exs = [{"text_id": str(i), "source": "x", "text": t,
            "spans": [(0, t.index(" "), "positive", [0.0, 0.1, 0.9], ASPECTS.index("Stock/Price Movement"))]}
           for i, t in enumerate(["Vingroup rose sharply", "FPT gained today", "Gold rose on demand"])]
    items = encode(tok, exs, 32)
    assert all(len(it["gold"]) == 1 and it["labels"].count(1) == 1 for it in items)
    torch.manual_seed(0)
    model = MultiTask(TINY, soft=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    b = next(batches(items, 8, tok.pad_token_id, shuffle=False))
    first = model.loss(b).item()
    for _ in range(60):
        opt.zero_grad()
        model.loss(b).backward()
        opt.step()
    assert model.loss(b).item() < first / 4
    pred = predict_all(model, items, 8, tok.pad_token_id)
    for ex in exs:
        for t in pred[ex["text_id"]]:
            assert ex["text"][t["start"]:t["end"]] == t["target"] and t["aspect"] in ASPECTS
    assert sum(len(v) for v in pred.values()) >= 1


def test_aggregate_over_seeds(tmp_path):
    import json
    from absa_models.aggregate import collect, table
    for system, variant, seed, f1 in [("pipeline", "", 1, 0.70), ("pipeline", "", 2, 0.72),
                                      ("multitask", "e10-", 1, 0.71), ("multitask", "e10-", 2, 0.75)]:
        d = tmp_path / system / f"{variant}s{seed}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"test": {"all triplet": {"f1": f1}}}))
    comp = tmp_path / "sentiment" / "s1"
    comp.mkdir(parents=True)
    (comp / "metrics.json").write_text(json.dumps({"test": {"all": {"macro_f1": 0.9}}}))
    groups = collect(tmp_path)
    assert set(groups) == {("pipeline", "-"), ("multitask", "e10")}
    text = table(groups)
    assert "| pipeline | - | 2 | 0.710 ± 0.014 |" in text and "| multitask | e10 | 2 | 0.730 ± 0.028 |" in text


@pytest.mark.parametrize("name", [TINY, "hf-internal-testing/tiny-random-RobertaModel"])
def test_levitated_markers_leave_text_states_unchanged(name):
    import torch
    from absa_models.common import load_encoder, load_tokenizer
    from absa_models.multitask import levitate
    try:
        tok, enc = load_tokenizer(name), load_encoder(name).eval()
    except Exception as e:  # offline
        pytest.skip(f"{name} unavailable: {e}")
    tok.add_special_tokens({"additional_special_tokens": ["<t>", "</t>"]})
    enc.resize_token_embeddings(len(tok))
    b = tok(["Vingroup rose while VN-Index fell", "FPT gained"], padding=True, return_tensors="pt")
    spans = torch.tensor([[0, 1, 1], [0, 4, 6], [1, 1, 1]])
    with torch.no_grad():
        plain = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        ids, allow, pos, io, ic = levitate(enc, b["input_ids"], b["attention_mask"], spans,
                                           *tok.convert_tokens_to_ids(["<t>", "</t>"]), tok.pad_token_id)
        marked = enc(input_ids=ids, attention_mask=allow, position_ids=pos).last_hidden_state
    n = b["input_ids"].size(1)
    real = b["attention_mask"].bool()
    assert torch.allclose(marked[:, :n][real], plain[real], atol=1e-5)
    assert ids[0, io[1]] == tok.convert_tokens_to_ids("<t>") and pos[0, io[1]] == pos[0, 4] and pos[0, ic[1]] == pos[0, 6]
    assert io[2] == n and ic[2] == n + 1  # second sentence: its only span uses the first marker slot


def test_hier_aspect_and_ordinal_sentiment_heads_give_distributions(tok):
    import torch
    from absa_models.multitask import ASPECT_L1, L1_OF, MultiTask
    torch.manual_seed(0)
    m = MultiTask(TINY, aspect_head="hier", sentiment_loss="ordinal", sentence_context=True)
    h = m.encoder.config.hidden_size
    r, a = torch.randn(5, 2 * h), torch.randn(5, 3 * h)
    pa = m.aspect_logprobs(a).exp()
    assert torch.allclose(pa.sum(-1), torch.ones(5), atol=1e-5)
    l1 = torch.zeros(5, len(ASPECT_L1)).index_add_(1, torch.tensor(L1_OF), pa)
    assert torch.allclose(l1, torch.softmax(m.aspect_l1_head(a), -1), atol=1e-5)  # marginal = l1 head
    ps = m.sentiment_logprobs(r).exp()
    assert (ps >= 0).all() and torch.allclose(ps.sum(-1), torch.ones(5), atol=1e-4)
    target = torch.tensor([[0.0, 0.1, 0.9]] * 5)
    loss = m.sentiment_loss(r, target)
    loss.backward()
    assert torch.isfinite(loss) and m.theta.grad is not None


def test_multitask_all_options_learn(tok):
    import torch
    from absa_models.multitask import MultiTask, batches, encode, predict_all, parse_weights
    from absa_models.pair_classify import ASPECTS
    tok.add_special_tokens({"additional_special_tokens": ["<t>", "</t>"]})
    exs = [{"text_id": str(i), "source": "x", "text": t,
            "spans": [(0, t.index(" "), "positive", [0.0, 0.1, 0.9], ASPECTS.index("Stock/Price Movement"), c)]}
           for i, (t, c) in enumerate([("Vingroup rose sharply", "high"), ("FPT gained today", "medium"),
                                       ("Gold rose on demand", "low")])]
    items = encode(tok, exs, 32)
    w = parse_weights("default")
    b = next(batches(items, 8, tok.pad_token_id, shuffle=False, confidence_weights=w))
    assert b["span_aspect_weight"].tolist() == [1.0, 0.5, 0.25]
    torch.manual_seed(0)
    model = MultiTask(TINY, soft=True, aspect_head="hier", sentence_context=True, sentiment_loss="ordinal",
                      markers=True, marker_ids=tuple(tok.convert_tokens_to_ids(["<t>", "</t>"])),
                      pad_id=tok.pad_token_id, vocab_size=len(tok))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = model.loss(b).item()
    for _ in range(60):
        opt.zero_grad()
        model.loss(b).backward()
        opt.step()
    assert model.loss(b).item() < first / 4
    pred = predict_all(model, items, 8, tok.pad_token_id)
    for ex in exs:
        for t in pred[ex["text_id"]]:
            assert ex["text"][t["start"]:t["end"]] == t["target"] and t["aspect"] in ASPECTS
    assert sum(len(v) for v in pred.values()) >= 1
