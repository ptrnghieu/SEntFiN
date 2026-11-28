"""Non-transformer baseline: BiLSTM over GloVe word embeddings (``--model lstm``).

* Step 1a: BiLSTM-CRF tagger (Lample et al., 2016): the same Tagger / CRF / decoding as the
  transformer runs, with this encoder underneath.
* Steps 1b / 1c: the text with the target marked ``<t> ... </t>``, then ``[SEP]`` and the target;
  a BiLSTM reads it and attention + max pooling give the vector the classifier sees.

The tokenizer mimics the part of the Hugging Face tokenizer API the scripts use (offsets, pairs,
special ids), so encoding, metrics and outputs are shared with the transformer baselines.

Embeddings: GloVe 6B 300d (lower-cased), from the Hub copy in
``sentence-transformers/average_word_embeddings_glove.6B.300d``. Vocabulary = every train word,
plus valid/test words GloVe knows (their vectors are pretrained, so no label information leaks).
Words GloVe lacks start random and are fine-tuned.
"""

from __future__ import annotations

import re
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

MODEL_NAME = "lstm"
GLOVE_REPO = "sentence-transformers/average_word_embeddings_glove.6B.300d"
PAD, UNK, SEP, T_OPEN, T_CLOSE = "[PAD]", "[UNK]", "[SEP]", "<t>", "</t>"
_TOKEN = re.compile(r"<t>|</t>|\w+(?:[.'’&-]\w+)*|[^\w\s]")


def word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


class WordTokenizer:
    """Lower-cased word tokenizer with character offsets and a fixed vocabulary."""

    def __init__(self, words: list[str]):
        self.itos = [PAD, UNK, SEP, T_OPEN, T_CLOSE] + [w for w in words if w not in (PAD, UNK, SEP, T_OPEN, T_CLOSE)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_token_id, self.unk_id, self.sep_id = 0, 1, 2
        self.all_special_ids = [self.pad_token_id, self.sep_id]

    def __len__(self) -> int:
        return len(self.itos)

    def add_special_tokens(self, _special) -> int:  # markers are built in
        return 0

    def _ids(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        spans = word_spans(text)
        ids = [self.stoi.get(w if w in (T_OPEN, T_CLOSE) else w.lower(), self.unk_id) for w, _, _ in spans]
        return ids, [(a, b) for _, a, b in spans]

    def _one(self, text, pair, max_length, offsets):
        ids, offs = self._ids(text)
        if pair is not None:
            pid, _ = self._ids(pair)
            keep = max(1, max_length - len(pid) - 1) if max_length else len(ids)
            ids, offs = ids[:keep] + [self.sep_id] + pid, offs[:keep] + [(0, 0)] * (len(pid) + 1)
        elif max_length:
            ids, offs = ids[:max_length], offs[:max_length]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if offsets:
            out["offset_mapping"] = offs
        return out

    def __call__(self, text, text_pair=None, truncation=True, max_length=None, return_offsets_mapping=False):
        if isinstance(text, str):
            return self._one(text, text_pair, max_length, return_offsets_mapping)
        pairs = text_pair if text_pair is not None else [None] * len(text)
        rows = [self._one(t, p, max_length, return_offsets_mapping) for t, p in zip(text, pairs)]
        return {k: [r[k] for r in rows] for k in rows[0]} if rows else {"input_ids": [], "attention_mask": []}


def load_glove(cache_dir=None) -> tuple[list[str], np.ndarray]:
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    vocab_f = hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/whitespacetokenizer_config.json", cache_dir=cache_dir)
    emb_f = hf_hub_download(GLOVE_REPO, "0_WordEmbeddings/model.safetensors", cache_dir=cache_dir)
    words = json.loads(open(vocab_f, encoding="utf-8").read())["vocab"]
    tensors = load_file(emb_f)
    return words, next(iter(tensors.values()))


def build(train_texts, other_texts, glove: tuple[list[str], np.ndarray] | None = None, seed: int = 42,
          log=print) -> tuple[WordTokenizer, "LSTMEncoder"]:
    """Tokenizer + encoder. `glove=None` downloads GloVe; pass ([], empty) for random embeddings (tests)."""
    words, vectors = glove if glove is not None else load_glove()
    gidx = {w: i for i, w in enumerate(words)}
    lower = lambda ts: Counter(w.lower() for t in ts for w, _, _ in word_spans(t))  # noqa: E731
    train_words = lower(train_texts)
    extra = [w for w in lower(other_texts) if w not in train_words and w in gidx]
    tok = WordTokenizer(sorted(train_words) + sorted(extra))
    dim = vectors.shape[1] if len(words) else 300
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.3, size=(len(tok), dim)).astype(np.float32)
    matrix[0] = 0.0
    hits = 0
    for i, w in enumerate(tok.itos):
        j = gidx.get(w)
        if j is not None:
            matrix[i] = vectors[j]
            hits += 1
    log(f"lstm vocab {len(tok):,} words, {hits / max(1, len(tok) - 5):.1%} covered by GloVe")
    return tok, LSTMEncoder(torch.from_numpy(matrix))


def save_vocab(tok: WordTokenizer, out_dir) -> None:
    import json
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "vocab.json").write_text(json.dumps(tok.itos, ensure_ascii=False), encoding="utf-8")


def from_saved(out_dir, dim: int = 300) -> tuple[WordTokenizer, "LSTMEncoder"]:
    """Tokenizer + an encoder of the right shape for a run's best.pt (the weights come from the state dict)."""
    import json
    from pathlib import Path
    tok = WordTokenizer(json.loads((Path(out_dir) / "vocab.json").read_text(encoding="utf-8")))
    return tok, LSTMEncoder(torch.zeros(len(tok), dim))


class LSTMEncoder(nn.Module):
    """Embedding -> 2-layer BiLSTM. Returns `last_hidden_state` (per token) and `pooled` (per text)."""

    def __init__(self, embeddings: torch.Tensor, hidden: int = 256, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.att = nn.Linear(2 * hidden, 1)
        self.proj = nn.Linear(4 * hidden, 2 * hidden)
        self.config = SimpleNamespace(hidden_size=2 * hidden)

    def resize_token_embeddings(self, n: int):
        if n > self.embedding.num_embeddings:
            raise ValueError("the LSTM vocabulary is fixed when the tokenizer is built")

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.bool()
        lengths = mask.sum(1).clamp(min=1).cpu()
        x = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        h = self.dropout(h)
        hf = h.float()
        scores = self.att(hf).squeeze(-1).masked_fill(~mask, -1e4)
        att = (torch.softmax(scores, -1).unsqueeze(-1) * hf).sum(1)
        mx = hf.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        pooled = torch.tanh(self.proj(torch.cat([att, mx], -1)))
        return SimpleNamespace(last_hidden_state=h, pooled=pooled)


# Defaults that differ from the transformer fine-tuning recipe.
DEFAULTS = {"lr": 1e-3, "tagging_epochs": 30, "tagging_patience": 5, "pair_epochs": 15, "pair_patience": 3}
