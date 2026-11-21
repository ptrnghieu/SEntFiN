"""Generate a Markdown data profile of the unified tables (deterministic output)."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .entities import surface_key
from .schema import LABELS, SPLITS
from .text import dedup_key

_TOK = re.compile(r"\w+")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def split_table(p: pd.DataFrame) -> str:
    rows = []
    for split in (*SPLITS, "all"):
        d = p if split == "all" else p[p["split"] == split]
        t = d.drop_duplicates("text_id")
        rows.append({
            "split": split,
            "texts": len(t),
            "pairs": len(d),
            **{lab: _pct(d["polarity"].eq(lab).mean()) for lab in LABELS},
            "targets/text": f"{len(d) / len(t):.2f}",
            "multi-target texts": _pct((t["n_targets"] > 1).mean()),
            "mixed-polarity texts": _pct(t["is_mixed"].mean()),
            "words/text": f"{t['text'].str.split().str.len().mean():.1f}",
        })
    return md_table(rows)


def baselines(p: pd.DataFrame) -> str:
    """Trivial predictors on the test split. A high entity-prior score = shortcut risk."""
    tr, te = p[p["split"] == "train"], p[p["split"] == "test"]
    majority = tr["polarity"].value_counts().index[0]
    ent_major = tr.groupby("entity_id")["polarity"].agg(lambda s: s.value_counts().index[0])
    pred_maj = pd.Series(majority, index=te.index)
    pred_ent = te["entity_id"].map(ent_major).fillna(majority)
    seen = te["entity_seen_in_train"].astype(bool)

    def score(y, yhat):
        return f"{accuracy_score(y, yhat):.3f}", f"{f1_score(y, yhat, labels=list(LABELS), average='macro', zero_division=0):.3f}"

    rows = []
    for name, pred, mask in [
        (f"majority class ({majority})", pred_maj, slice(None)),
        ("entity prior (train-majority label of the entity)", pred_ent, slice(None)),
        ("entity prior, test pairs with entity seen in train", pred_ent, seen),
        ("entity prior, test pairs with unseen entity", pred_ent, ~seen),
    ]:
        y, yhat = te.loc[mask, "polarity"], pred.loc[mask]
        acc, f1 = score(y, yhat) if len(y) else ("-", "-")
        rows.append({"baseline": name, "test pairs": len(y), "accuracy": acc, "macro-F1": f1})
    return md_table(rows) + f"\nTest pairs whose entity occurs in train: **{_pct(seen.mean())}**.\n"


def class_weights(p: pd.DataFrame) -> str:
    tr = p[p["split"] == "train"]
    counts = tr["polarity"].value_counts()
    w = {lab: len(tr) / (len(LABELS) * counts.get(lab, 1)) for lab in LABELS}
    return ", ".join(f"{lab}={w[lab]:.2f}" for lab in LABELS)


def span_table(p: pd.DataFrame) -> str:
    ct = p["span_match"].value_counts()
    return md_table([{"span_match": k, "pairs": int(v), "share": _pct(v / len(p))} for k, v in ct.items()])


def issue_table(issues: pd.DataFrame, source: str) -> str:
    d = issues[issues["source"] == source]
    ct = d["issue"].value_counts()
    return md_table([{"issue": k, "count": int(v)} for k, v in ct.items()])


def fiqa_section(p: pd.DataFrame) -> str:
    out = ["**Text types per split**\n", md_table([
        {"split": s, **{t: int(n) for t, n in p[p["split"] == s].drop_duplicates("text_id")["text_type"].value_counts().items()}}
        for s in SPLITS
    ])]
    l1 = p["aspect_l1"].value_counts()
    out += ["\n**Aspect level 1**\n", md_table([{"aspect_l1": k, "pairs": int(v)} for k, v in l1.items()])]
    ct = pd.crosstab(p["aspect_l2"], p["split"]).reindex(columns=list(SPLITS), fill_value=0)
    out.append(
        f"\n{ct.shape[0]} level-2 aspects after normalization; "
        f"{int((ct['train'] == 0).sum())} never appear in train, "
        f"{int((ct['train'] < 5).sum())} have fewer than 5 train examples.\n"
    )
    out += ["\n**Aspect level 2 × split**\n", md_table(
        [{"aspect_l2": k, **{s: int(ct.loc[k, s]) for s in SPLITS}} for k in ct.sum(axis=1).sort_values(ascending=False).index]
    )]
    by_type = p.groupby("text_type")["polarity"].value_counts(normalize=True).unstack().reindex(columns=list(LABELS), fill_value=0)
    out += ["\n**Polarity by text type**\n", md_table(
        [{"text_type": t, **{lab: _pct(by_type.loc[t, lab]) for lab in LABELS}} for t in by_type.index]
    )]
    return "\n".join(out)


def _vocab(texts) -> set[str]:
    return {w for t in texts for w in _TOK.findall(t.lower())}


def _oov(texts, vocab) -> float:
    toks = [w for t in texts for w in _TOK.findall(t.lower())]
    return sum(w not in vocab for w in toks) / max(1, len(toks))


def cross_section(sf: pd.DataFrame, fq: pd.DataFrame) -> str:
    sf_tr, fq_tr = sf[sf["split"] == "train"], fq[fq["split"] == "train"]
    sf_te, fq_te = sf[sf["split"] == "test"], fq[fq["split"] == "test"]
    v_sf = _vocab(sf_tr.drop_duplicates("text_id")["text"])
    v_fq = _vocab(fq_tr.drop_duplicates("text_id")["text"])
    ents_sf = set(sf_tr["target"].map(surface_key))
    ents_fq = set(fq_tr["target"].map(surface_key))
    shared_texts = len(set(sf["text"].map(dedup_key)) & set(fq["text"].map(dedup_key)))
    rows = [
        {"test set": "SEntFiN test", "OOV vs SEntFiN train vocab": _pct(_oov(sf_te.drop_duplicates("text_id")["text"], v_sf)),
         "OOV vs FiQA train vocab": _pct(_oov(sf_te.drop_duplicates("text_id")["text"], v_fq)),
         "targets seen in SEntFiN train": _pct(sf_te["target"].map(surface_key).isin(ents_sf).mean()),
         "targets seen in FiQA train": _pct(sf_te["target"].map(surface_key).isin(ents_fq).mean())},
        {"test set": "FiQA test", "OOV vs SEntFiN train vocab": _pct(_oov(fq_te.drop_duplicates("text_id")["text"], v_sf)),
         "OOV vs FiQA train vocab": _pct(_oov(fq_te.drop_duplicates("text_id")["text"], v_fq)),
         "targets seen in SEntFiN train": _pct(fq_te["target"].map(surface_key).isin(ents_sf).mean()),
         "targets seen in FiQA train": _pct(fq_te["target"].map(surface_key).isin(ents_fq).mean())},
    ]
    return (
        md_table(rows)
        + f"\nTrain vocabulary sizes: SEntFiN {len(v_sf):,} word types, FiQA {len(v_fq):,}. "
        f"Near-identical texts shared by the two datasets: {shared_texts}.\n"
    )


def build_report(pairs: pd.DataFrame, issues: pd.DataFrame, meta: dict) -> str:
    parts = ["# Data profile\n",
             "_Generated by `python -m absa_data build`; do not edit by hand._\n",
             md_table([{"setting": k, "value": f"`{v}`"} for k, v in meta.items()])]
    for source, title in (("sentfin", "SEntFiN 1.0"), ("fiqa", "FiQA 2018 Task 1 (TheFinAI/en-fiqa-absa)")):
        p = pairs[pairs["source"] == source]
        if p.empty:
            continue
        parts += [f"\n## {title}\n", "### Size and label distribution\n", split_table(p),
                  f"\nInverse-frequency class weights (train): {class_weights(p)}\n",
                  "\n### Target alignment\n", span_table(p),
                  "\n### Data-quality issues found and handled\n", issue_table(issues, source),
                  "\n### Trivial baselines on test\n", baselines(p)]
        if source == "fiqa":
            parts += ["\n### FiQA specifics\n", fiqa_section(p)]
    if pairs["source"].nunique() == 2:
        parts += ["\n## Cross-dataset domain shift\n",
                  cross_section(pairs[pairs["source"] == "sentfin"], pairs[pairs["source"] == "fiqa"])]
    return "\n".join(parts)
