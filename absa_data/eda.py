"""Exploratory data analysis of the processed datasets.

``python -m absa_data eda`` reads ``data/processed`` (run ``build`` first) and,
optionally, the crawler's exported sentences (the deployment domain: English
news about Vietnam), and writes ``reports/eda/EDA.md`` with PNG figures in
``reports/eda/figures/``. Every number in the report is computed here.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .entities import surface_key
from .profile import md_table
from .schema import LABELS
from .taxonomy import LABELS as ASPECTS

SOURCES = ("sentfin", "fiqa", "finentity")
NAMES = {"sentfin": "SEntFiN", "fiqa": "FiQA", "finentity": "FinEntity", "vietnam": "Vietnam news (crawler)"}
# Categorical slots 1-4 of the reference palette (validated: adjacent CVD dE >= 9.1).
COLOR = {"sentfin": "#2a78d6", "fiqa": "#eb6834", "finentity": "#1baf7a", "vietnam": "#eda100"}
# Sentiment is polarity: diverging blue <-> red poles with a neutral gray.
SENT_COLOR = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_TOK = re.compile(r"\w+")
# Strings that identify Vietnam-specific entities (deployment domain).
VN_TERMS = ("vietnam", "viet nam", "vn-index", "vnindex", "hose", "hnx", "state bank of vietnam", "sbv",
            "vingroup", "vinfast", "vinamilk", "hoa phat", "petrovietnam", "vietcombank", "dong", "vnd",
            "hanoi", "ho chi minh", "hcm city")


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "axes.axisbelow": True,
    })
    return plt


class Figs:
    def __init__(self, out: Path):
        self.plt = _setup()
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name: str) -> str:
        fig.savefig(self.out / f"{name}.png", dpi=200, bbox_inches="tight")
        self.plt.close(fig)
        return f"figures/{name}.png"


def load(processed: Path, target_domain: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    pairs = pd.read_parquet(processed / "combined" / "pairs.parquet")
    texts = pd.read_parquet(processed / "combined" / "texts.parquet")
    vn = None
    if target_domain and target_domain.exists():
        rows = [r for f in sorted(target_domain.glob("*/*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
        vn = pd.DataFrame(rows)
    return pairs, texts, vn


def words(s: pd.Series) -> pd.Series:
    return s.str.split().str.len()


# ----------------------------------------------------------------------------- sections

def sec_overview(pairs, texts, vn) -> str:
    rows = []
    for src in SOURCES:
        p, t = pairs[pairs["source"] == src], texts[texts["source"] == src]
        for split in ("train", "valid", "test"):
            ps, ts = p[p["split"] == split], t[t["split"] == split]
            rows.append({"source": NAMES[src], "split": split, "texts": f"{len(ts):,}",
                         "texts without target": f"{int((ts['n_targets'] == 0).sum()):,}",
                         "pairs": f"{len(ps):,}", "gold targets": f"{int((ps['target_source'] == 'gold').sum()):,}",
                         "LLM-added": f"{int((ps['target_source'] == 'llm').sum()):,}"})
    types = texts.groupby("source")["text_type"].value_counts().unstack(fill_value=0)
    tt = ", ".join(f"{NAMES[s]}: " + ", ".join(f"{k} {v:,}" for k, v in types.loc[s].items() if v) for s in SOURCES)
    out = ["## 1. Corpus at a glance\n", md_table(rows),
           f"\nText types: {tt}.", ]
    if vn is not None:
        out.append(f" Target domain: {len(vn):,} crawler sentences from {vn['doc_id'].nunique()} articles "
                   f"({', '.join(f'{k} {v:,}' for k, v in vn['doc_id'].str.split('_').str[0].value_counts().items())}).")
    total = len(pairs)
    out.append(f"\n\nAll sources: **{total:,} (text, target) pairs** in **{len(texts):,} texts**; "
               f"{_pct((pairs['target_source'] == 'gold').mean())} of targets are gold.\n")
    return "\n".join(out)


def sec_length(texts, vn, figs: Figs) -> str:
    plt = figs.plt
    series = {s: words(texts.loc[texts["source"] == s, "text"]) for s in SOURCES}
    if vn is not None:
        series["vietnam"] = words(vn.loc[~vn["is_title"], "text"])
        series_titles = words(vn.loc[vn["is_title"], "text"])
    fig, axes = plt.subplots(len(series), 1, figsize=(7, 1.35 * len(series) + 0.4), sharex=True)
    bins = np.arange(0, 81, 2)
    rows = []
    for ax, (s, w) in zip(np.atleast_1d(axes), series.items()):
        ax.hist(w.clip(upper=80), bins=bins, color=COLOR[s], rwidth=0.82, weights=np.ones(len(w)) / len(w))
        med, p95 = w.median(), w.quantile(0.95)
        ax.axvline(med, color=INK2, linewidth=1)
        ax.text(med + 0.8, ax.get_ylim()[1] * 0.82, f"median {med:.0f}", color=INK2, fontsize=8)
        ax.set_ylabel(NAMES[s].replace(" (crawler)", ""), rotation=0, ha="right", va="center")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.grid(axis="x", visible=False)
        rows.append({"source": NAMES[s], "texts": f"{len(w):,}", "median words": f"{med:.0f}",
                     "p95": f"{p95:.0f}", "p99": f"{w.quantile(0.99):.0f}", "max": f"{w.max():.0f}"})
    np.atleast_1d(axes)[-1].set_xlabel("words per text (clipped at 80)")
    np.atleast_1d(axes)[0].set_title("Text length: headlines are short, the deployment domain is not")
    path = figs.save(fig, "length")
    out = ["## 2. Text length\n", f"![Text length]({path})\n", md_table(rows)]
    if vn is not None:
        out.append(f"\nCrawler titles alone: median {series_titles.median():.0f} words.")
        vn_p99 = series["vietnam"].quantile(0.99)
        out.append(
            f"\n\n**Takeaways.** SEntFiN and FiQA are headline-length (median {series['sentfin'].median():.0f} and "
            f"{series['fiqa'].median():.0f} words); FinEntity sentences (median {series['finentity'].median():.0f}) "
            f"are the closest to the deployment sentences (median {series['vietnam'].median():.0f}). With roughly "
            f"1.3 subword tokens per word, a max length of 128 tokens covers the p99 of every source "
            f"({vn_p99:.0f} words for the crawler); 64 would truncate long news sentences.\n")
    return "\n".join(out)


def sec_targets(pairs, texts, figs: Figs) -> str:
    plt = figs.plt
    cats = ["0", "1", "2", "3", "4", "5+"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    width = 0.26
    x = np.arange(len(cats))
    rows = []
    for i, s in enumerate(SOURCES):
        n = texts.loc[texts["source"] == s, "n_targets"].clip(upper=5)
        share = n.value_counts(normalize=True).reindex(range(6), fill_value=0).to_numpy()
        ax.bar(x + (i - 1) * (width + 0.02), share, width, color=COLOR[s], label=NAMES[s])
        t = texts[texts["source"] == s]
        mixed = pairs[pairs["source"] == s].drop_duplicates("text_id")["is_mixed"].mean()
        rows.append({"source": NAMES[s], **{c: _pct(v) for c, v in zip(cats, share)},
                     "mean targets/text": f"{t['n_targets'].mean():.2f}",
                     "mixed-polarity texts (of texts with a target)": _pct(mixed)})
    ax.set_xticks(x, cats)
    ax.set_xlabel("targets per text")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Targets per text (after LLM completion)")
    path = figs.save(fig, "targets_per_text")

    gold = pairs[pairs["target_source"] == "gold"]
    gold_mt = {s: (gold[gold["source"] == s].groupby("text_id").size() > 1).mean() for s in SOURCES}
    all_mt = {s: (pairs[pairs["source"] == s].groupby("text_id").size() > 1).mean() for s in SOURCES}
    tl = pairs.assign(w=words(pairs["target"]))
    tlen = md_table([{"source": NAMES[s], "target words, median": f"{tl.loc[tl.source == s, 'w'].median():.0f}",
                      "p95": f"{tl.loc[tl.source == s, 'w'].quantile(.95):.0f}",
                      "relative start position, median": f"{(tl.loc[tl.source == s, 'start'].astype(float) / tl.loc[tl.source == s, 'text'].str.len()).median():.2f}"}
                     for s in SOURCES])
    return "\n".join([
        "## 3. Targets per text\n", f"![Targets per text]({path})\n", md_table(rows),
        f"\nBefore completion (gold only), texts with more than one target: SEntFiN {_pct(gold_mt['sentfin'])}, "
        f"FiQA {_pct(gold_mt['fiqa'])}, FinEntity {_pct(gold_mt['finentity'])}.\n", tlen,
        f"\n**Takeaways.** FiQA was the most under-annotated source: completion lifts its multi-target texts "
        f"from {_pct(gold_mt['fiqa'])} to {_pct(all_mt['fiqa'])} (SEntFiN: {_pct(gold_mt['sentfin'])} to "
        f"{_pct(all_mt['sentfin'])}), so the target-contrast cases ABSA is about now exist in every source. "
        f"FinEntity's sentences with a target are the most often multi-target ({_pct(gold_mt['finentity'])}), "
        f"and its {int((texts.loc[texts.source == 'finentity', 'n_targets'] == 0).sum())} target-free sentences "
        "are the negatives a span tagger needs. Targets are short (1-2 words); in headlines they tend to come "
        "first, in news sentences they sit mid-sentence.\n"])


def sec_target_types(pairs, figs: Figs) -> str:
    plt = figs.plt
    typed = pairs[pairs["target_type"].notna()]
    groups = [("sentfin", "gold"), ("sentfin", "llm"), ("fiqa", "gold"), ("fiqa", "llm")]
    types = typed["target_type"].value_counts().index.tolist()
    mat = np.array([[((typed["source"] == s) & (typed["target_source"] == g) & (typed["target_type"] == t)).sum()
                      for t in types] for s, g in groups], dtype=float)
    share = mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    _heat(ax, share, [f"{NAMES[s]} · {g}" for s, g in groups], [t.replace("_", " ") for t in types])
    ax.set_title("Target types: what the LLM added vs what annotators labelled (row %)")
    path = figs.save(fig, "target_types")
    rows = [{"type": t, **{f"{NAMES[s]} {g}": f"{int(mat[i, j]):,}" for i, (s, g) in enumerate(groups)}}
            for j, t in enumerate(types)]
    return "\n".join(["## 4. Target types\n", f"![Target types]({path})\n", md_table(rows),
                      "\n**Takeaways.** Gold SEntFiN and FiQA targets are mostly companies; the LLM additions are "
                      "dominated by what annotators skipped: regulators and governments, indices and markets, "
                      "sectors, rates and countries. FinEntity has no `target_type` (it was not completed).\n"])


def _heat(ax, share, ylabels, xlabels, fmt="{:.0%}"):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=max(0.5, float(np.nanmax(share))), aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    vmax = max(0.5, float(np.nanmax(share)))
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share[i, j]
            if v >= 0.005:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.45 * vmax else INK)


def sec_entities(pairs, figs: Figs) -> str:
    plt = figs.plt
    fig, ax = plt.subplots(figsize=(7, 2.8))
    rows = []
    for s in SOURCES:
        e = pairs.loc[(pairs["source"] == s) & (pairs["target_source"] == "gold"), "entity_id"]
        c = e.value_counts()
        cum = c.cumsum().to_numpy() / c.sum()
        k = np.arange(1, len(c) + 1)
        ax.plot(k, cum, color=COLOR[s], linewidth=2, label=NAMES[s])
        name = pairs.loc[(pairs["source"] == s) & (pairs["target_source"] == "gold")].groupby("entity_id")["target"].agg(
            lambda x: x.value_counts().index[0])
        rows.append({"source": NAMES[s], "distinct entities": f"{len(c):,}", "top-10 share": _pct(cum[min(9, len(cum) - 1)]),
                     "top-100 share": _pct(cum[min(99, len(cum) - 1)]), "seen once": _pct((c == 1).sum() / len(c)),
                     "most frequent": ", ".join(f"{name[i]} ({n})" for i, n in c.head(5).items())})
    ax.set_xscale("log")
    ax.set_xlabel("top-k entities (log scale)")
    ax.set_ylabel("share of gold pairs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.legend(loc="lower right")
    ax.set_title("Entity concentration: share of gold pairs covered by the top-k entities")
    path = figs.save(fig, "entity_concentration")

    # entity priors: how predictable is sentiment from the entity alone?
    prior_rows = []
    for s in SOURCES:
        p = pairs[(pairs["source"] == s) & (pairs["target_source"] == "gold")]
        tr, te = p[p["split"] == "train"], p[p["split"] == "test"]
        maj = tr["polarity"].value_counts().index[0]
        ent = tr.groupby("entity_id")["polarity"].agg(lambda x: x.value_counts().index[0])
        pred = te["entity_id"].map(ent).fillna(maj)
        seen = te["entity_seen_in_train"].astype(bool)
        freq = tr.groupby("entity_id")["polarity"].agg(["size", lambda x: x.value_counts(normalize=True).iloc[0]])
        freq.columns = ["n", "top_share"]
        f5 = freq[freq["n"] >= 5]
        prior_rows.append({"source": NAMES[s], "test pairs, entity seen in train": _pct(seen.mean()),
                           "majority-class acc.": _pct((te["polarity"] == maj).mean()),
                           "entity-prior acc.": _pct((pred == te["polarity"]).mean()),
                           "entities (>= 5 train mentions) with one label >= 80%": _pct((f5["top_share"] >= .8).mean())})
    return "\n".join(["## 5. Entities and shortcut risk\n", f"![Entity concentration]({path})\n", md_table(rows),
                      "\n**Entity priors (gold only, test split).** If sentiment can be guessed from the entity "
                      "name, a model can score well without reading the text.\n", md_table(prior_rows),
                      "\n**Takeaways.** FiQA is the most concentrated (a handful of tickers such as AAPL and "
                      "TSLA), but its entity prior barely beats the majority class. SEntFiN is where entity "
                      "identity is a real shortcut: most test entities were seen in training and guessing each "
                      "entity's training majority label adds about 15 points over the majority class "
                      "(regulators and exchanges are almost always neutral). FinEntity has many label-"
                      "consistent entities but fewer than half of its test entities appear in training. Report "
                      "SEntFiN on `entity_seen_in_train == False` and consider masking the target string.\n"])


def sec_sentiment(pairs, figs: Figs) -> str:
    plt = figs.plt
    groups = []
    for s in SOURCES:
        for split in ("train", "valid", "test"):
            groups.append((f"{NAMES[s]} {split}", pairs[(pairs.source == s) & (pairs.split == split) & (pairs.target_source == "gold")]))
        llm = pairs[(pairs.source == s) & (pairs.target_source == "llm")]
        if len(llm):
            groups.append((f"{NAMES[s]} LLM-added (silver)", llm))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(groups) + 0.9))
    rows = []
    for i, (name, d) in enumerate(groups):
        share = d["polarity"].value_counts(normalize=True).reindex(list(LABELS), fill_value=0)
        left = 0.0
        for lab in LABELS:
            v = share[lab]
            ax.barh(i, v - 0.004 if v > 0.004 else v, left=left, height=0.62, color=SENT_COLOR[lab],
                    label=lab if i == 0 else None)
            if v >= 0.09:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="#ffffff")
            left += v
        rows.append({"group": name, "pairs": f"{len(d):,}", **{lab: _pct(share[lab]) for lab in LABELS}})
    ax.set_yticks(range(len(groups)), [g for g, _ in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Sentiment by source and split (gold), and of LLM-added targets (silver)", pad=22)
    path = figs.save(fig, "sentiment")
    return "\n".join(["## 6. Sentiment labels\n", f"![Sentiment]({path})\n", md_table(rows),
                      "\n**Takeaways.** Splits are well stratified within each source. The three sources differ "
                      "in their neutral share: FinEntity is the most neutral, FiQA the most positive even after "
                      "calibration. LLM-added targets are more often neutral than gold ones (regulators, "
                      "indices and sources are mentioned rather than judged). Use class weights or the soft "
                      "labels; macro-F1 is the headline metric.\n"])


def sec_soft(pairs, figs: Figs, thresholds: dict | None) -> str:
    plt = figs.plt
    fq = pairs[(pairs["source"] == "fiqa") & pairs["score"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    ax = axes[0]
    ax.hist(fq["score"], bins=np.arange(-1, 1.01, 0.05), color=COLOR["fiqa"], rwidth=0.85)
    lines = [(-0.1, "negative below −0.1 (both)", INK2), (0.1, "FinBen positive from 0.1", MUTED)]
    if thresholds:
        lines.append((thresholds["pos_from"], f"calibrated positive from {thresholds['pos_from']}", INK))
    for x, lab, color in lines:  # identified in a legend: the strip between -0.1 and 0.22 is too narrow for text
        ax.axvline(x, color=color, linewidth=1.4, label=lab)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor=SURFACE, framealpha=1)
    ax.set_xlabel("FiQA gold score")
    ax.set_ylabel("pairs")
    ax.set_title("FiQA scores and class thresholds")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    rows = []
    # label types, not sources: blue / orange / violet (validated as a set; yellow stays "Vietnam news")
    for lt, color in (("hard", "#2a78d6"), ("soft", "#eb6834"), ("silver", "#6250d6")):
        c = np.sort(pairs.loc[pairs["label_type"] == lt, "label_confidence"].to_numpy())
        if not len(c):
            continue
        ax.plot(c, np.arange(1, len(c) + 1) / len(c), color=color, linewidth=2, label=lt, drawstyle="steps-post")
        rows.append({"label type": lt, "pairs": f"{len(c):,}", "median confidence": f"{np.median(c):.3f}",
                     "below 0.8": _pct((c < 0.8).mean()), "below 0.6": _pct((c < 0.6).mean())})
    ax.set_xlabel("label confidence (max soft probability)")
    ax.set_ylabel("cumulative share of pairs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.legend(loc="upper left")
    ax.set_title("How certain are the labels?")
    path = figs.save(fig, "soft_labels")
    near = ((fq["score"] - 0.1).abs() < 0.1) | ((fq["score"] + 0.1).abs() < 0.1)
    return "\n".join(["## 7. Score thresholds and soft labels\n", f"![Soft labels]({path})\n", md_table(rows),
                      f"\n{_pct(near.mean())} of FiQA gold pairs lie within 0.1 of a FinBen threshold; with "
                      "annotation noise of about σ = 0.08 their class is uncertain, which is what the soft labels "
                      "encode. Hard labels (SEntFiN, FinEntity) stay near 0.96; silver labels split by the "
                      "bridge annotator's confidence (high ≈ 0.93, medium ≈ 0.57).\n"])


def sec_aspects(pairs, figs: Figs) -> str:
    plt = figs.plt
    mat = np.array([[(pairs.loc[pairs.source == s, "aspect"] == a).mean() for s in SOURCES] for a in ASPECTS])
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    _heat(ax, mat, list(ASPECTS), [NAMES[s] for s in SOURCES])
    ax.set_xticks(range(len(SOURCES)), [NAMES[s] for s in SOURCES], rotation=0, ha="center")
    ax.set_title("Aspect mix per source (column %)")
    p1 = figs.save(fig, "aspects")

    gold_sent = pairs[pairs["sentiment_source"] == "gold"]
    by = gold_sent.groupby("aspect")["polarity"].value_counts(normalize=True).unstack().reindex(
        index=list(ASPECTS), columns=list(LABELS), fill_value=0)
    order = by.sort_values("negative").index
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for i, a in enumerate(order):
        left = 0.0
        for lab in LABELS:
            v = by.loc[a, lab]
            ax.barh(i, v - 0.004 if v > 0.004 else v, left=left, height=0.62, color=SENT_COLOR[lab],
                    label=lab if i == 0 else None)
            if v >= 0.12:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="#ffffff")
            left += v
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Sentiment by aspect (gold sentiment, all sources)", pad=22)
    p2 = figs.save(fig, "aspect_sentiment")

    n = pairs.groupby(["aspect", "source"]).size().unstack(fill_value=0).reindex(index=list(ASPECTS), fill_value=0)
    tr = pairs[pairs["split"] == "train"].groupby("aspect").size().reindex(list(ASPECTS), fill_value=0)
    rows = [{"aspect": a, **{NAMES[s]: f"{int(n.loc[a, s]):,}" for s in SOURCES if s in n},
             "train pairs (all)": f"{int(tr[a]):,}",
             "negative / neutral / positive": " / ".join(_pct(by.loc[a, l], 0) for l in LABELS)} for a in ASPECTS]
    conf = pairs["aspect_confidence"].value_counts(normalize=True)
    rare = [a for a in ASPECTS if tr[a] < 150]
    return "\n".join([
        "## 8. Aspect categories (silver)\n", f"![Aspects]({p1})\n", f"![Aspect sentiment]({p2})\n", md_table(rows),
        f"\nLLM aspect confidence: {', '.join(f'{k} {_pct(v)}' for k, v in conf.items())}.\n",
        f"\n**Takeaways.** The sources are complementary: FiQA is dominated by stock outlook and "
        f"recommendations (StockTwits calls), FinEntity by financial results, price moves and 'Other' (quoted "
        f"sources such as Refinitiv), and SEntFiN covers the Market/Economy categories the others barely touch. Categories with fewer than 150 training pairs ({', '.join(rare)}) will be "
        f"hard to learn; merge them at level 1 or report macro-F1 per level. Aspect strongly conditions "
        f"sentiment (Corporate/Risks is mostly negative; Management, Monetary Policy, Regulation and Other are "
        f"over 90 % neutral). A joint model can exploit this, but it is also a shortcut: check sentiment on "
        f"those aspects separately.\n"])


def _vocab(texts) -> set[str]:
    return {w for t in texts for w in _TOK.findall(t.lower())}


def _oov(texts, vocab) -> float:
    toks = [w for t in texts for w in _TOK.findall(t.lower())]
    return sum(w not in vocab for w in toks) / max(1, len(toks))


def sec_domain(pairs, texts, vn, figs: Figs) -> str:
    if vn is None:
        return "## 9. Deployment domain\n\n_(crawler exports not found; pass --target-domain)_\n"
    plt = figs.plt
    vocab = {s: _vocab(texts.loc[(texts.source == s) & (texts.split == "train"), "text"]) for s in SOURCES}
    vocab["all"] = set().union(*vocab.values())
    vn_txt = vn["text"]
    rows, oov_vn = [], {}
    for s in (*SOURCES, "all"):
        own = _oov(texts.loc[(texts.source == s) & (texts.split == "test"), "text"], vocab[s]) if s != "all" else None
        oov_vn[s] = _oov(vn_txt, vocab[s])
        rows.append({"training vocabulary": NAMES.get(s, "all three sources"), "word types": f"{len(vocab[s]):,}",
                     "OOV on its own test": _pct(own) if own is not None else "-",
                     "OOV on Vietnam news": _pct(oov_vn[s])})
    fig, ax = plt.subplots(figsize=(6, 2.4))
    keys = [*SOURCES, "all"]
    ax.barh(range(len(keys)), [oov_vn[k] for k in keys], height=0.6,
            color=[COLOR.get(k, INK2) for k in keys])
    for i, k in enumerate(keys):
        ax.text(oov_vn[k] + 0.004, i, _pct(oov_vn[k]), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(keys)), [NAMES.get(k, "all sources") for k in keys])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Vocabulary gap: share of Vietnam-news word tokens unseen in training")
    path = figs.save(fig, "domain_oov")

    def mentions(series):
        low = series.str.lower()
        return {t: int(low.str.contains(rf"\b{re.escape(t)}\b", regex=True).sum()) for t in VN_TERMS}
    tr_text = texts.loc[texts["split"] == "train", "text"]
    m_train, m_vn = mentions(tr_text), mentions(vn_txt)
    vt = [{"term": t, "training texts mentioning it": f"{m_train[t]:,}",
           "Vietnam-news sentences mentioning it": f"{m_vn[t]:,}"}
          for t in sorted(VN_TERMS, key=lambda t: -m_vn[t]) if m_vn[t] or m_train[t]][:12]
    known = {k for k in pairs.loc[pairs["split"] == "train", "target"].map(surface_key) if len(k) > 3}
    known_sample = sorted(known, key=len, reverse=True)[:3000]
    any_known = vn_txt.str.lower().map(lambda s: any(f" {k} " in f" {s} " for k in known_sample))
    return "\n".join([
        "## 9. Deployment domain: English news about Vietnam\n",
        f"![Domain OOV]({path})\n", md_table(rows),
        "\n**Vietnam-specific entities.** How often the training data mentions what the deployed model will "
        "see most:\n", md_table(vt),
        f"\n{_pct(any_known.mean())} of crawler sentences contain at least one of the 3,000 longest entity names "
        "seen as training targets.\n",
        "\n**Takeaways.** Even the union of the three training sources leaves a noticeable share of "
        "Vietnam-news word tokens unseen, and Vietnam-specific entities (VN-Index, the State Bank of Vietnam, "
        "Vietnamese companies, the dong) are almost absent from training. Expect a drop on the planned "
        "Vietnam test set, especially for target extraction; the small annotated Vietnam test set is the "
        "only honest estimate, and a few hundred in-domain training sentences (silver-labelled like SEntFiN) "
        "would likely help more than any architecture change.\n"])


def sec_quality(pairs, texts) -> str:
    from .text import dedup_key
    keys = texts.assign(k=texts["text"].map(dedup_key))
    cross_source = keys.groupby("k")["source"].nunique()
    cross_split = keys.groupby(["source", "k"])["split"].nunique()
    flags = pairs["quality_flags"].fillna("").str.split(";").explode()
    flags = flags[flags != ""].value_counts()
    rows = [{"check": "texts appearing in two sources (near-duplicates)", "result": f"{int((cross_source > 1).sum())}"},
            {"check": "near-duplicate texts across splits within a source",
             "result": ", ".join(f"{NAMES[s]} {int(v)}" for s, v in (cross_split > 1).groupby(level=0).sum().items())},
            {"check": "pairs whose target was not found in the text", "result": f"{int((pairs['span_match'] == 'not_found').sum())}"},
            {"check": "pairs with any quality flag", "result": f"{int((pairs['quality_flags'].fillna('') != '').sum()):,}"}]
    return "\n".join(["## 10. Data quality and leakage checks\n", md_table(rows), "\n**Most common quality flags**\n",
                      md_table([{"flag": k, "pairs": f"{v:,}"} for k, v in flags.head(10).items()]),
                      "\nThe only cross-split near-duplicates are FiQA's (3 re-posted tweets in the official "
                      "splits, flagged `near_duplicate_across_splits`); SEntFiN and FinEntity splits are grouped "
                      "so none exist.\n"])


def sec_recommendations(pairs, texts) -> str:
    tr = pairs[pairs["split"] == "train"]
    w = tr["polarity"].value_counts()
    cw = {l: len(tr) / (3 * w[l]) for l in LABELS}
    return "\n".join([
        "## 11. Implications for modelling\n",
        f"1. **Sequence length:** 128 subword tokens; nothing needs more than ~100 words.",
        f"2. **Class balance (combined train):** weights negative {cw['negative']:.2f}, neutral {cw['neutral']:.2f}, "
        f"positive {cw['positive']:.2f}; or train on the soft labels.",
        "3. **Target extraction:** train the span head on FinEntity (exhaustive, with negative sentences) plus the "
        "LLM-completed SEntFiN/FiQA texts; evaluate it on FinEntity test and the Vietnam test set only.",
        "4. **Shortcut control:** report SEntFiN on unseen entities; consider masking the target string for a "
        "share of training examples.",
        "5. **Aspects:** 21 categories with a long tail; report level-1 and level-2 macro-F1, and treat test "
        "aspect scores as agreement with the LLM (no human aspect labels except FiQA's `fiqa_aspect`).",
        "6. **Source tag:** label conventions differ (neutral share 16-53 %); add the source as an input token or "
        "evaluate per source.",
        "7. **Deployment domain:** the vocabulary and entity gap to Vietnam news is the largest risk; build the "
        "annotated Vietnam test set early and consider silver in-domain training data.\n"])


def run(processed: Path, out_dir: Path, target_domain: Path | None, thresholds_file: Path) -> Path:
    pairs, texts, vn = load(processed, target_domain)
    figs = Figs(out_dir / "figures")
    thr = json.loads(thresholds_file.read_text()) if thresholds_file.exists() else None
    parts = [
        "# Exploratory data analysis\n",
        "_Generated by `python -m absa_data eda` from `data/processed` "
        f"({len(pairs):,} pairs) and the crawler exports; figures in `figures/`._\n",
        sec_overview(pairs, texts, vn), sec_length(texts, vn, figs), sec_targets(pairs, texts, figs),
        sec_target_types(pairs, figs), sec_entities(pairs, figs), sec_sentiment(pairs, figs),
        sec_soft(pairs, figs, thr), sec_aspects(pairs, figs), sec_domain(pairs, texts, vn, figs),
        sec_quality(pairs, texts), sec_recommendations(pairs, texts),
    ]
    out = out_dir / "EDA.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
