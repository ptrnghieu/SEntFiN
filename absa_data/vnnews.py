"""Vietnam news (in-domain target data): LLM triplet pre-labels and a human review set.

Input: ``data/sentences_clean.json`` of minhhai205/aspect-based-FSA (crawled English-language
Vietnamese news, one record per sentence: sent_id, doc_id, sent_idx, text, lang, is_title).
The text is kept exactly as given: the crawler computes target offsets on its own normalised
text, so normalising again would shift spans.

Steps (each cached in ``annotations/vnnews/``; re-running only labels what is new):
  1. targets    the target-completion prompt with no given targets (= full extraction)
  2. sentiment  the bridge sentiment prompt (calibration.SENTIMENT_TASK)
  3. aspect     the 21-category aspect prompt (annotate.ASPECT_TASK)

Split: whole articles (plus near-duplicate sentences across articles) go to one split.
About ``n_test`` sentences become the test split, which is the human review set; the
rest is 90/10 train/valid. The split is frozen in ``splits/vnnews_split_v1.csv``.

Review: pre-annotation, one workbook per reviewer, with an overlap block every reviewer
labels (for inter-annotator agreement). Reviewers correct the LLM's triplets in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annotate import ASPECT_TASK, Annotator, latest_labels
from .calibration import DEFAULT_BRIDGE_FLIP, SENTIMENT_TASK, soft_from_hard
from .completion import TARGET_TYPES, TargetCompleter
from .completion import Item as TextItem
from .entities import entity_id
from .schema import PAIR_COLUMNS, TEXT_COLUMNS
from .splits import connected_groups
from .taxonomy import LABELS, guidelines_markdown
from .text import align_targets, dedup_key

SOURCE = "vnnews"
MODEL = "gpt-6-luna"
SENTIMENTS = ("positive", "neutral", "negative")


def load_sentences(path: str | Path) -> pd.DataFrame:
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    t = pd.DataFrame(recs)
    t = t[t["lang"] == "en"].drop_duplicates("sent_id").reset_index(drop=True)
    t["publisher"] = t["doc_id"].str.split("_").str[0]
    t["day"] = pd.to_datetime(t["doc_id"].str.extract(r"_(\d{8})_")[0], format="%Y%m%d").dt.date.astype(str)
    return pd.DataFrame({
        "text_id": t["sent_id"], "source": SOURCE, "text_type": np.where(t["is_title"], "headline", "sentence"),
        "text": t["text"], "paragraph_id": t["doc_id"], "sent_idx": t["sent_idx"],
        "publisher": t["publisher"], "day": t["day"]})


def ann_paths(annotations_dir: Path) -> dict[str, Path]:
    d = annotations_dir / SOURCE
    return {"targets": d / f"targets_{MODEL}.jsonl", "sentiment": d / f"sentiment_{MODEL}.jsonl",
            "aspect": d / f"aspect_{MODEL}.jsonl", "hashes": d / "text_hashes.json"}


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def invalidate_changed(texts: pd.DataFrame, annotations_dir: Path) -> list[str]:
    """The crawler may fix a sentence after it was labelled. Labels are cached by sent_id, so drop
    every cached record of a sentence whose text changed since it was labelled; `label` redoes them."""
    paths = ann_paths(annotations_dir)
    old = json.loads(paths["hashes"].read_text()) if paths["hashes"].exists() else {}
    changed = [t for t, x in zip(texts["text_id"], texts["text"]) if t in old and old[t] != _sha(x)]
    if changed:
        bad = set(changed)
        for key in ("targets", "sentiment", "aspect"):
            f = paths[key]
            if f.exists():
                keep = [l for l in f.read_text(encoding="utf-8").splitlines()
                        if l.strip() and json.loads(l).get("text_id") not in bad]
                f.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    return changed


def save_hashes(texts: pd.DataFrame, annotations_dir: Path) -> None:
    f = ann_paths(annotations_dir)["hashes"]
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update({t: _sha(x) for t, x in zip(texts["text_id"], texts["text"])})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(sorted(old.items())), indent=0) + "\n")


# ----------------------------------------------------------------------------- labelling

def target_pairs(texts: pd.DataFrame, targets_path: Path) -> pd.DataFrame:
    recs = {}
    for line in targets_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["text_id"]] = r  # last record wins
    rows = []
    for t in texts.itertuples(index=False):
        added = recs.get(t.text_id, {}).get("added", [])
        spans = align_targets(t.text, [a["target"] for a in added], fuzzy=False)
        for i, (a, sp) in enumerate(zip(added, spans)):
            rows.append({"uid": f"{t.text_id}@{i}", "text_id": t.text_id, "text": t.text, "text_type": t.text_type,
                         "target": a["target"], "target_type": a["type"], "target_confidence": a["confidence"],
                         "start": sp.start, "end": sp.end, "span_match": sp.match})
    return pd.DataFrame(rows)


def label(texts: pd.DataFrame, annotations_dir: Path, client, workers: int = 6, log=print) -> dict:
    paths = ann_paths(annotations_dir)
    changed = invalidate_changed(texts, annotations_dir)
    if changed:
        log(f"  {len(changed)} sentences changed since labelling; relabelling them: {changed[:5]}")
    items = [TextItem(r.text_id, r.text, r.text_type, []) for r in texts.itertuples(index=False)]
    stats = {"targets": TargetCompleter(client, model=MODEL).run(items, paths["targets"], workers=workers, log=log)}
    pairs = target_pairs(texts, paths["targets"])
    for task in (SENTIMENT_TASK, ASPECT_TASK):
        stats[task.name] = Annotator(client, model=MODEL, task=task).run(
            pairs, paths[task.name], workers=workers, log=log)
    save_hashes(texts, annotations_dir)
    return stats


# ----------------------------------------------------------------------------- tables + split

def split_texts(texts: pd.DataFrame, manifest: Path, n_test: int = 300, valid_frac: float = 0.1,
                seed: int = 42) -> pd.Series:
    """Frozen article-level split; new sentences of a known article follow it, new articles go to train."""
    groups = connected_groups(texts.assign(dk=texts["text"].map(dedup_key)), ["paragraph_id", "dk"])
    if manifest.exists():
        m = pd.read_csv(manifest)
        by_text = dict(zip(m["text_id"], m["split"]))
        by_group = dict(zip(m["group_id"], m["split"]))
        return pd.Series([by_text.get(t, by_group.get(g, "train")) for t, g in zip(texts["text_id"], groups)],
                         index=texts.index, name="split")
    rng = np.random.default_rng(seed)
    sizes = groups.value_counts()
    order = list(rng.permutation(sorted(sizes.index)))
    split, n = {}, 0
    for g in order:  # test: whole articles until ~n_test sentences
        if n >= n_test:
            break
        split[g], n = "test", n + sizes[g]
    rest = [g for g in order if g not in split]
    n_valid, n = valid_frac * sizes[rest].sum(), 0
    for g in rest:
        if n < n_valid:
            split[g], n = "valid", n + sizes[g]
        else:
            split[g] = "train"
    out = groups.map(split).rename("split")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text_id": texts["text_id"], "group_id": groups, "split": out}).to_csv(manifest, index=False)
    return out


def build_tables(texts: pd.DataFrame, annotations_dir: Path, manifest: Path, n_test: int = 300,
                 aliases: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ann_paths(annotations_dir)
    texts = texts.copy()
    texts["group_id"] = connected_groups(texts.assign(dk=texts["text"].map(dedup_key)), ["paragraph_id", "dk"])
    texts["split"] = split_texts(texts, manifest, n_test=n_test)
    pairs = target_pairs(texts, paths["targets"])
    sent = latest_labels(paths["sentiment"], SENTIMENT_TASK.version)
    asp = latest_labels(paths["aspect"], ASPECT_TASK.version)
    pairs = pairs.merge(sent[["uid", "sentiment", "confidence"]].rename(
        columns={"sentiment": "polarity", "confidence": "sentiment_confidence"}), on="uid", how="left")
    pairs = pairs.merge(asp[["uid", "aspect", "confidence"]].rename(columns={"confidence": "aspect_confidence"}),
                        on="uid", how="left")
    meta = texts.set_index("text_id")
    pairs["source"] = SOURCE
    pairs["split"] = pairs["text_id"].map(meta["split"])
    pairs["group_id"] = pairs["text_id"].map(meta["group_id"])
    pairs["paragraph_id"] = pairs["text_id"].map(meta["paragraph_id"])
    pairs["text_raw"], pairs["target_raw"] = pairs["text"], pairs["target"]
    pairs["target_annotation"] = "llm_completed"
    pairs["target_source"] = pairs["sentiment_source"] = pairs["aspect_source"] = "llm"
    pairs["label"] = pairs["polarity"].map({"negative": 0, "neutral": 1, "positive": 2})
    pairs["aspect_l1"] = pairs["aspect"].str.split("/").str[0]
    pairs["entity_id"] = pairs["target"].map(lambda t: entity_id(t, aliases))
    pairs["n_targets"] = pairs.groupby("text_id")["uid"].transform("size")
    pairs["is_mixed"] = pairs.groupby("text_id")["polarity"].transform(
        lambda s: s.isin(["positive"]).any() and s.isin(["negative"]).any())
    seen = set(pairs.loc[pairs["split"] == "train", "entity_id"])
    pairs["entity_seen_in_train"] = pairs["entity_id"].isin(seen)
    probs = np.array([soft_from_hard(p, DEFAULT_BRIDGE_FLIP.get(c, DEFAULT_BRIDGE_FLIP["medium"]))
                      if isinstance(p, str) else (np.nan,) * 3
                      for p, c in zip(pairs["polarity"], pairs["sentiment_confidence"])], dtype=float).reshape(-1, 3)
    pairs["label_type"] = "silver"
    pairs["p_negative"], pairs["p_neutral"], pairs["p_positive"] = probs.T.round(4)
    pairs["label_confidence"] = probs.max(axis=1).round(4)
    flags = np.where(pairs["polarity"].isna(), "no_sentiment_label", "")
    flags = np.where(pairs["aspect"].isna(), np.char.add(flags.astype(str), ";no_aspect_label"), flags)
    pairs["quality_flags"] = pd.Series(flags, index=pairs.index).str.strip(";")
    extra = ["target_type", "target_confidence", "sentiment_confidence", "aspect_confidence"]
    cols = list(dict.fromkeys([c for c in PAIR_COLUMNS] + [c for c in extra if c not in PAIR_COLUMNS]))
    pairs = pairs.reindex(columns=cols)

    texts["target_annotation"] = "llm_completed"
    texts["n_targets"] = texts["text_id"].map(pairs.groupby("text_id").size()).fillna(0).astype(int)
    texts = texts.reindex(columns=list(dict.fromkeys(TEXT_COLUMNS + ["publisher", "day"])))
    return pairs.reset_index(drop=True), texts.reset_index(drop=True)


def profile(pairs: pd.DataFrame, texts: pd.DataFrame, stats: dict | None = None) -> str:
    lines = ["# Vietnam news: LLM triplet pre-labels\n",
             f"Source: `data/sentences_clean.json` of minhhai205/aspect-based-FSA. Labels: {MODEL} "
             f"(targets: target-completion prompt; sentiment: `{SENTIMENT_TASK.version}`; "
             f"aspect: `{ASPECT_TASK.version}`). **All labels are silver until reviewed.**\n",
             "| split | sentences | articles | targets | sentences without target |", "|---|---|---|---|---|"]
    for s in ("train", "valid", "test"):
        t = texts[texts["split"] == s]
        lines.append(f"| {s} | {len(t)} | {t['paragraph_id'].nunique()} | {(pairs['split'] == s).sum()} | "
                     f"{(t['n_targets'] == 0).sum()} |")
    lines.append(f"| all | {len(texts)} | {texts['paragraph_id'].nunique()} | {len(pairs)} | "
                 f"{(texts['n_targets'] == 0).sum()} |\n")

    def dist(col, title):
        vc = pairs[col].value_counts(dropna=False)
        out = [f"## {title}\n", "| value | pairs | share |", "|---|---|---|"]
        out += [f"| {k} | {v} | {v / len(pairs):.1%} |" for k, v in vc.items()]
        return out + [""]

    lines += dist("polarity", "Sentiment") + dist("sentiment_confidence", "Sentiment confidence")
    lines += dist("target_type", "Target type") + dist("aspect_l1", "Aspect, level 1") + dist("aspect", "Aspect")
    lines += ["## Most frequent targets\n", "| target | pairs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in pairs["target"].value_counts().head(25).items()] + [""]
    if stats:
        lines += ["## LLM usage (this run)\n", "```json", json.dumps(
            {k: {x: v.get(x) for x in ("input_tokens", "output_tokens", "seconds", "invalid_outputs") if x in v}
             for k, v in stats.items()}, indent=1), "```", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- review workbooks

REVIEW_COLUMNS = ["sent_id", "text", "target", "target_type", "sentiment", "aspect",
                  "keep", "sentiment_fix", "aspect_fix", "note"]
MISSING_COLUMNS = ["sent_id", "text", "target", "sentiment", "aspect", "note"]


def review_assignment(texts: pd.DataFrame, n_reviewers: int = 3, n_overlap: int = 30, seed: int = 7
                      ) -> dict[str, list[str]]:
    """Test sentences per reviewer: a shared overlap block (whole articles) + an equal share of the rest."""
    test = texts[texts["split"] == "test"]
    docs = list(np.random.default_rng(seed).permutation(sorted(test["paragraph_id"].unique())))
    sizes = test.groupby("paragraph_id").size()
    overlap, n = [], 0
    for d in docs:
        if n >= n_overlap:
            break
        overlap.append(d)
        n += sizes[d]
    rest = [d for d in docs if d not in overlap]
    share: dict[int, list[str]] = {i: [] for i in range(n_reviewers)}
    load = [0] * n_reviewers
    for d in sorted(rest, key=lambda d: -sizes[d]):  # balance sentence counts
        i = int(np.argmin(load))
        share[i].append(d)
        load[i] += sizes[d]
    order = test.sort_values(["paragraph_id", "sent_idx"])
    out = {}
    for i in range(n_reviewers):
        keep = set(overlap) | set(share[i])
        out[f"R{i + 1}"] = order.loc[order["paragraph_id"].isin(keep), "text_id"].tolist()
    out["overlap"] = order.loc[order["paragraph_id"].isin(overlap), "text_id"].tolist()
    return out


def write_review_workbook(pairs: pd.DataFrame, texts: pd.DataFrame, text_ids: list[str], overlap: set[str],
                          path: Path, reviewer: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(REVIEW_COLUMNS)
    tx = texts.set_index("text_id")
    by_text = {k: g for k, g in pairs.groupby("text_id")}
    grey, yellow, blue = PatternFill("solid", fgColor="F2F2F2"), PatternFill("solid", fgColor="FFF2CC"), \
        PatternFill("solid", fgColor="DDEBF7")
    shade = False
    for tid in text_ids:
        g = by_text.get(tid)
        rows = ([[tid, tx.at[tid, "text"], None, None, None, None, None, None, None, "(no target found)"]]
                if g is None else
                [[tid, tx.at[tid, "text"], r.target, r.target_type, r.polarity, r.aspect, "y", None, None, None]
                 for r in g.sort_values("start", na_position="last").itertuples()])
        for row in rows:
            ws.append(row)
            if shade:
                for c in ws[ws.max_row][:6]:
                    c.fill = grey
            if tid in overlap:
                ws.cell(row=ws.max_row, column=1).fill = blue
        shade = not shade
    n = ws.max_row
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col, w in zip("ABCDEFGHIJ", (30, 70, 26, 18, 10, 34, 6, 13, 34, 28)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, max_row=n):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[6:9]:
            c.fill = yellow
    ws.freeze_panes = "C2"

    lab = wb.create_sheet("labels")
    for i, a in enumerate(LABELS, 1):
        lab.cell(row=i, column=1, value=a)
    for i, s in enumerate(SENTIMENTS, 1):
        lab.cell(row=i, column=2, value=s)
    for i, t in enumerate(TARGET_TYPES, 1):
        lab.cell(row=i, column=3, value=t)

    def dv(formula, cells, title):
        v = DataValidation(type="list", formula1=formula, allow_blank=True, showErrorMessage=True,
                           errorTitle=title, error="Pick a value from the list.")
        ws.add_data_validation(v)
        v.add(cells)

    dv('"y,n"', f"G2:G{n}", "keep")
    dv(f"=labels!$B$1:$B${len(SENTIMENTS)}", f"H2:H{n}", "sentiment")
    dv(f"=labels!$A$1:$A${len(LABELS)}", f"I2:I{n}", "aspect")

    miss = wb.create_sheet("missing_targets")
    miss.append(MISSING_COLUMNS)
    for cell in miss[1]:
        cell.font = Font(bold=True)
    for col, w in zip("ABCDEF", (30, 60, 26, 12, 34, 28)):
        miss.column_dimensions[col].width = w
    m = DataValidation(type="list", formula1=f"=labels!$B$1:$B${len(SENTIMENTS)}", allow_blank=True)
    miss.add_data_validation(m)
    m.add("D2:D400")
    a = DataValidation(type="list", formula1=f"=labels!$A$1:$A${len(LABELS)}", allow_blank=True)
    miss.add_data_validation(a)
    a.add("E2:E400")

    gd = wb.create_sheet("guidelines")
    gd.column_dimensions["A"].width = 140
    intro = [
        f"Reviewer {reviewer}. Each row is one (target, sentiment, aspect) triplet proposed by the LLM; "
        "rows of the same sentence are adjacent (alternating grey). Blue sent_id = overlap block that every "
        "reviewer labels (used to measure agreement): label it independently, do not discuss it first.",
        "keep: 'y' (default) if the target is a real, specific entity the sentence expresses something about; "
        "'n' if it is not a target (generic word, an event name, a person, wrong span).",
        "sentiment_fix / aspect_fix: fill ONLY when the LLM label is wrong; leave empty to accept it.",
        "Sentiment = what the sentence implies for THAT target (financial/business impact), not the tone of the "
        "whole sentence. Purely factual/procedural statements are neutral.",
        "Missing targets: add a row in the 'missing_targets' sheet (sent_id, the exact target text as it appears "
        "in the sentence, sentiment, aspect).",
        "Target types considered in scope: company, index_market, sector, commodity, currency, bonds_rates, "
        "fund_investor_group, regulator_government, country_economy (people and events are not targets).",
        "",
    ]
    for line in intro + guidelines_markdown().splitlines():
        gd.append([line])
        gd.cell(row=gd.max_row, column=1).alignment = Alignment(wrap_text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_review_set(pairs: pd.DataFrame, texts: pd.DataFrame, out_dir: Path, n_reviewers: int = 3,
                     n_overlap: int = 30) -> dict:
    asg = review_assignment(texts, n_reviewers, n_overlap)
    overlap = set(asg.pop("overlap"))
    out = {}
    for name, ids in asg.items():
        p = out_dir / f"vnnews_test_review_v1_{name}.xlsx"
        write_review_workbook(pairs, texts, ids, overlap, p, name)
        out[name] = {"file": str(p), "sentences": len(ids),
                     "pairs": int(pairs["text_id"].isin(ids).sum())}
    out["overlap_sentences"] = len(overlap)
    return out


# ----------------------------------------------------------------------------- export for the crawler repo

def export_labeled(input_path: str | Path, pairs: pd.DataFrame, texts: pd.DataFrame, out_path: Path) -> dict:
    """sentences_clean.json with two added fields per sentence, in the input's order:
    ``split`` and ``triplets`` (character offsets index ``text`` exactly as given)."""
    recs = json.loads(Path(input_path).read_text(encoding="utf-8"))
    split = dict(zip(texts["text_id"], texts["split"]))
    by_text: dict[str, list[dict]] = {}
    for r in pairs.sort_values(["text_id", "start"]).itertuples(index=False):
        by_text.setdefault(r.text_id, []).append({
            "target": r.target, "start": int(r.start), "end": int(r.end), "target_type": r.target_type,
            "sentiment": r.polarity, "sentiment_confidence": r.sentiment_confidence,
            "aspect": r.aspect, "aspect_l1": r.aspect_l1, "aspect_confidence": r.aspect_confidence,
            "source": "llm"})
    out = [{**x, "split": split.get(x["sent_id"]), "triplets": by_text.get(x["sent_id"], [])}
           for x in recs if x.get("lang") == "en"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"sentences": len(out), "triplets": sum(len(x["triplets"]) for x in out), "file": str(out_path)}
