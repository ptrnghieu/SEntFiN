"""Human verification of the silver aspect labels.

1. ``python -m absa_data review-sheet`` samples headlines from the SEntFiN
   *test* split and writes a blind spreadsheet (the LLM's label is not shown,
   to avoid anchoring) with a dropdown of the taxonomy and the guidelines.
2. You fill the ``aspect`` column.
3. ``python -m absa_data review-eval --sheet <file>`` measures agreement with
   the LLM and stores your labels as gold (``annotations/human/``); the build
   then uses them instead of the LLM's for those pairs, so the SEntFiN test
   split gets a human-verified aspect subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from .taxonomy import LABELS, TAXONOMY_VERSION, guidelines_markdown, level1

REVIEW_COLUMNS = ["uid", "text_id", "text", "target", "sentiment", "aspect", "unsure", "note"]


def _num(text_id: str) -> int:
    return int(text_id.rsplit("-", 1)[1])


def sample_review_pairs(pairs: pd.DataFrame, n_texts: int = 400, seed: int = 13) -> pd.DataFrame:
    """Uniform random sample of SEntFiN test headlines (all their targets)."""
    test = pairs[(pairs["source"] == "sentfin") & (pairs["split"] == "test")]
    ids = sorted(test["text_id"].unique(), key=_num)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(ids, size=min(n_texts, len(ids)), replace=False))
    out = test[test["text_id"].isin(chosen)].copy()
    out["_n"] = out["text_id"].map(_num)
    return out.sort_values(["_n", "uid"]).drop(columns="_n")


def write_review_sheet(sample: pd.DataFrame, path: str | Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(REVIEW_COLUMNS)
    for r in sample.itertuples(index=False):
        ws.append([r.uid, r.text_id, r.text, r.target, r.polarity, None, None, None])
    n = len(sample) + 1
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col, width in zip("ABCDEFGH", (14, 13, 70, 24, 10, 38, 8, 30)):
        ws.column_dimensions[col].width = width
    fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2, max_row=n):
        row[2].alignment = Alignment(wrap_text=True, vertical="top")
        row[5].fill = fill
    ws.freeze_panes = "C2"

    labels = wb.create_sheet("labels")
    for label in LABELS:
        labels.append([label])
    dv = DataValidation(type="list", formula1=f"=labels!$A$1:$A${len(LABELS)}", allow_blank=True,
                        showErrorMessage=True, errorTitle="Unknown aspect",
                        error="Pick a category from the list (see the guidelines sheet).")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{n}")
    unsure = DataValidation(type="list", formula1='"y"', allow_blank=True)
    ws.add_data_validation(unsure)
    unsure.add(f"G2:G{n}")

    g = wb.create_sheet("guidelines")
    g.column_dimensions["A"].width = 140
    g.append(["How to label: pick one category per row in the yellow 'aspect' column of the 'review' sheet. "
              "Rows of the same headline are adjacent. Mark 'unsure' with y when two labels are plausible."])
    for line in guidelines_markdown().splitlines():
        g.append([line])
        g.cell(row=g.max_row, column=1).alignment = Alignment(wrap_text=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_review(path: str | Path) -> pd.DataFrame:
    """Filled review rows (aspect non-empty). Raises on labels outside the taxonomy."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path, sheet_name="review")
    df = df.rename(columns=str.strip)
    df["aspect"] = df["aspect"].astype("string").str.strip()
    df = df[df["aspect"].notna() & (df["aspect"] != "")]
    bad = df[~df["aspect"].isin(LABELS)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have labels outside taxonomy {TAXONOMY_VERSION}: "
                         f"{bad[['uid', 'aspect']].head().to_dict('records')}")
    df["unsure"] = df["unsure"].astype("string").str.strip().str.lower().eq("y").fillna(False)
    return df[["uid", "target", "aspect", "unsure", "note"]].reset_index(drop=True)


def agreement(reference: pd.Series, predicted: pd.Series, confidence: pd.Series | None = None,
              ref_name: str = "reference", pred_name: str = "LLM") -> str:
    """Markdown agreement report; `reference` is treated as ground truth."""
    from .profile import md_table

    y, yhat = reference.reset_index(drop=True), predicted.reset_index(drop=True)
    n = len(y)
    if n == 0:
        return "_(no overlapping pairs)_\n"
    acc = (y == yhat).mean()
    acc1 = (y.map(level1) == yhat.map(level1)).mean()
    kappa = cohen_kappa_score(y, yhat)
    se = (acc * (1 - acc) / n) ** 0.5
    out = [md_table([{"pairs": n, "accuracy (level 2)": f"{acc:.3f} ± {1.96 * se:.3f}",
                      "accuracy (level 1)": f"{acc1:.3f}", "Cohen's kappa": f"{kappa:.3f}"}])]
    if confidence is not None:
        c = confidence.reset_index(drop=True)
        out.append(f"\n**Accuracy by {pred_name} confidence**\n")
        out.append(md_table([{"confidence": lev, "pairs": int((c == lev).sum()),
                              "accuracy": f"{(y[c == lev] == yhat[c == lev]).mean():.3f}" if (c == lev).any() else "-"}
                             for lev in ("high", "medium", "low")]))
    labs = [l for l in LABELS if (y == l).any() or (yhat == l).any()]
    p, r, f, s = precision_recall_fscore_support(y, yhat, labels=labs, zero_division=0)
    out.append(f"\n**Per category ({ref_name} as reference)**\n")
    out.append(md_table([{"category": l, "support": int(s_), "precision": f"{p_:.2f}", "recall": f"{r_:.2f}",
                          "F1": f"{f_:.2f}"} for l, p_, r_, f_, s_ in zip(labs, p, r, f, s)]))
    conf = pd.DataFrame({"ref": y, "pred": yhat})[y != yhat].value_counts().head(10)
    out.append(f"\n**Most frequent disagreements**\n")
    out.append(md_table([{ref_name: a, pred_name: b, "pairs": int(k)} for (a, b), k in conf.items()]))
    return "\n".join(out)


def evaluate_review(review: pd.DataFrame, llm: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    m = review.merge(llm[["uid", "target", "aspect", "confidence", "model", "prompt_version"]],
                     on="uid", how="left", suffixes=("_human", "_llm"))
    if (m["target_llm"].notna() & (m["target_llm"] != m["target_human"])).any():
        raise ValueError("targets differ between review sheet and LLM labels: uids are stale")
    have = m["aspect_llm"].notna()
    header = (f"# Silver aspect label quality (taxonomy {TAXONOMY_VERSION})\n\n"
              f"Human-labelled pairs: {len(m)} (marked unsure: {int(m['unsure'].sum())}); "
              f"with an LLM label: {int(have.sum())}. "
              f"LLM: `{', '.join(sorted(m.loc[have, 'model'].unique()))}` prompt "
              f"`{', '.join(sorted(m.loc[have, 'prompt_version'].unique()))}`.\n\n## All reviewed pairs\n")
    mh = m[have]
    body = agreement(mh["aspect_human"], mh["aspect_llm"], mh["confidence"], "human", "LLM")
    sure = mh[~mh["unsure"]]
    body += "\n## Excluding pairs the reviewer marked unsure\n" + agreement(sure["aspect_human"], sure["aspect_llm"],
                                                                         ref_name="human", pred_name="LLM")
    return header + body, m
