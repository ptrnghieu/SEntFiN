"""Recompute the test metrics of finished pair-classification runs from their saved predictions.

Use after a change to the metrics (no retraining): reads runs/{sentiment,aspect}/**/test_predictions.jsonl,
joins the pairs by uid, rewrites the run's test block in metrics.json / metrics.md, then the summary.
The valid block is left as trained (validation predictions are not saved).

    python -m absa_models.rescore runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import summarize
from .common import ALL_SOURCES, load_table, write_results
from .pair_classify import report


def rescore(root: Path, processed: Path = Path("data/processed")) -> list[Path]:
    available = [s for s in ALL_SOURCES if (processed / s / "pairs.parquet").exists()]
    pairs = load_table(processed, "pairs", available).set_index("uid")
    done = []
    for task in ("sentiment", "aspect"):
        for f in sorted((root / task).glob("**/test_predictions.jsonl")):
            preds = pd.read_json(f, lines=True)
            missing = set(preds["uid"]) - set(pairs.index)
            if missing:
                raise ValueError(f"{f}: {len(missing)} uids not in {processed}; rebuild the data first")
            df = pairs.loc[preds["uid"]].reset_index()
            metrics = json.loads((f.parent / "metrics.json").read_text())
            metrics["test"] = report(df, None, task, pred=preds["pred"])
            metrics["config"]["rescored"] = True
            write_results(f.parent, metrics, f"{task.capitalize()} given the target")
            done.append(f.parent)
            print(f"rescored {f.parent}")
    return done


def main(argv=None) -> None:
    root = Path((argv or sys.argv[1:] or ["runs"])[0])
    rescore(root)
    summarize.main([str(root)])


if __name__ == "__main__":
    main()
