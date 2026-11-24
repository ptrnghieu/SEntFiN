"""Collect runs/**/metrics.json into one comparison table: python -m absa_models.summarize [runs]."""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = {  # (task dir, metric subset, metric) shown per run
    "target_extraction": [("finentity spans", "f1"), ("finentity spans+sentiment", "macro_f1"), ("all spans", "f1")],
    "sentiment": [("all, gold sentiment only", "macro_f1"), ("sentfin, gold only", "accuracy"), ("sentfin, gold only", "macro_f1"),
                  ("sentfin, gold, unseen entities", "macro_f1"), ("fiqa, gold only", "macro_f1"),
                  ("fiqa vs FinBen classes (polarity_finben)", "macro_f1"), ("finentity, gold only", "macro_f1")],
    "aspect": [("all", "macro_f1"), ("all, level 1", "macro_f1"), ("fiqa vs human gold aspects (mapped)", "accuracy"),
               ("fiqa vs human gold aspects (mapped)", "macro_f1")],
}


def main(argv=None) -> None:
    root = Path((argv or sys.argv[1:] or ["runs"])[0])
    lines = ["# Baseline results (test split)\n"]
    for task, keys in KEYS.items():
        runs = sorted((root / task).glob("**/metrics.json"))
        if not runs:
            continue
        lines += [f"## {task}\n", "| run | " + " | ".join(f"{s} · {m}" for s, m in keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for f in runs:
            test = json.loads(f.read_text())["test"]
            vals = [test.get(s, {}).get(m, "") for s, m in keys]
            lines.append(f"| {f.parent.relative_to(root / task)} | " + " | ".join(str(v) for v in vals) + " |")
        lines.append("")
    out = root / "SUMMARY.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
