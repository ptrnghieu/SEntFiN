"""Mean ± standard deviation over seeds.

Runs are laid out as ``<root>/<system>/<variant>-s<seed>/metrics.json`` (scripts/run_seeds.sh).
Writes ``<root>/SEEDS.md``: one row per (system, variant) with the mean ± sd over seeds, the
per-seed values, and the number of seeds.

    python -m absa_models.aggregate runs/seeds
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KEYS = [("all triplet", "f1"), ("all span+sentiment", "f1"), ("all span+aspect", "f1"), ("all span", "f1"),
        ("finentity triplet", "f1"), ("finentity span+sentiment", "f1"), ("sentfin triplet", "f1"),
        ("fiqa triplet", "f1")]
_SEED = re.compile(r"^(?P<variant>.*?)-?s(?P<seed>\d+)$")


def collect(root: Path) -> dict[tuple[str, str], dict[int, dict]]:
    groups: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for f in sorted(root.glob("*/*/metrics.json")):
        m = _SEED.match(f.parent.name)
        if not m:
            continue
        test = json.loads(f.read_text())["test"]
        if "all triplet" not in test:  # component runs (extractor, sentiment, aspect): no triplet scores
            continue
        groups[(f.parent.parent.name, m["variant"] or "-")][int(m["seed"])] = test
    return groups


def fmt(vals: list[float]) -> str:
    if not vals:
        return ""
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.3f} ± {sd:.3f}"


def table(groups) -> str:
    lines = ["# Results over seeds (test split, mean ± sd)\n",
             "| system | variant | seeds | " + " | ".join(f"{k} · {m}" for k, m in KEYS) + " |",
             "|---|---|---|" + "---|" * len(KEYS)]
    for (system, variant), by_seed in sorted(groups.items()):
        cells = [fmt([r[k][m] for r in by_seed.values() if k in r]) for k, m in KEYS]
        lines.append(f"| {system} | {variant} | {len(by_seed)} | " + " | ".join(cells) + " |")
    lines += ["", "## Per seed: all triplet F1\n", "| system | variant | " + " | ".join(
        f"s{s}" for s in sorted({s for g in groups.values() for s in g})) + " |"]
    seeds = sorted({s for g in groups.values() for s in g})
    lines.append("|---|---|" + "---|" * len(seeds))
    for (system, variant), by_seed in sorted(groups.items()):
        lines.append(f"| {system} | {variant} | " + " | ".join(
            f"{by_seed[s]['all triplet']['f1']:.3f}" if s in by_seed else "" for s in seeds) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    root = Path((argv or sys.argv[1:] or ["runs/seeds"])[0])
    text = table(collect(root))
    (root / "SEEDS.md").write_text(text)
    print(text)
    print(f"-> {root / 'SEEDS.md'}")


if __name__ == "__main__":
    main()
