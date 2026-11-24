"""Publish the processed datasets to the Hugging Face Hub.

``python -m absa_data push-hub --repo <user>/<name>`` stages ``data/processed``
in the Hub layout (one config per dataset x format, splits train / validation /
test), writes a dataset card generated from the data, and uploads it.

FiQA is distributed by TheFinAI behind an access form, so the repo is created
private, filled, gated (access form restating FiQA's terms) and only then made
public. Re-running replaces the data files and the card.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .schema import LABELS, SPLITS
from .taxonomy import LABELS as ASPECTS
from .taxonomy import map_fiqa_aspect

HF_SPLIT = {"train": "train", "valid": "validation", "test": "test"}
SOURCES = {"sentfin": "SEntFiN 1.0", "fiqa": "FiQA 2018 Task 1", "finentity": "FinEntity",
           "combined": "all sources", "fiqa_finben": "FiQA 2018 Task 1, FinBen ±0.1 classes"}
DERIVED = ("combined", "fiqa_finben")  # re-packagings of other sources; left out of the statistics


def split_config(name: str) -> tuple[str, str]:
    """'fiqa_finben_instruct_atsc' -> ('fiqa_finben', 'instruct_atsc'); longest source name first."""
    for source in sorted(SOURCES, key=len, reverse=True):
        if name == source or name.startswith(source + "_"):
            return source, name[len(source) + 1:] or "pairs"
    raise KeyError(name)
# Everything is published as parquet: `datasets` infers one file format per repo, so
# mixing parquet and JSONL configs makes the JSONL ones fail to load.
KINDS = {
    "pairs": "one (text, target) per row, all columns",
    "sentences": "one text per row, all its targets with spans, sentiment and aspect",
    "instruct_atsc": "chat messages: targets given -> sentiment per target",
    "instruct_aspect": "chat messages: targets given -> aspect + sentiment per target",
    "instruct_e2e": "chat messages: extract entities + sentiment",
    "instruct_e2e_aspect": "chat messages: extract (entity, aspect, sentiment) triplets",
}
DOCS = ("data_profile.md", "sentiment_label_calibration.md", "ASPECT_GUIDELINES.md", "aspect_label_quality.md",
        "target_completion_validation.md")

GATE = {
    "extra_gated_heading": "Access processed SEntFiN + FiQA data",
    "extra_gated_prompt": (
        "This dataset contains processed FiQA 2018 Task 1 data. TheFinAI distributes FiQA behind an access "
        "form (https://huggingface.co/datasets/TheFinAI/en-fiqa-absa). By requesting access you agree to the "
        "licenses of FiQA (MIT, as released by TheFinAI), SEntFiN 1.0 (MIT) and FinEntity (ODC-BY), and to cite "
        "FinBen, FiQA, SEntFiN and FinEntity in any resulting publication."
    ),
    "extra_gated_button_content": "Agree and access",
    "extra_gated_fields": {"I agree to the terms above and will cite FinBen, FiQA, SEntFiN and FinEntity": "checkbox"},
}


def _write_splits(table: pa.Table, sizes: list[int], out: Path, staging: Path) -> list[dict]:
    """Slice one table (one schema for all splits) into train / validation / test parquet files."""
    out.mkdir(parents=True, exist_ok=True)
    data_files, offset = [], 0
    for split, n in zip(SPLITS, sizes):
        dest = out / f"{HF_SPLIT[split]}.parquet"
        pq.write_table(table.slice(offset, n), dest)
        offset += n
        data_files.append({"split": HF_SPLIT[split], "path": dest.relative_to(staging).as_posix()})
    return data_files


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stage(processed: Path, reports: Path, staging: Path) -> list[dict]:
    """Copy processed files into `staging` in the Hub layout. Returns the config list."""
    if staging.exists():
        shutil.rmtree(staging)
    configs = []
    for source in SOURCES:
        src = processed / source
        if not (src / "pairs.parquet").exists():
            continue
        pairs = pd.read_parquet(src / "pairs.parquet")
        for kind in KINDS:
            if kind == "pairs":
                parts = [pairs[pairs["split"] == s] for s in SPLITS]
                table = pa.Table.from_pandas(pd.concat(parts, ignore_index=True), preserve_index=False)
                sizes = [len(p) for p in parts]
            else:
                files = [src / f"{kind}_{s}.jsonl" for s in SPLITS]
                if not all(f.exists() for f in files):
                    continue
                records = [_read_jsonl(f) for f in files]
                table = pa.Table.from_pylist([r for rs in records for r in rs])
                sizes = [len(rs) for rs in records]
            data_files = _write_splits(table, sizes, staging / source / kind, staging)
            name = source if kind == "pairs" else f"{source}_{kind}"
            configs.append({"config_name": name, "data_files": data_files, **({"default": True} if name == "sentfin" else {})})
    docs = staging / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    for d in DOCS:
        if (reports / d).exists():
            shutil.copyfile(reports / d, docs / d)
    if (processed / "issues.csv").exists():
        shutil.copyfile(processed / "issues.csv", docs / "issues.csv")
    return configs


def _count(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_card(staging: Path, configs: list[dict], code_url: str, calibration: dict | None) -> str:
    parts = [pd.read_parquet(p) for p in sorted(staging.glob("*/pairs/*.parquet")) if p.parts[-3] not in DERIVED]
    pairs = pd.concat(parts, ignore_index=True)
    sf, fq = pairs[pairs["source"] == "sentfin"], pairs[pairs["source"] == "fiqa"]
    has_fiqa = len(fq) > 0

    front = {
        "license": "mit",
        "language": ["en"],
        "pretty_name": "SEntFiN + FiQA for aspect-based financial sentiment",
        "task_categories": ["text-classification"],
        "task_ids": ["sentiment-classification"],
        "tags": ["finance", "aspect-based-sentiment-analysis", "absa", "entity-sentiment", "sentfin", "fiqa"],
        "size_categories": ["10K<n<100K"],
        "configs": configs,
        **(GATE if has_fiqa else {}),
    }
    yml = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000)

    rows = []
    for c in configs:
        counts = {d["split"]: _count(staging / d["path"]) for d in c["data_files"]}
        source, kind = split_config(c["config_name"])
        rows.append(f"| `{c['config_name']}` | {SOURCES[source]} | {KINDS[kind]} | "
                    f"{counts['train']:,} | {counts['validation']:,} | {counts['test']:,} |")

    def mix(d: pd.DataFrame) -> str:
        if "target_source" in d:
            d = d[d["target_source"] == "gold"]  # human labels only
        return " / ".join(_pct(d["polarity"].eq(l).mean()) for l in LABELS) if len(d) else "-"

    asp_rows = "\n".join(
        f"| {a} | {int((sf['aspect'] == a).sum()):,} | {int((fq['aspect'] == a).sum()):,} | "
        f"{int((pairs.loc[pairs['source'] == 'finentity', 'aspect'] == a).sum()):,} |" for a in ASPECTS)
    quality = ""
    if has_fiqa:
        g = fq.assign(gold=fq["fiqa_aspect"].map(map_fiqa_aspect)).dropna(subset=["gold"])
        quality = (f"* Agreement with FiQA's own gold aspects, where those map unambiguously onto the taxonomy "
                   f"({len(g)} pairs): {_pct((g['gold'] == g['aspect']).mean())} at level 2, "
                   f"{_pct((g['gold'].str.split('/').str[0] == g['aspect_l1']).mean())} at level 1. "
                   "FiQA's labels are themselves inconsistent, so this is a lower bound.\n")
    human = int((pairs["aspect_source"] == "human").sum())
    cal = calibration or {}
    thresholds = (f"negative < {cal['neg_below']} ≤ neutral < {cal['pos_from']} ≤ positive" if cal
                  else "negative < -0.1 ≤ neutral < 0.1 ≤ positive")

    body = f"""
# SEntFiN + FiQA for aspect-based financial sentiment analysis

Three financial (target, sentiment) datasets, cleaned and mapped onto **one schema**:

* **SEntFiN 1.0**: Indian financial news headlines with entity-level sentiment.
* **FiQA 2018 Task 1**, as packaged in [`TheFinAI/en-fiqa-absa`](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa):
  UK/US headlines and StockTwits posts with a sentiment score.
* **FinEntity** ([`yixuantt/FinEntity`](https://huggingface.co/datasets/yixuantt/FinEntity)): Reuters news,
  split into sentences, with every financial entity annotated (span + sentiment).

Both get:

* **harmonized sentiment labels** (hard classes + soft labels);
* **aspect categories** in one 21-category taxonomy (LLM silver labels);
* **leakage-safe splits**;
* ready-made files for encoder classifiers, span models and LLM instruction tuning.

Code that produced this release: {code_url}

## Access and licensing

Access is gated because FiQA is distributed behind an access form by TheFinAI. This repo repeats
its terms: agree to the licenses and cite FinBen. Licenses:

* SEntFiN 1.0: © 2021 Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo.
* FiQA: MIT, as released by TheFinAI.
* FinEntity: ODC-BY (Tang et al., EMNLP 2023).

Aspect labels and the calibration labels were generated with an OpenAI model (`gpt-6-luna`).

## Configurations

| config | source | content | train | validation | test |
|---|---|---|---|---|---|
{chr(10).join(rows)}

```python
from datasets import load_dataset

pairs = load_dataset("{{repo_id}}", "sentfin")            # or "fiqa", "finentity", "combined"
sft = load_dataset("{{repo_id}}", "combined_instruct_aspect")
```

You need to accept the access form first and be logged in (`huggingface-cli login`).

## Main fields (`*` pairs configs)

| field | meaning |
|---|---|
| `text`, `target` | the cleaned text and the entity / target |
| `start`, `end` | character span of the target in `text` (null if the target does not occur) |
| `polarity`, `label` | sentiment class: negative=0, neutral=1, positive=2. Same meaning in both sources (see below) |
| `p_negative`, `p_neutral`, `p_positive` | soft sentiment label; `soft_label` in JSONL files |
| `label_type`, `label_confidence` | `hard` (annotated class), `soft` (FiQA score) or `silver` (sentiment of an LLM-added target); confidence = max soft probability |
| `target_source`, `sentiment_source` | `gold` or `llm`: who found the target / labelled its sentiment |
| `target_annotation` | `exhaustive` (FinEntity), `llm_completed` (SEntFiN, FiQA: gold + LLM-added targets) |
| `target_type` | company, index_market, sector, commodity, currency, regulator_government, ... (LLM) |
| `score`, `polarity_finben` | FiQA only: original score in [-1, 1], and its class under FinBen's ±0.1 thresholds |
| `aspect`, `aspect_l1` | unified aspect category, e.g. `Corporate/Financial Results`, and its top level |
| `aspect_source`, `aspect_confidence` | `llm:<model>@<prompt version>` or `human`; LLM self-reported confidence |
| `fiqa_aspect*` | FiQA's original gold aspect path (normalized) |
| `n_targets`, `is_mixed` | number of targets in the text; whether their sentiments differ |
| `entity_seen_in_train` | target entity occurs in the same source's train split (for unseen-entity evaluation) |
| `group_id` | near-duplicate cluster (never split across train / validation / test) |
| `text_raw`, `target_raw`, `quality_flags` | original strings and the cleaning applied |

## Sentiment labels: hard (SEntFiN) vs soft (FiQA)

SEntFiN annotates classes; FiQA annotates a score. Under FinBen's usual ±0.1 thresholds, only 7.8 %
of FiQA is neutral, against 38 % of SEntFiN: the two "neutral"s do not mean the same thing.

**Calibrated thresholds.**
* An LLM taught SEntFiN's conventions with SEntFiN *train* examples was used as a bridge
  annotator.
* It reproduces SEntFiN gold on *validation* with {_pct(cal.get('bridge_vs_sentfin_accuracy', float('nan')))}
  accuracy.
* On FiQA, the pairs it calls neutral have a median score of about +0.2.
* FiQA scores are therefore cut at **{thresholds}**. A FinBen threshold was kept unless a
  bootstrap 95 % interval excluded it.
* FiQA labels are still computed from the human scores. `polarity_finben` keeps FinBen's classes
  for comparison with published results.

**Soft labels.**
* FiQA: the class probabilities if the true score is Normal(score, σ={cal.get('fiqa_score_sigma', 0.08)}).
  σ was estimated from tweets scored twice.
* SEntFiN: ordinal smoothing with the {cal.get('sentfin_flip_rate', 0.044)} adjacent-class flip rate
  measured on duplicated headlines.
* Train encoders with cross-entropy against the soft label. For LLM SFT, weight or filter records
  by `label_confidence`.

| | negative / neutral / positive |
|---|---|
| SEntFiN (gold) | {mix(sf)} |
| FiQA (gold, calibrated) | {mix(fq) if has_fiqa else '-'} |
| FinEntity (gold) | {mix(pairs[pairs["source"] == "finentity"])} |

Evidence: [`docs/sentiment_label_calibration.md`](docs/sentiment_label_calibration.md).

## Targets: gold and LLM-completed

SEntFiN and FiQA do not annotate every target: FiQA labels only the target its score is about, SEntFiN
matched a fixed entity list. Trained as plain span tagging, a model would learn to skip real entities.
So an LLM (`gpt-6-luna`) listed the missing targets of every SEntFiN and FiQA text:

* gold targets are never changed;
* an added target must occur verbatim, must not overlap a gold target, and references such as
  "the market" are dropped by rule;
* added targets get their sentiment from the bridge annotator (`label_type = silver`, soft label from
  its measured error rate) and an aspect from the aspect labeller.

Validated on FinEntity, which is exhaustive: with half of the gold targets hidden, the completion
recovers 94.3 % of them, and 88.1 % of its additions within FinEntity's own scope (companies, indices,
regulators) are gold ([`docs/target_completion_validation.md`](docs/target_completion_validation.md)).
Filter on `target_source == "gold"` to evaluate against human labels only.

## Aspect categories

* SEntFiN has no aspect labels. FiQA's taxonomy barely covers what SEntFiN is about (indices,
  regulators, commodities, currencies).
* So every pair of both datasets was labelled by `gpt-6-luna` in one taxonomy of 21 categories
  under Corporate / Stock / Market / Economy / Other
  ([`docs/ASPECT_GUIDELINES.md`](docs/ASPECT_GUIDELINES.md)).
* The model gets the category definitions, decision rules and examples taken only from training
  data.
* It must echo each target with its label, so labels cannot shift between targets.

**These are silver labels.**
{quality}* **No human verification** in this release ({human:,} human-checked pairs): every aspect label and
  every LLM-added target is model-generated. The evidence of quality is the agreement figures above,
  the FinEntity validation of the target completion and the bridge annotator's 83.9 % agreement
  with SEntFiN gold sentiment.
* Aspect results on `test` therefore measure agreement with the LLM, not accuracy. For an aspect
  evaluation against human labels, use FiQA's original gold aspects (`fiqa_aspect`).

| aspect | SEntFiN | FiQA | FinEntity |
|---|---|---|---|
{asp_rows}

## Splits

* **SEntFiN** has no official split. This release uses a frozen ~80/10/10 `StratifiedGroupKFold`
  split:
  * stratified on polarity × single/multi-target;
  * all targets of a headline, and near-duplicate headlines, stay in one split.
* **FiQA** keeps its official train / valid / test splits (here `validation` = valid). 3 tweets
  appear in two splits with different t.co links; they are flagged `near_duplicate_across_splits`.
* **`fiqa_finben`** is FiQA as benchmarks define it, for comparison with published numbers: only
  the 1,173 originally annotated pairs, the official splits, and FinBen's classes
  (negative < -0.1 ≤ neutral < 0.1 ≤ positive; test 84 / 12 / 138), with soft labels around ±0.1.
  The `fiqa` config instead uses the thresholds calibrated to SEntFiN (see below) and adds the
  LLM-completed targets. Do not train `fiqa_finben` together with the other sources: for
  0.1 ≤ score < 0.22 its classes contradict SEntFiN's convention. It is not part of `combined`.
* **⚠ Benchmark leak.** FinBen's sentence-level FiQA-SA (`ChanceFocus/flare-fiqasa`) splits the same
  rows differently: 175 of its 233 test sentences are in this FiQA `train`. Do not train here and
  report on FiQA-SA test, or the other way round.

## Cleaning applied

**SEntFiN**
* Repaired 19 entity names corrupted by a global find/replace ("Hold" became "neutral", "sell"
  became "negative", e.g. `Bajaj neutralings`).
* Repaired 8 truncated entities and parsed 13 malformed label cells.
* Merged 67 duplicate headlines, dropping 7 conflicting labels.
* Aligned target spans with word boundaries.

**FiQA**
* Repaired mojibake (`Â£` → `£`) and removed 22 glued section labels (`CompaniesTesco …`).
* Corrected 14 misspelled targets (original in `target_raw`); flagged 6 targets that do not occur
  in the text.
* Normalized aspect paths.

The full log is in [`docs/issues.csv`](docs/issues.csv); statistics are in [`docs/data_profile.md`](docs/data_profile.md).

## Limitations

* Aspect labels are LLM-generated (see above).
* The FiQA threshold calibration relies on an LLM bridge annotator. σ comes from only a handful of
  repeated tweets.
* SEntFiN has strong entity → label priors (`Sebi` is neutral 313/319 times). Report results on
  `entity_seen_in_train == False` too.
* FiQA test is small (234 pairs, ±5 pp at 80 % accuracy) and has almost no multi-target
  sentences.

## Citation

Please cite the original datasets and FinBen:

* Sinha, Kedas, Kumar & Malo. *SEntFiN 1.0: Entity-aware sentiment analysis for financial news.*
  JASIST, 2022.
* Maia et al. *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering.* WWW '18
  Companion, 2018.
* Xie et al. *FinBen: A Holistic Financial Benchmark for Large Language Models.* arXiv:2402.12659.
* Tang, Yang, Huang, Tam & Tang. *FinEntity: Entity-level Sentiment Classification for Financial Texts.*
  EMNLP 2023.
"""
    return f"---\n{yml}---\n{body}"


def push(staging: Path, repo_id: str, gated: str | None, public: bool, commit_message: str, log=print) -> dict:
    """Create (private) -> upload -> gate -> make public, verifying the gate before going public."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    if gated and not info.private and not info.gated:
        # An existing public repo: gate it before any gated content lands.
        api.update_repo_settings(repo_id, repo_type="dataset", gated=gated)
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=str(staging), commit_message=commit_message,
        delete_patterns=[f"{s}/**" for s in SOURCES] + ["docs/**"],
    )
    log(f"uploaded {staging} -> {repo_id}")
    if gated:
        api.update_repo_settings(repo_id, repo_type="dataset", gated=gated)
        if api.dataset_info(repo_id).gated != gated:
            raise RuntimeError("gating could not be verified; repo left private")
    if public:
        api.update_repo_settings(repo_id, repo_type="dataset", private=False)
    info = api.dataset_info(repo_id)
    return {"repo": repo_id, "private": info.private, "gated": info.gated, "sha": info.sha,
            "url": f"https://huggingface.co/datasets/{repo_id}"}


def write_staging(processed: Path, reports: Path, staging: Path, repo_id: str, code_url: str,
                  calibration_file: Path) -> Path:
    configs = stage(processed, reports, staging)
    cal = json.loads(calibration_file.read_text()) if calibration_file.exists() else None
    card = build_card(staging, configs, code_url, cal).replace("{repo_id}", repo_id)
    (staging / "README.md").write_text(card, encoding="utf-8")
    return staging
