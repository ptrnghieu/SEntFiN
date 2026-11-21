# SEntFiN 1.0
Dataset and codes for SEntFiN 1.0

* `SEntFiN.csv`: 10,753 financial news headlines with entity-level sentiment
  (`{'entity': 'positive' | 'negative' | 'neutral'}`).
* `entity_list_comprehensive.csv`: ~1,000 NSE symbols and sector tags with their surface aliases.

## Data pipeline for aspect/target-based financial sentiment (`absa_data/`)

`absa_data` cleans **SEntFiN 1.0** and **FiQA 2018 Task 1**
([`TheFinAI/en-fiqa-absa`](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa)) and maps both
onto one *(text, target, polarity)* schema. It then builds leakage-safe splits, exports
ready-to-train files for encoder and LLM fine-tuning, and writes a data profile.

* **Which dataset to use, and why:** [`reports/DATASET_COMPARISON.md`](reports/DATASET_COMPARISON.md)
* **Generated statistics:** [`reports/data_profile.md`](reports/data_profile.md)

### Quick start

```bash
pip install -r requirements.txt
```

FiQA is gated. Accept the access form once on its Hugging Face page, then log in:

```bash
huggingface-cli login
```

Build everything:

```bash
python -m absa_data build
```

Run the tests:

```bash
python -m pytest -q
```

Without Hugging Face access the build still produces the SEntFiN outputs and prints a warning.
Useful options:

* `--no-fiqa`: SEntFiN only.
* `--fiqa-dir DIR`: use local `{train,valid,test}*.parquet` files instead of downloading.
* `--require-fiqa`: fail instead of skipping FiQA.
* `--resplit [--seed N]`: draw a new SEntFiN split and overwrite the manifest.

### Outputs (`data/processed/`, gitignored)

| path | content | use it for |
|---|---|---|
| `{sentfin,fiqa,combined}/pairs.parquet` / `.csv` | full unified table, all splits | analysis, custom formats |
| `*/pairs_{split}.jsonl` | one (text, target) per line, with `label` 0/1/2 | BERT/FinBERT/DeBERTa sentence-pair or target-marked classification |
| `*/sentences_{split}.jsonl` | one text per line, all targets with char spans | span / joint extraction + polarity models |
| `*/instruct_atsc_{split}.jsonl` | chat `messages` with targets given and a JSON answer | LLM SFT (LoRA etc.) |
| `sentfin/instruct_e2e_{split}.jsonl` | chat `messages`; the model must also find the entities | end-to-end LLM SFT |
| `issues.csv` | every repair, drop and flag, with the raw id | auditing |

Labels: `negative=0, neutral=1, positive=2`.

FiQA's 3-class label follows FinBen: `negative < -0.1 ≤ neutral < 0.1 ≤ positive`. This rule
reproduces all FinBen labels; the raw `score` is kept.

Never commit `data/`: it contains FiQA text, which may only be obtained through the gate.

### Splits

* **FiQA:** the official train/valid/test splits.
* **SEntFiN:** it has no official split, so `splits/sentfin_split_v1.csv` fixes one (≈80/10/10,
  `StratifiedGroupKFold`, seed 42).
  * All targets of a headline stay in one split, and so do near-duplicate headlines
    (`Infosys' Q2` vs `Infosys Q2`).
  * It is stratified on polarity × single/multi-target.
  * The manifest is committed, so everyone trains and tests on the same headlines whatever
    their library versions. The build fails loudly if `SEntFiN.csv` changes and the manifest
    no longer matches.

### Unified schema (`absa_data/schema.py`)

`uid, source, text_id, split, text_type, text, text_raw, target, target_raw, start, end,
span_match, polarity, label, score, aspect, aspect_l1, aspect_l2, entity_id, n_targets, is_mixed,
group_id, entity_seen_in_train, quality_flags`

* `span_match`: `exact | case_insensitive | substring | fuzzy | not_found`.
* `entity_seen_in_train`: the entity occurs in the training split of the same source. Use it to
  report results on unseen entities; SEntFiN has strong entity → label priors.
* `quality_flags` (`;`-separated):
  * text-level: `mojibake_repaired`, `section_prefix_removed`, `near_duplicate_across_splits`,
    `duplicate_merged`;
  * target-level: `entity_repaired`, `entity_whitespace_stripped`, `target_corrected`,
    `target_not_in_text`;
  * aspect-level: `aspect_normalized`, `aspect_multiple_kept_first`.

### Code layout

| module | role |
|---|---|
| `absa_data/sentfin.py` | SEntFiN parsing and repairs |
| `absa_data/fiqa.py` | FiQA download (pinned revision), cleaning, score → label |
| `absa_data/text.py` | encoding repair, near-duplicate keys, target-span alignment |
| `absa_data/entities.py` | alias table → canonical entity ids |
| `absa_data/splits.py` | grouped stratified split, manifest, leakage checks |
| `absa_data/export.py` | JSONL / parquet writers and prompt templates |
| `absa_data/profile.py` | `reports/data_profile.md` |
| `absa_data/cli.py` | `python -m absa_data build` |

### Citations

* SEntFiN: Sinha, Kedas, Kumar & Malo, *SEntFiN 1.0: Entity-aware sentiment analysis for
  financial news*, JASIST, 2022.
* FiQA: Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*, 2018.
* The FiQA access terms require citing FinBen: Xie et al., *FinBen: A Holistic Financial Benchmark
  for Large Language Models*, arXiv:2402.12659.
