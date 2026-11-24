# SEntFiN 1.0
Dataset and codes for SEntFiN 1.0

* `SEntFiN.csv`: 10,753 financial news headlines with entity-level sentiment
  (`{'entity': 'positive' | 'negative' | 'neutral'}`).
* `entity_list_comprehensive.csv`: ~1,000 NSE symbols and sector tags with their surface aliases.

## Data pipeline for aspect/target-based financial sentiment (`absa_data/`)

`absa_data` cleans **SEntFiN 1.0**, **FiQA 2018 Task 1**
([`TheFinAI/en-fiqa-absa`](https://huggingface.co/datasets/TheFinAI/en-fiqa-absa)) and **FinEntity**
([`yixuantt/FinEntity`](https://huggingface.co/datasets/yixuantt/FinEntity)) and maps them onto one
*(text, target, polarity)* schema. It then builds leakage-safe splits, exports
ready-to-train files for encoder and LLM fine-tuning, and writes a data profile.

* **Which dataset to use, and why:** [`reports/DATASET_COMPARISON.md`](reports/DATASET_COMPARISON.md)
* **Generated statistics:** [`reports/data_profile.md`](reports/data_profile.md)
* **Exploratory data analysis (figures, deployment-domain gap):** [`reports/eda/EDA.md`](reports/eda/EDA.md),
  regenerated with `python -m absa_data eda` (needs `matplotlib`; reads the crawler exports in
  `../minhhai205/aspect-based-fsa/data/exports` when present, `--target-domain` to change).

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
| `*/pairs_{split}.jsonl` | one (text, target) per line, with `label` 0/1/2 and `soft_label` | BERT/FinBERT/DeBERTa sentence-pair or target-marked classification |
| `*/sentences_{split}.jsonl` | one text per line, all targets with char spans | span / joint extraction + polarity models |
| `*/instruct_atsc_{split}.jsonl` | chat `messages` with targets given and a JSON answer | LLM SFT (LoRA etc.) |
| `sentfin/instruct_e2e_{split}.jsonl` | chat `messages`; the model must also find the entities | end-to-end LLM SFT |
| `*/instruct_aspect_{split}.jsonl` | chat `messages`; targets given, answer = aspect category + sentiment per target | aspect-category sentiment (ACSA) SFT |
| `sentfin/instruct_e2e_aspect_{split}.jsonl` | chat `messages`; answer = (target, aspect, sentiment) triplets | end-to-end ABSA SFT |
| `issues.csv` | every repair, drop and flag, with the raw id | auditing |

Labels: `negative=0, neutral=1, positive=2`.

Sentiment labels are harmonized across the two sources:
* SEntFiN's classes are **hard**; FiQA's come from a noisy **score**.
* FiQA scores are cut at `negative < -0.1 ≤ neutral < 0.22 ≤ positive`, a cut calibrated to
  SEntFiN's notion of neutral (see below). FinBen's ±0.1 cut is kept in `polarity_finben`, and the
  raw `score` is kept too.
* Every pair also gets a soft label, `soft_label` = [P(neg), P(neu), P(pos)].

Never commit `data/`: it contains FiQA text, which may only be obtained through the gate.

### Aspect categories (silver labels from an LLM, verified by hand)

Only FiQA has aspect labels, and its taxonomy fits SEntFiN poorly: SEntFiN is dominated by indices,
regulators, commodities and currencies, which FiQA barely covers. So both datasets are labelled in
one **unified taxonomy** of 21 categories under Corporate / Stock / Market / Economy / Other.
`absa_data/taxonomy.py` is the single source of truth; it generates
[`reports/ASPECT_GUIDELINES.md`](reports/ASPECT_GUIDELINES.md).

1. **LLM labels.** The OpenAI Responses API is called with a strict JSON schema; the default model is
   `gpt-6-luna`.

   ```bash
   export OPENAI_API_KEY=...
   ```

   ```bash
   python -m absa_data annotate --source fiqa
   ```

   ```bash
   python -m absa_data annotate --source sentfin
   ```

   * Labels are appended to `annotations/llm/<source>_<model>.jsonl` (committed).
   * Runs are resumable: re-running only labels pairs that are missing or that were labelled with an
     older prompt version.
   * Token usage per run goes to `*.usage.jsonl`.
   * `--dry-run` prints the prompt without calling the API.
   * `--limit N`, `--splits`, `--text-ids FILE` restrict the run.
2. **Human check.** `annotations/review/sentfin_test_review_v1.xlsx` holds 400 random SEntFiN *test*
   headlines (538 pairs).
   * It is blind: the LLM's label is not shown.
   * It has a dropdown of the categories and a guidelines sheet.
   * Fill the yellow `aspect` column, then run:

   ```bash
   python -m absa_data review-eval
   ```

   This writes `reports/aspect_label_quality.md` (accuracy, Cohen's kappa, per-category scores,
   confusions). It also stores your labels as gold in `annotations/human/`; they override the LLM's
   labels in the build.
3. **Build.** `python -m absa_data build` attaches the labels:
   * `aspect`, `aspect_l1`;
   * `aspect_source`: `human` or `llm:<model>@<prompt version>`;
   * `aspect_confidence`.

   It also writes the aspect exports, and the profile compares the LLM labels with FiQA's own gold
   aspects where those map unambiguously. FiQA's original labels stay in `fiqa_aspect*`.

### Sentiment labels: hard (SEntFiN) vs soft (FiQA)

Evidence and numbers: [`reports/sentiment_label_calibration.md`](reports/sentiment_label_calibration.md).
Summary and training recipes: the "Hard vs soft sentiment labels" section of
[`reports/DATASET_COMPARISON.md`](reports/DATASET_COMPARISON.md).

* **Calibration.** An LLM taught SEntFiN's conventions acts as a bridge annotator: it scores
  83.9 % against SEntFiN gold on valid, and labels FiQA, which places FiQA's neutral/positive cut
  at 0.22 instead of 0.1 (fitted on FiQA train + valid only). FiQA labels are still computed from
  the human scores. To re-run it:

  ```bash
  python -m absa_data annotate --task sentiment --source sentfin --splits valid
  ```

  ```bash
  python -m absa_data annotate --task sentiment --source fiqa
  ```

  ```bash
  python -m absa_data calibrate
  ```

  This writes `annotations/calibration/fiqa_thresholds.json`, which the build reads.
* **Soft labels** (`p_negative`, `p_neutral`, `p_positive`, `label_confidence`, `label_type`):
  * FiQA: Normal(score, σ) around the thresholds, with σ ≈ 0.08 estimated from re-scored tweets.
  * SEntFiN: ordinal smoothing with the flip rate measured on duplicate headlines (≈ 4 %).
  * Build options: `--fiqa-thresholds {calibrated,finben}`, `--fiqa-sigma`,
    `--sentfin-flip-rate` (0 = one-hot).

### Publishing to the Hugging Face Hub

```bash
python -m absa_data push-hub --repo <user>/<name>
```

* The command stages `data/processed` as one config per dataset × format, with splits
  `train / validation / test`, and writes a dataset card generated from the data.
* If FiQA is included, the repo is created private, uploaded, **gated** with a form that restates
  FiQA's terms (cite FinBen), and only then made public. It refuses to publish FiQA ungated.
* `--private` keeps the repo private; `--dry-run` only stages the files under `data/hub_staging/`.

### Targets: FinEntity and LLM completion

* **FinEntity** is the only source whose targets are exhaustive.
  * It is loaded from the Hub at a pinned revision, or locally with `--finentity-path`; skip it
    with `--no-finentity`.
  * Spans are repaired, Reuters codes removed, and paragraphs split into sentences with the
    crawler's splitter (vendored in `absa_data/sentsplit.py`).
  * Sentences without any entity are kept in `texts.parquet` and in the `sentences` /
    end-to-end exports.
* **SEntFiN and FiQA** only label some targets, so an LLM lists the missing ones:

  ```bash
  python -m absa_data complete-targets --source finentity --validate --splits valid test
  ```

  ```bash
  python -m absa_data complete-targets --source sentfin
  ```

  ```bash
  python -m absa_data complete-targets --source fiqa
  ```

  Then label the added targets:

  ```bash
  python -m absa_data annotate --task sentiment --llm-targets --source sentfin
  ```

  ```bash
  python -m absa_data annotate --source sentfin
  ```

  * The first command checks the method on FinEntity; results are in
    [`reports/target_completion_validation.md`](reports/target_completion_validation.md).
  * Completions are cached in `annotations/completion/`; sentiment of added targets in
    `annotations/llm_sentiment/`.
  * Added targets carry `target_source = llm` and `label_type = silver`; gold rows are never
    changed.

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
span_match, polarity, label, score, polarity_finben, label_type, sentiment_confidence, p_negative,
p_neutral, p_positive, label_confidence, paragraph_id, target_annotation, target_source,
sentiment_source, target_type, target_confidence, aspect, aspect_l1, aspect_source, aspect_confidence,
fiqa_aspect, fiqa_aspect_l1, fiqa_aspect_l2, entity_id, n_targets, is_mixed, group_id,
entity_seen_in_train, quality_flags`

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
| `absa_data/taxonomy.py` | unified aspect taxonomy, rules, examples, FiQA mapping |
| `absa_data/annotate.py` | LLM aspect labelling (batched, strict JSON, resumable) |
| `absa_data/review.py` | blind review sheet, agreement metrics |
| `absa_data/aspects.py` | attach LLM and human aspect labels to the pairs |
| `absa_data/calibration.py` | soft sentiment labels, noise estimates, FiQA threshold calibration |
| `absa_data/finentity.py`, `sentsplit.py` | FinEntity loader; crawler's normalizer and sentence splitter |
| `absa_data/completion.py` | LLM target completion, validation on FinEntity |
| `absa_data/eda.py` | exploratory data analysis report and figures |
| `absa_data/hub.py` | Hugging Face Hub staging, dataset card, gated upload |
| `absa_data/splits.py` | grouped stratified split, manifest, leakage checks |
| `absa_data/export.py` | JSONL / parquet writers and prompt templates |
| `absa_data/profile.py` | `reports/data_profile.md` |
| `absa_data/cli.py` | `python -m absa_data build / annotate / complete-targets / review-sheet / review-eval / calibrate / push-hub / eda` |

### Citations

* SEntFiN: Sinha, Kedas, Kumar & Malo, *SEntFiN 1.0: Entity-aware sentiment analysis for
  financial news*, JASIST, 2022.
* FiQA: Maia et al., *WWW'18 Open Challenge: Financial Opinion Mining and Question Answering*, 2018.
* FinEntity: Tang et al., *FinEntity: Entity-level Sentiment Classification for Financial Texts*,
  EMNLP 2023 (ODC-BY).
* The FiQA access terms require citing FinBen: Xie et al., *FinBen: A Holistic Financial Benchmark
  for Large Language Models*, arXiv:2402.12659.
