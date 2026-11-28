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

#### FiQA under FinBen's thresholds (`data/processed/fiqa_finben/`)

`fiqa` cuts FiQA scores at the thresholds calibrated to SEntFiN (neutral = [-0.1, 0.22)), so the
three sources share one meaning of the classes. Published FiQA numbers use FinBen's ±0.1. For
comparisons, `build` also writes `fiqa_finben`:

- only the 1,173 originally annotated pairs (no LLM-added targets), the official splits
  (822 / 117 / 234);
- `polarity` = FinBen's class (test: 84 negative / 12 neutral / 138 positive), soft labels from
  Normal(score, sigma) around ±0.1;
- ids `fiqa_finben-<n>#<k>`, the same numbers as in `fiqa`;
- not part of `combined`, and not a default training source: its neutral/positive boundary
  conflicts with SEntFiN's convention for 0.1 <= score < 0.22.

```bash
python -m absa_models.pair_classify --task sentiment --train-sources fiqa_finben --eval-sources fiqa_finben
```

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

### In-domain data: Vietnam news (`absa_data/vnnews.py`)

The deployment domain is English-language Vietnamese news crawled by
[minhhai205/aspect-based-FSA](https://github.com/minhhai205/aspect-based-FSA)
(`data/sentences_clean.json`: one record per sentence, no labels). One command pre-labels
every sentence with (target, sentiment, aspect) triplets and writes the review workbooks:

```bash
OPENAI_API_KEY=... python -m absa_data vnnews        # --no-llm: rebuild from the cached labels only
```

- **Labels** (all silver, `gpt-6-luna`, cached in `annotations/vnnews/`): targets with the
  target-completion prompt and no given targets (validated on FinEntity: recall 94%,
  in-scope precision 88%), then the same sentiment and 21-category aspect prompts as the
  training data. The text is kept exactly as crawled, so spans line up with the crawler's offsets.
- **Split** (`splits/vnnews_split_v1.csv`, frozen): whole articles, plus near-duplicate
  sentences across articles, go to one split. About 300 sentences form `test`, the rest is
  90/10 `train`/`valid`.
- **Human review of the test split**: `annotations/review/vnnews_test_review_v1_R{1,2,3}.xlsx`,
  one workbook per reviewer. The LLM's triplets are pre-filled: set `keep` to `n` for a
  non-target, fill `sentiment_fix` / `aspect_fix` only when a label is wrong, add missed
  targets on the `missing_targets` sheet. The rows with a blue `sent_id` (about 30 sentences)
  appear in every workbook, so agreement between reviewers can be measured; label them
  independently.
- Outputs: `data/processed/vnnews/{pairs,texts}.parquet` (same schema as the other sources,
  with `publisher` and `day` on texts), `reports/vnnews_labels.md`, and
  `annotations/vnnews/sentences_clean_labeled.json`: the crawler's file in its own format and
  order, with `split` and `triplets` (target, start, end, target_type, sentiment, aspect,
  confidences) added to every sentence.
- When the crawler fixes a sentence after labelling, its cached labels are dropped and only
  that sentence is labelled again (`annotations/vnnews/text_hashes.json`).

### Baselines: single-task FinBERT (steps 1a-1c)

Package `absa_models/`; backbone `yiyanghkust/finbert-pretrain` by default (`--model` to change).
Every run writes `metrics.json`, `metrics.md`, `test_predictions.jsonl` and `best.pt` under
`runs/<task>/<model>/<run>/`.

Encoders compared (`bash scripts/run_models.sh` runs the whole matrix for each; `MODELS="..."` to pick):

| `--model` | what it is |
|---|---|
| `yiyanghkust/finbert-pretrain` | FinBERT pretrained on financial text (10-K/10-Q, analyst reports, earnings calls), no task head |
| `ProsusAI/finbert` | BERT-base further trained on financial news, then fine-tuned for sentence sentiment (Financial PhraseBank); its classifier is dropped |
| `yiyanghkust/finbert-tone` | FinBERT fine-tuned for sentence tone on analyst reports; its classifier is dropped |
| `roberta-base` | general-domain RoBERTa, to measure what financial pretraining adds |
| `lstm` | no pretrained encoder: 2-layer BiLSTM over GloVe 6B-300d (`absa_models/lstm.py`); BiLSTM-CRF for 1a, attention + max pooling for 1b/1c; lr 1e-3, more epochs |

The two fine-tuned FinBERTs have seen sentence-level sentiment data: PhraseBank and analyst-report
sentences, not this project's test sets.

| step | command | trains on | key test metrics |
|---|---|---|---|
| 1a | `python -m absa_models.target_extraction --crf [--scheme bio-sentiment]` | FinEntity (+ completed SEntFiN/FiQA with `--train-sources`) | exact span P/R/F1 on FinEntity; `bio-sentiment` reproduces FinEntity's FinBERT-CRF (macro-F1 0.85) |
| 1b | `python -m absa_models.pair_classify --task sentiment [--soft] [--gold-only]` | pairs, target marked in the text | macro-F1 on human sentiment, per source, SEntFiN unseen entities, FiQA vs FinBen classes |
| 1c | `python -m absa_models.pair_classify --task aspect` | pairs | level-2 / level-1 macro-F1 (silver), FiQA vs human gold aspects |

On a GPU (Colab: Runtime > Change runtime type > T4 GPU), run each step as its own cell:

1. Clone the repository and switch to the branch:

   ```bash
   git clone https://github.com/ptrnghieu/SEntFiN && cd SEntFiN && git checkout claude/adoring-fermat-r9ta0k
   ```

2. Install the data and training requirements:

   ```bash
   pip install -r requirements.txt -r requirements-train.txt
   ```

3. Log in to Hugging Face. You need to have accepted the FiQA access form.

   ```bash
   huggingface-cli login
   ```

4. Rebuild `data/processed`. This takes about 15 s, and the LLM labels are already committed.

   ```bash
   python -m absa_data build
   ```

5. Run every baseline for one encoder (about 35-45 min on a T4), or for all five (about 3 h),
   and collect the results in `runs/SUMMARY.md`:

   ```bash
   bash scripts/run_baselines.sh                 # MODEL=roberta-base bash scripts/run_baselines.sh
   bash scripts/run_models.sh                    # all five encoders
   ```

On **Kaggle**, use [`notebooks/kaggle_baselines.ipynb`](notebooks/kaggle_baselines.ipynb). It reads the Hugging Face token from a Kaggle secret named `HF_TOKEN`; GPU T4 and internet must be enabled. Run it with *Save & Run All (Commit)*, so it keeps running in the background.

`tests/test_models.py` checks the code on CPU with a tiny random BERT: tags round-trip to the gold
spans, and every model can overfit a small set.

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
| `absa_models/` | single-task FinBERT baselines (target extraction, sentiment, aspect), `summarize` |
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
