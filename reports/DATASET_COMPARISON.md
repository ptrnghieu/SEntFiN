# SEntFiN 1.0 vs FiQA-ABSA: which one to train on

All numbers come from `reports/data_profile.md`, which `python -m absa_data build` regenerates.
They are counted after cleaning (see "What was wrong in the raw data" below).

## Recommendation

The project predicts **aspect categories** as well as sentiment. Neither dataset is enough on
its own:
* SEntFiN has no aspect labels.
* FiQA has aspect labels, but only 1,173 pairs. They are inconsistent, and its taxonomy does not
  cover what SEntFiN is mostly about (indices, regulators, commodities, currencies).

The repo therefore does the following:

1. **One unified taxonomy for both datasets:** 21 categories under Corporate / Stock / Market /
   Economy / Other ([`ASPECT_GUIDELINES.md`](ASPECT_GUIDELINES.md)).
2. **Silver aspect labels for every pair of both datasets** from `gpt-6-luna`, using the same prompt
   and rules (15,491 pairs). FiQA's original gold aspects are kept in `fiqa_aspect*`.
3. **No human review in this release** (decided to skip). Quality evidence comes from agreement
   with FiQA's gold aspects, the FinEntity validation of the target completion, and the bridge
   annotator's agreement with SEntFiN gold sentiment. The blind review tooling
   (`review-sheet` / `review-eval`) stays in the repo if a check is wanted later.

**Main training source: SEntFiN.** It has 14,318 pairs with human sentiment labels and LLM aspect
labels. Add FiQA train for US/UK news and tweets. Aspect results on the silver test labels say
how well a model imitates the LLM; FiQA's original gold aspects are the only human aspect labels.

## Side by side

| | SEntFiN 1.0 | FiQA 2018 Task 1 (`TheFinAI/en-fiqa-absa`) |
|---|---|---|
| Texts / (text, target) pairs | **10,679 / 14,318** | 1,111 / 1,173 |
| Text | Indian financial news headlines | 436 UK/US headlines + 675 StockTwits posts |
| Annotation | entity → negative / neutral / positive | target → score in [-1, 1] + aspect category path |
| 3-class labels | annotated by humans (**hard**) | derived from a noisy score (**soft**). Here: neg < -0.1 ≤ neu < **0.22** ≤ pos, calibrated to SEntFiN on train + valid; FinBen's ±0.1 kept in `polarity_finben` |
| Label mix neg / neu / pos | 26.5 / 38.2 / 35.3 % | 30.9 / 16.2 / 52.9 % (FinBen: 30.9 / **7.8** / 61.3 %) |
| Label noise (estimated) | 7 of 80 repeated judgements disagree, all negative ↔ neutral | score noise σ ≈ 0.08, as wide as FinBen's neutral band |
| Texts with >1 target | **2,836 (26.6 %)** | 57 (5.1 %) |
| Texts whose targets disagree in polarity | **1,228 (11.5 %)** | 11 (1.0 %); **0 in test** |
| Official split | none (this repo ships a frozen, leakage-safe 80/10/10) | 822 / 117 / 234 pairs |
| Test pairs / 95 % CI on an accuracy near 80 % | 1,433 / ±2.1 pp | 234 / ±5.1 pp |
| Neutral pairs in test | 547 | 30 (FinBen: **12**, ±28 pp on that class) |
| Aspect categories (original) | none | 4 top-level, 28 second-level (3 never in train, 5 with < 5 train examples) |
| Aspect categories (this repo) | silver, 21 unified categories (not human-verified) | silver, same 21 categories |
| Access | MIT, in this repo | MIT, gated (access form, cite FinBen) |

## Hard vs soft sentiment labels

**The problem.** SEntFiN gives a class; FiQA gives a score in [-1, 1] that has to be cut into
classes. Three findings ([`sentiment_label_calibration.md`](sentiment_label_calibration.md)):

1. **FiQA labels near a threshold are close to coin flips.** Tweets that appear twice were scored
   with noise σ ≈ 0.08, about as wide as FinBen's neutral band (±0.1). 201 pairs (17 %) lie within
   0.1 of a FinBen threshold. The estimate rests on only 4 repeated pairs.
2. **SEntFiN's hard labels are fairly stable.** 73 of 80 repeated judgements in duplicated
   headlines agree, and all 7 disagreements are negative ↔ neutral. That is a flip rate of about
   4 % per label, only ever to an adjacent class.
3. **"Neutral" means different things.**
   * Under FinBen's thresholds, 7.8 % of FiQA is neutral, against 38 % of SEntFiN.
   * An LLM taught SEntFiN's conventions (21 SEntFiN *train* examples) reproduces SEntFiN gold on
     *valid* with 83.9 % ± 1.9 accuracy (κ 0.76). It calls 22.9 % of FiQA neutral.
   * The FiQA pairs it calls neutral have a **median score of +0.20**, not 0: FiQA's scale is
     shifted positive.

**What the build does**

* **One meaning of negative / neutral / positive across sources.**
  * FiQA scores are cut at **neg < −0.1 ≤ neu < 0.22 ≤ pos**, fitted on FiQA **train + valid**
    only (939 pairs), so the test split plays no part in defining its own labels.
  * The positive cut-off moved: the best fit is 0.22, with a bootstrap 95 % interval
    [0.18, 0.28] that excludes FinBen's 0.1. On the held-out test, the new cut agrees with the
    bridge on 80.8 % of pairs against 78.2 % for FinBen.
  * The negative cut-off stays at FinBen's −0.1, because the fit is flat from −0.16 to 0.
  * In effect, 99 mildly positive FiQA pairs (scores 0.10–0.22) become neutral.
  * FinBen's classes stay in `polarity_finben` for comparison with published FiQA results.
* **Soft labels for every pair.** `label_type` says hard / soft. `p_negative`, `p_neutral`,
  `p_positive` are the soft label (`soft_label` in the JSONL files), and `label_confidence` is
  their maximum.
  * FiQA: the probability of each class if the true score is Normal(score, σ = 0.083), cut at
    the thresholds above. A score of 0.15 → [0.00, 0.80, 0.20]; 0.22 → [0.00, 0.50, 0.50]; 0.6 → [0, 0, 1].
  * SEntFiN: ordinal smoothing with the measured flip rate (4.4 %), moved only to adjacent
    classes. Neutral → [0.02, 0.96, 0.02]; positive → [0, 0.04, 0.96].
  * Options: `--sentfin-flip-rate 0` gives SEntFiN one-hot labels, and
    `--fiqa-thresholds finben` keeps FinBen's cut.

**How to train with them**

* **Encoder classifiers:** use cross-entropy against `soft_label` (identical to ordinary CE for
  one-hot rows), so one loss covers both sources:

  ```python
  loss = -(soft_label * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
  ```

  Optionally add a regression head on `score`, masked to FiQA rows.
* **LLM SFT** (text answers, so hard labels): every instruct record carries `label_confidence`
  (lowest among its targets). Use it to weight the loss, or drop records below a cut-off.
  A cut-off of 0.8 removes the 200 gold FiQA pairs closest to a threshold and the 1,196 silver
  pairs whose bridge label had medium or low confidence; gold SEntFiN and FinEntity rows (0.956)
  are never affected.
* **Evaluation:** use hard `polarity` (SEntFiN gold, calibrated FiQA). For numbers comparable
  with FiQA papers, also report FiQA test on `polarity_finben`, or regression metrics on `score`.

## FinEntity and LLM-completed targets

**FinEntity** (Tang et al., EMNLP 2023; ODC-BY) is the third source and the only one with
*exhaustive* targets: every financial entity in its 979 Reuters paragraphs has a span and a
sentiment.

* **Cleaning.**
  * 70 spans shifted by broken characters were re-anchored.
  * Reuters codes (`<JNJ.N>`, `[nL4N…]`) were removed and the spans recomputed.
* **Sentence splitting.** Paragraphs are split with the crawler's own normalizer and splitter, so
  training text looks like deployment text; no gold span is ever cut.
* **Result:** 1,371 sentences with 2,131 targets, plus 215 sentences without any entity (true
  negatives for extraction).
* **Split:** frozen 80/10/10 manifest, grouped by paragraph and duplicate text.
* **Sentiment convention:** the bridge annotator agrees with FinEntity gold on 90.9 %, so its labels
  are used as they are.

**SEntFiN and FiQA targets are completed by an LLM.** FiQA labels only the target its score is
about (all 93 posts with ≥ 2 cashtags have one target), and SEntFiN matched a fixed entity list.
`gpt-6-luna` therefore listed the missing targets of every text, under rules enforced in code:

* gold targets are never changed;
* additions must be verbatim and must not overlap gold;
* "the market"-style references are dropped.

| | gold targets | LLM-added | texts with an addition |
|---|---|---|---|
| SEntFiN | 14,318 | 2,865 | 2,502 of 10,679 |
| FiQA | 1,173 | 702 | 451 of 1,111 |

The additions are mostly what the original annotators skipped: regulators and governments, indices,
sectors, fund / investor groups, rates and countries.

* **Labels for the added targets.** Sentiment comes from the bridge annotator (`label_type =
  silver`, soft label from its error rate at the confidence it reported), and the aspect from the
  aspect labeller.
* **Validation on FinEntity.** Half of each sentence's gold targets were hidden and the completion
  had to find them: **94.3 %** were recovered, and **88.1 %** of its additions within FinEntity's
  scope (companies, indices, regulators) are gold. Additions outside that scope (currencies,
  sectors, countries) follow SEntFiN's convention, which FinEntity does not share
  ([`target_completion_validation.md`](target_completion_validation.md)).
* **Columns:** `target_source` / `sentiment_source` (`gold` or `llm`), `target_annotation`
  (`exhaustive` / `llm_completed`), `target_type`. The end-to-end instruct exports now use only
  exhaustive or LLM-completed texts.
* **Evaluation:** target extraction should be evaluated on FinEntity and the Vietnam test set. Sentiment and aspect results against human labels need
  `target_source == "gold"`.

## Silver aspect labels: how they were made and how good they are

* **Model and settings:** `gpt-6-luna` through the OpenAI Responses API, low reasoning effort,
  10 texts per request, strict JSON schema restricted to the 21 categories.
* **Prompt:** category definitions, 8 decision rules, and 20 examples taken only from SEntFiN
  *train* or invented. A test checks that none comes from a valid or test split.
* **Cost:** all runs including the pilots used 3.63M input tokens (3.11M of them served from the
  prompt cache) and 0.63M output tokens. The full SEntFiN run took 12 minutes.
* **A bug caught in the pilot.** With bare keys, the model sometimes **shifted labels by one
  position** inside a batch, so ~10 % of labels belonged to the neighbouring target
  ("DLF's Q3 profits fall 30 %" → Market/Equity Market).
  * The model must now echo each target next to its label, and a batch whose echo does not match
    is rejected and retried.
  * After the fix, 0 of 1,180 requests were rejected.
* **Agreement with FiQA's gold aspects**, on the 569 FiQA pairs whose original label maps
  unambiguously onto the unified taxonomy:

  | | value |
  |---|---|
  | Level 2 accuracy | 77.3 % ± 3.4 |
  | Level 1 accuracy | 88.6 % |
  | Cohen's kappa | 0.72 |
  | Accuracy, LLM confidence "high" (84 % of pairs) | 82.4 % |
  | Accuracy, LLM confidence "medium" | 51.6 % |

  * This is a lower bound. FiQA's labels are themselves inconsistent, and most remaining
    disagreements are definitional: FiQA files a tweet like "Still short $LNG from $11.70" under
    Short Selling, while the unified rules treat an author's own position as a recommendation.
  * The mapping was revised once after the first pilot, then frozen so that it is not tuned
    towards the LLM's answers.
* **Human agreement:** not measured; the human review was skipped for this release.
* **Distribution on SEntFiN:** ~40 % of pairs fall in Market/* and Economy/* categories, where
  FiQA has at most 25 examples per category. The largest categories are Financial Results 11.7 %,
  Equity Market 11.4 %, Outlook & Recommendations 11.4 %, Sectors 9.4 % and Commodities 8.6 %.
  The smallest are Funds & Flows (66 pairs), Macroeconomy & Trade (79) and Risks & Reputation (96).

**How to use them**
* Train on all silver labels.
* Optionally down-weight or drop `aspect_confidence == "medium" | "low"` (16 % of pairs) and
  compare.
* If the human agreement turns out low for some categories, refine their definitions in
  `taxonomy.py` and re-run `annotate`. Only pairs whose prompt version changed are re-labelled.

## Why SEntFiN should be the main source

1. **It is 12 times larger.** 14.3k labelled pairs, against 1.2k.
2. **It tests what makes ABSA harder than plain sentiment classification.** In 1,228 headlines,
   different entities get different polarities ("Gold shines on seasonal demand; Silver dull").
   FiQA's test split has no such sentence, and 98 % of its test sentences have a single target.
   A model that ignores the target entirely therefore loses almost nothing on FiQA test.
3. **Its labels are balanced and annotated directly.** FiQA's classes come from cutting a noisy
   score. Its neutral class is 7.8 % under FinBen's thresholds (12 test pairs) and 16 % after
   calibration (30 test pairs). Macro-F1 on FiQA test still swings by several points with a
   handful of predictions.
4. **Its test numbers are about 2.5 times more precise** (±2.1 pp against ±5.1 pp).

## SEntFiN caveats to design around

* **Entity shortcuts.** 86 % of test pairs mention an entity seen in training. Many entities
  almost always carry the same label: `Sebi` is neutral 313 of 319 times, and `RBI` 86 of 88.
  Predicting each entity's majority training label scores **53.4 % accuracy / 0.516 macro-F1**,
  against 38.2 % for always predicting the majority class.
  * Report results on the **199 test pairs whose entity is unseen**
    (`entity_seen_in_train == False`) as well as on the full test set.
  * Consider target-masking augmentation: replace the target with a placeholder for a share of
    the training examples.
* **One domain.** Only 18.4 % of FiQA test targets appear in SEntFiN train, and 25 % of FiQA
  test word tokens are out of SEntFiN's vocabulary. The FiQA test set is therefore a
  meaningful out-of-domain check.
* **Some label noise.** 7 headlines were annotated twice with different labels; those
  conflicting labels were dropped.

## How to use FiQA well

* **As an out-of-domain test.** Report headlines and posts separately; posts are far noisier
  (cashtags, URLs, slang).
* **For multi-source training, keep `source` and evaluate per source.** The calibrated thresholds
  and soft labels (above) bring the label meanings closer, but FiQA posts are still a different
  genre. A source tag in the input lets the model learn the remaining difference.
* **For aspect categories,** collapse to the 4 top-level classes or the frequent second-level
  ones. The tail has too few examples to learn.
* **⚠ Benchmark leak.** FinBen's sentence-level FiQA-SA task (`ChanceFocus/flare-fiqasa`) contains
  the same 1,173 rows split differently:
  * 175 of its 233 test sentences are in `en-fiqa-absa` **train**.
  * 147 of `en-fiqa-absa`'s 230 test sentences are in its train split.

  Never train on one and report on the other's test split. If you compare against published
  FinBen FiQA-SA numbers, train on nothing from FiQA except FiQA-SA's own train split.
* The official splits contain 3 tweets that also appear in another split, differing only in
  their t.co link. They are flagged `near_duplicate_across_splits`; drop them from valid/test
  for a clean estimate.

## What was wrong in the raw data (all fixed or flagged by the pipeline)

**SEntFiN**
* 19 entity names corrupted by a global find-and-replace of trading calls: "Hold" became
  "neutral" and "sell" became "negative" (`Bajaj neutralings`, `FTSE Rusnegative`).
  Repaired.
* 8 truncated entities, extended to the full word of the title (`Alibab`→`Alibaba`,
  `Karur Vysya Ban`→`Karur Vysya Bank`, `B`→`Brent`, `G`→`Gold`, ...).
* 13 `Decisions` cells that are not valid Python (unescaped apostrophe, e.g. `'Moody's'`).
  Parsed with a tolerant fallback; a plain `ast.literal_eval` would crash or silently lose
  these rows.
* 67 duplicate headlines, merged. Labels that conflict between copies were dropped (7 entities).
* Whitespace noise: 86 titles and 4 entities. The `Words` column was dropped (it is a stale
  token count).
* Naive substring matching mislocates targets (`IT` inside "Nitin", `Jet` inside "SpiceJet").
  Spans are aligned with word boundaries, longest target first.

**FiQA**
* 15 sentences with mojibake (`Â£100m`, `BÃ¶rse`), repaired. 22 headlines with a scraped
  section label glued to the first word (`CompaniesTesco…`, `News FeedSchroders…`), removed.
* 14 misspelled targets, corrected by fuzzy matching against the sentence; the original is kept
  in `target_raw`. Examples: `Perssimon`, `Tailor Wimpey`, `AB InBrev`, `SAB Miller`.
* 6 targets that do not occur in the sentence, flagged `target_not_in_text`. Three are aliases
  ("Standard Chartered" for "StanChart", "IBM" for "International Business Machines", SABMiller
  for "SAB's"), one is implicit ("Berkshire Hathaway" for "Buffett's Company"), and two look
  wrong ("Morrissons" on a Smiths Group headline, "CRM" on an `$XLI` post). Drop them for
  span-based models.
* Aspect labels with typos (`Market/Volatilit`, `Behaviour`/`Behavior`, `downgrade`, trailing
  `/`, `/?`), plus one value that is a stringified list. Normalized.

## Suggested protocol

1. Fine-tune on `data/processed/sentfin/*_train` (or `combined/`), select on `*_valid`. Train
   encoders against `soft_label`; weight or filter LLM SFT records by `label_confidence`.
   For aspect + sentiment, use
   `instruct_aspect_*` / `instruct_e2e_aspect_*` (LLM SFT), or `pairs_*` with the `aspect` column
   (encoder with two heads).
2. Report accuracy and macro-F1 on:
   * SEntFiN test: all pairs, unseen-entity pairs, and pairs in multi-target or mixed-polarity
     headlines (`n_targets > 1`, `is_mixed`);
   * FiQA test, as out-of-domain, split into headlines and posts. Also report it on
     `polarity_finben` when comparing with published FiQA numbers;
   * for aspects: silver test labels (agreement with the LLM), plus FiQA's original gold aspects
     (`fiqa_aspect`) as the only human reference.
3. Optional: add FiQA train (the `combined/` export), then measure the in-domain gain on FiQA
   test and any regression on SEntFiN test.
4. Use the trivial baselines in `data_profile.md` as the floor any model must clearly beat,
   especially the entity-prior baseline.
