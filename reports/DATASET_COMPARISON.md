# SEntFiN 1.0 vs FiQA-ABSA: which one to train on

All numbers come from `reports/data_profile.md`, which `python -m absa_data build` regenerates.
They are counted after cleaning (see "What was wrong in the raw data" below).

## Recommendation

**Use SEntFiN as the main training and evaluation source. Use FiQA as an out-of-domain
test set, and as optional second-stage data if your target domain includes US/UK stock tweets.**

Choose FiQA as the main source only if the project needs one of the two things only FiQA has:
**aspect categories** (Stock/Price Action, Corporate/M&A, ...) or a **continuous sentiment score**.
Even then, expect a small and noisy training set.

## Side by side

| | SEntFiN 1.0 | FiQA 2018 Task 1 (`TheFinAI/en-fiqa-absa`) |
|---|---|---|
| Texts / (text, target) pairs | **10,679 / 14,318** | 1,111 / 1,173 |
| Text | Indian financial news headlines | 436 UK/US headlines + 675 StockTwits posts |
| Annotation | entity → negative / neutral / positive | target → score in [-1, 1] + aspect category path |
| 3-class labels | annotated by humans | derived from the score (FinBen rule: neg < -0.1 ≤ neu < 0.1 ≤ pos) |
| Label mix neg / neu / pos | 26.5 / 38.2 / 35.3 % | 30.9 / **7.8** / 61.3 % |
| Texts with >1 target | **2,836 (26.6 %)** | 57 (5.1 %) |
| Texts whose targets disagree in polarity | **1,228 (11.5 %)** | 11 (1.0 %); **0 in test** |
| Official split | none (this repo ships a frozen, leakage-safe 80/10/10) | 822 / 117 / 234 pairs |
| Test pairs / 95 % CI on an accuracy near 80 % | 1,433 / ±2.1 pp | 234 / ±5.1 pp |
| Neutral pairs in test | 547 | **12** (±28 pp on that class) |
| Aspect categories | no | 4 top-level, 28 second-level (3 never in train, 5 with < 5 train examples) |
| Access | MIT, in this repo | MIT, gated (access form, cite FinBen) |

## Why SEntFiN should be the main source

1. **It is 12 times larger.** 14.3k labelled pairs, against 1.2k.
2. **It tests what makes ABSA harder than plain sentiment classification.** In 1,228 headlines,
   different entities get different polarities ("Gold shines on seasonal demand; Silver dull").
   FiQA's test split has no such sentence, and 98 % of its test sentences have a single target.
   A model that ignores the target entirely therefore loses almost nothing on FiQA test.
3. **Its labels are balanced and its neutral class is real.** FiQA's neutral class is 7.8 % of
   the data and only 12 test pairs. Its macro-F1 therefore swings by several points with
   2–3 predictions.
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
* **For multi-source training, keep `source` and evaluate per source.** The label semantics
  differ: FiQA "neutral" only means |score| < 0.1, while SEntFiN "neutral" is an annotator's
  judgement. A source tag in the input is a cheap way to let the model learn the difference.
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

1. Fine-tune on `data/processed/sentfin/*_train`, select on `*_valid`.
2. Report accuracy and macro-F1 on:
   * SEntFiN test: all pairs, unseen-entity pairs, and pairs in multi-target or mixed-polarity
     headlines (`n_targets > 1`, `is_mixed`);
   * FiQA test, as out-of-domain, split into headlines and posts.
3. Optional: add FiQA train (the `combined/` export), then measure the in-domain gain on FiQA
   test and any regression on SEntFiN test.
4. Use the trivial baselines in `data_profile.md` as the floor any model must clearly beat,
   especially the entity-prior baseline.
