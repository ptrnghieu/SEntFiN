# Encoder comparison: single-task baselines (test split)

Kaggle T4, one seed per run, 2026-11-28. Same data, splits and recipe for every encoder (transformers: lr 2e-5 / 3e-5; BiLSTM: lr 1e-3, more epochs). Bold = best in the row. Per-run details: `runs/<task>/<model>/<run>/metrics.md` (`runs/` is not committed).

| step | metric | FinBERT-pretrain | ProsusAI FinBERT | FinBERT-tone | RoBERTa-base | BiLSTM-GloVe |
|---|---|---|---|---|---|---|
| 1a | span F1, train FinEntity | 0.859 | 0.907 | 0.881 | **0.944** | 0.793 |
| 1a | span F1, + LLM-completed SEntFiN/FiQA | 0.887 | 0.924 | 0.885 | **0.970** | 0.777 |
| 1a | joint span+sentiment macro-F1 (FinEntity paper: 0.85) | 0.735 | 0.815 | 0.772 | **0.863** | 0.651 |
| 1b | SEntFiN macro-F1 (train SEntFiN) | 0.874 | **0.894** | 0.867 | 0.893 | 0.836 |
| 1b | SEntFiN unseen entities (n=194) | 0.826 | **0.865** | 0.850 | 0.863 | 0.802 |
| 1b | SEntFiN unseen, combined soft-cw | 0.858 | **0.865** | 0.834 | 0.835 | 0.790 |
| 1b | FiQA macro-F1, calibrated classes | 0.649 | 0.652 | 0.644 | **0.687** | 0.579 |
| 1b | FiQA accuracy, FinBen ±0.1 classes | 0.791 | 0.829 | 0.821 | **0.850** | 0.611 |
| 1b | FiQA macro-F1, FinBen ±0.1 (12 neutral) | **0.636** | 0.610 | 0.601 | 0.580 | 0.368 |
| 1b | FinEntity macro-F1 | 0.865 | 0.907 | 0.856 | **0.934** | 0.829 |
| 1b | all sources, combined soft-cw | 0.848 | 0.866 | 0.847 | **0.882** | 0.810 |
| 1c | aspect level-2 macro-F1 (LLM labels) | 0.788 | 0.800 | 0.785 | **0.832** | 0.772 |
| 1c | aspect level-1 macro-F1 | 0.896 | 0.895 | 0.899 | **0.916** | 0.874 |
| 1c | FiQA human aspects accuracy (LLM ceiling 0.848) | 0.652 | 0.636 | 0.686 | **0.763** | 0.678 |

## Findings

1. **RoBERTa-base is best on nearly every metric**, by the widest margin on target extraction (span F1 0.944 vs 0.859 for FinBERT-pretrain, 0.970 vs 0.887 with the LLM-completed targets). Its joint span+sentiment macro-F1 (0.863) reaches the FinEntity paper's FinBERT-CRF (0.85), which our FinBERT run did not (0.735).
2. **Likely cause: casing.** All three FinBERTs and the GloVe BiLSTM are uncased (`Apple` -> `apple`, `BIDV` -> `bid ##v`); RoBERTa keeps case, a strong cue for company names. Financial pretraining does not make up for it here. To separate the two effects, add a cased general BERT (`bert-base-cased`).
3. **Sentiment given the target:** RoBERTa and ProsusAI FinBERT lead (SEntFiN 0.89, FinEntity 0.91-0.93); the sentence-sentiment fine-tuning of ProsusAI helps a little over FinBERT-pretrain (+2 on SEntFiN, +4 on FinEntity).
4. **The earlier "combined training shrinks the unseen-entity gap" finding does not hold up:** it improved FinBERT-pretrain (0.826 -> 0.858) but hurt RoBERTa (0.863 -> 0.835) and left ProsusAI unchanged, on 194 pairs. Treat it as noise until several seeds say otherwise.
5. **FiQA under FinBen's classes:** RoBERTa has the best accuracy (0.850) but the worst transformer macro-F1, because it never predicts the 12-example neutral class (F1 0.0). Macro-F1 on this test set is dominated by those 12 pairs; report accuracy and per-class F1 alongside it.
6. **BiLSTM-GloVe** trails by 4-8 points on classification and ~8-10 on extraction; it collapses on FiQA-FinBen (macro 0.37), where 822 training pairs are too few without pretraining.
7. **Aspects:** RoBERTa again best (level-2 0.832, FiQA human aspects 0.763 accuracy vs the LLM labels' own 0.848).

**Choice:** RoBERTa-base as the main encoder for the multi-task model, FinBERT-pretrain kept as the domain baseline. One seed only: gaps under ~2-3 points are within noise.
