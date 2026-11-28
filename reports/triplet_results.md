# Triplet extraction: pipeline vs multi-task (test split)

Kaggle T4, 2026-11-28, one seed. Pipeline = three single-task models chained (1a BIO-CRF extractor trained on FinEntity + LLM-completed SEntFiN/FiQA -> 1b sentiment, soft labels + class weights -> 1c aspect). Multi-task = `absa_models/multitask.py` (one encoder, BIO-CRF + per-span sentiment and aspect heads, soft sentiment labels), same data, 10 epochs (BiLSTM 30). Exact-span micro F1 from `absa_models/triplets.py`. FinEntity targets and sentiment are human labels; aspects are LLM labels in every source; SEntFiN/FiQA targets are gold + LLM-completed.

## All test texts (1,439: FinEntity 139, SEntFiN 1,070, FiQA 230)

| encoder | system | span | span+sentiment | span+aspect | triplet | triplet (aspect level 1) |
|---|---|---|---|---|---|---|
| RoBERTa-base | pipeline | 0.919 | 0.789 | 0.815 | 0.709 | 0.745 |
| RoBERTa-base | multitask | 0.922 | 0.795 | 0.812 | 0.711 | 0.747 |
| BERT-base-cased | pipeline | 0.912 | 0.763 | 0.761 | 0.652 | 0.701 |
| BERT-base-cased | multitask | 0.916 | 0.772 | 0.780 | 0.672 | 0.716 |
| FinBERT-pretrain | pipeline | 0.897 | 0.749 | 0.756 | 0.646 | 0.693 |
| FinBERT-pretrain | multitask | 0.897 | 0.756 | 0.777 | 0.666 | 0.704 |
| BiLSTM-GloVe | pipeline | 0.855 | 0.678 | 0.712 | 0.587 | 0.623 |
| BiLSTM-GloVe | multitask | 0.852 | 0.695 | 0.712 | 0.600 | 0.639 |

## FinEntity (human targets and sentiment)

| encoder | system | span | span+sentiment | span+aspect | triplet | triplet (aspect level 1) |
|---|---|---|---|---|---|---|
| RoBERTa-base | pipeline | 0.970 | 0.904 | 0.839 | 0.792 | 0.825 |
| RoBERTa-base | multitask | 0.956 | 0.897 | 0.809 | 0.777 | 0.828 |
| BERT-base-cased | pipeline | 0.942 | 0.844 | 0.751 | 0.704 | 0.732 |
| BERT-base-cased | multitask | 0.937 | 0.845 | 0.761 | 0.710 | 0.761 |
| FinBERT-pretrain | pipeline | 0.887 | 0.771 | 0.716 | 0.656 | 0.716 |
| FinBERT-pretrain | multitask | 0.872 | 0.777 | 0.706 | 0.670 | 0.697 |
| BiLSTM-GloVe | pipeline | 0.777 | 0.644 | 0.583 | 0.535 | 0.564 |
| BiLSTM-GloVe | multitask | 0.782 | 0.680 | 0.570 | 0.556 | 0.584 |

## SEntFiN

| encoder | system | span | span+sentiment | span+aspect | triplet | triplet (aspect level 1) |
|---|---|---|---|---|---|---|
| RoBERTa-base | pipeline | 0.917 | 0.801 | 0.839 | 0.740 | 0.765 |
| RoBERTa-base | multitask | 0.920 | 0.799 | 0.835 | 0.729 | 0.752 |
| BERT-base-cased | pipeline | 0.914 | 0.787 | 0.811 | 0.703 | 0.739 |
| BERT-base-cased | multitask | 0.918 | 0.791 | 0.820 | 0.715 | 0.743 |
| FinBERT-pretrain | pipeline | 0.904 | 0.780 | 0.799 | 0.695 | 0.731 |
| FinBERT-pretrain | multitask | 0.907 | 0.778 | 0.816 | 0.704 | 0.733 |
| BiLSTM-GloVe | pipeline | 0.870 | 0.719 | 0.765 | 0.639 | 0.670 |
| BiLSTM-GloVe | multitask | 0.872 | 0.730 | 0.770 | 0.652 | 0.683 |

## FiQA

| encoder | system | span | span+sentiment | span+aspect | triplet | triplet (aspect level 1) |
|---|---|---|---|---|---|---|
| RoBERTa-base | pipeline | 0.903 | 0.683 | 0.706 | 0.545 | 0.627 |
| RoBERTa-base | multitask | 0.911 | 0.728 | 0.725 | 0.604 | 0.684 |
| BERT-base-cased | pipeline | 0.890 | 0.628 | 0.567 | 0.422 | 0.537 |
| BERT-base-cased | multitask | 0.896 | 0.658 | 0.636 | 0.486 | 0.591 |
| FinBERT-pretrain | pipeline | 0.876 | 0.613 | 0.604 | 0.444 | 0.527 |
| FinBERT-pretrain | multitask | 0.870 | 0.660 | 0.657 | 0.516 | 0.594 |
| BiLSTM-GloVe | pipeline | 0.830 | 0.533 | 0.567 | 0.408 | 0.469 |
| BiLSTM-GloVe | multitask | 0.811 | 0.562 | 0.551 | 0.414 | 0.497 |

## Findings

1. **Multi-task matches or beats the pipeline with one model instead of three.** Triplet F1 on all texts: RoBERTa 0.711 vs 0.709, BERT-cased 0.672 vs 0.652, FinBERT 0.666 vs 0.646, BiLSTM 0.600 vs 0.587. It runs the encoder once per sentence; the pipeline runs it 1 + 2 x (number of targets) times.
2. **The gain is largest on FiQA** (triplet +6 to +7 points for the transformers), whose short posts give the pipeline's classifiers little to work with on their own; on FinEntity the RoBERTa pipeline stays ahead (0.792 vs 0.777).
3. **The CPU smoke test's worry did not hold:** on the GPU the span heads are not behind on sentiment (all texts, span+sentiment: multi-task >= pipeline for every encoder). No target-marker change is needed for now.
4. **The multi-task model is still learning at epoch 10** (best valid epoch 10 for BERT-cased and FinBERT, 9 for RoBERTa): more epochs are the cheapest next gain.
5. **Casing explains most of the FinBERT gap on extraction:** BERT-base-cased reaches span F1 0.942 vs FinBERT-pretrain 0.887 and RoBERTa 0.970. Financial pretraining does not show on extraction; on aspects FinBERT (0.788) edges BERT-cased (0.779).
6. **Noise:** rerunning identical configs reproduces FinBERT and BiLSTM to the digit but not RoBERTa (aspect level-2 macro-F1 0.832 vs 0.851, sentiment 0.882 vs 0.876). Treat differences under ~2 points as noise until several seeds are run.

**Best system so far:** RoBERTa-base multi-task, triplet F1 0.711 on all test texts (0.777 on FinEntity, 0.828 with level-1 aspects).
