#!/usr/bin/env bash
# Steps 1a-1c: single-task baselines for one encoder (MODEL=..., default FinBERT; MODEL=lstm for BiLSTM).
# Prerequisite: python -m absa_data build   (needs Hugging Face access to FiQA)
set -euo pipefail
M=${MODEL:-yiyanghkust/finbert-pretrain}
TE="python -m absa_models.target_extraction --model $M"
PC="python -m absa_models.pair_classify --model $M"

# 1a target extraction
$TE --train-sources finentity --scheme bio-sentiment --crf              # reproduce FinEntity FinBERT-CRF (0.85)
$TE --train-sources finentity --crf                                     # extraction head, FinEntity only
$TE --train-sources finentity sentfin fiqa --crf                        # + LLM-completed SEntFiN/FiQA

# 1b sentiment given the target
$PC --task sentiment --train-sources sentfin --eval-sources sentfin     # compare with SEntFiN paper (~94% acc)
$PC --task sentiment --train-sources fiqa --eval-sources fiqa
$PC --task sentiment --train-sources fiqa_finben --eval-sources fiqa_finben   # FinBen's +-0.1 classes
$PC --task sentiment --train-sources finentity --eval-sources finentity
$PC --task sentiment --gold-only                                        # all sources, human sentiment only
$PC --task sentiment --soft --class-weights                             # all sources + silver, soft labels

# 1c aspect given the target
$PC --task aspect
$PC --task aspect --class-weights

python -m absa_models.summarize runs
