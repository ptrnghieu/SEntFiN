#!/usr/bin/env bash
# The baseline matrix (scripts/run_baselines.sh) for several encoders, one after the other.
# Results: runs/<task>/<model>/<run>/; comparison table: runs/SUMMARY.md.
# Checkpoints (~440 MB per run) are deleted after each model unless KEEP_CHECKPOINTS=1:
# 5 models x 11 runs would not fit in Kaggle's 20 GB output.
set -euo pipefail
MODELS=${MODELS:-"yiyanghkust/finbert-pretrain ProsusAI/finbert yiyanghkust/finbert-tone roberta-base lstm"}
for M in $MODELS; do
  echo "=== $M"
  MODEL=$M bash scripts/run_baselines.sh
  # Pipeline triplets 1a -> 1b -> 1c from this model's runs (needs the checkpoints, so before deleting them)
  T=$(python -c "from absa_models.common import model_tag; print(model_tag('$M'))")
  python -m absa_models.pipeline --out runs/pipeline/$T \
    --extractor runs/target_extraction/$T/bio-crf-finentity+sentfin+fiqa \
    --sentiment runs/sentiment/$T/sentfin+fiqa+finentity-soft-cw \
    --aspect runs/aspect/$T/sentfin+fiqa+finentity
  # Direction 1: the multi-task triplet model with the same encoder
  python -m absa_models.multitask --model $M --soft
  [ "${KEEP_CHECKPOINTS:-0}" = 1 ] || find runs -name best.pt -delete
done
python -m absa_models.summarize runs
