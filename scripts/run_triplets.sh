#!/usr/bin/env bash
# Triplet systems only, per encoder: the three runs the pipeline chains (1a extractor, 1b sentiment,
# 1c aspect), the pipeline itself, then the multi-task model. A lighter follow-up to run_models.sh
# when the single-task tables already exist (about 30-35 min per transformer on a T4).
set -euo pipefail
MODELS=${MODELS:-"yiyanghkust/finbert-pretrain ProsusAI/finbert yiyanghkust/finbert-tone roberta-base lstm"}
for M in $MODELS; do
  echo "=== $M"
  T=$(python -c "from absa_models.common import model_tag; print(model_tag('$M'))")
  python -m absa_models.target_extraction --model $M --train-sources finentity sentfin fiqa --crf
  python -m absa_models.pair_classify --model $M --task sentiment --soft --class-weights
  python -m absa_models.pair_classify --model $M --task aspect
  python -m absa_models.pipeline --out runs/pipeline/$T \
    --extractor runs/target_extraction/$T/bio-crf-finentity+sentfin+fiqa \
    --sentiment runs/sentiment/$T/sentfin+fiqa+finentity-soft-cw \
    --aspect runs/aspect/$T/sentfin+fiqa+finentity
  python -m absa_models.multitask --model $M --soft
  [ "${KEEP_CHECKPOINTS:-0}" = 1 ] || find runs -name best.pt -delete
done
python -m absa_models.summarize runs
