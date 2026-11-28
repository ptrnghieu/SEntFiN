#!/usr/bin/env bash
# Ablation of the multi-task improvements, one encoder (default RoBERTa-base), several seeds.
# Each variant adds ONE option to the base model (--soft), plus one with all five:
#   base   --soft
#   hier   (8)  two-level aspect head               --aspect-head hier
#   sent   (10) sentence vector for the aspect head --sentence-context
#   conf   (11) aspect labels weighted by LLM confidence --aspect-confidence
#   ord    (5)  ordinal sentiment head              --sentiment-loss ordinal
#   mark   (12) levitated markers                   --markers
#   all    all five
# Results: runs/ablation/multitask/<variant>-s<seed>/; mean ± sd: runs/ablation/SEEDS.md.
# About 16-19 min per run for RoBERTa on a T4: 7 variants x 3 seeds is about 6 h.
# Fewer: VARIANTS="base mark all" SEEDS="42 43" bash scripts/run_ablation.sh
set -euo pipefail
M=${MODEL:-roberta-base}
SEEDS=${SEEDS:-"42 43 44"}
EPOCHS=${EPOCHS:-10}
VARIANTS=${VARIANTS:-"base hier sent conf ord mark all"}
R=runs/ablation
declare -A FLAGS=(
  [base]=""
  [hier]="--aspect-head hier"
  [sent]="--sentence-context"
  [conf]="--aspect-confidence"
  [ord]="--sentiment-loss ordinal"
  [mark]="--markers"
  [all]="--aspect-head hier --sentence-context --aspect-confidence --sentiment-loss ordinal --markers"
)
for S in $SEEDS; do
  for V in $VARIANTS; do
    echo "=== $M $V seed $S"
    # shellcheck disable=SC2086
    python -m absa_models.multitask --model $M --seed $S --soft --epochs $EPOCHS ${FLAGS[$V]} \
      --out $R/multitask/$V-s$S
    [ "${KEEP_CHECKPOINTS:-0}" = 1 ] || find $R -name best.pt -delete
  done
  python -m absa_models.aggregate $R   # partial table after every seed
done
