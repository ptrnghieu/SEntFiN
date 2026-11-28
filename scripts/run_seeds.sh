#!/usr/bin/env bash
# Pipeline vs multi-task over several seeds, one encoder (default RoBERTa-base).
# Per seed:
#   pipeline        1a extractor -> 1b sentiment (soft, class weights) -> 1c aspect; the pair
#                   classifiers get 10 epochs (patience 3), since 5 often stopped while improving
#   multitask e10   the multi-task model, 10 epochs (as in run_triplets.sh)
#   multitask e20   the same with 20 epochs, patience 4: does longer training help?
# Results: runs/seeds/<system>/<variant>-s<seed>/; mean ± sd over seeds: runs/seeds/SEEDS.md.
# About 75-80 min per seed for RoBERTa on a T4. Checkpoints are deleted after each seed.
set -euo pipefail
M=${MODEL:-roberta-base}
SEEDS=${SEEDS:-"42 43 44"}
R=runs/seeds
for S in $SEEDS; do
  echo "=== $M seed $S"
  python -m absa_models.target_extraction --model $M --seed $S --train-sources finentity sentfin fiqa --crf \
    --out $R/extractor/s$S
  python -m absa_models.pair_classify --model $M --seed $S --task sentiment --soft --class-weights \
    --epochs 10 --patience 3 --out $R/sentiment/s$S
  python -m absa_models.pair_classify --model $M --seed $S --task aspect \
    --epochs 10 --patience 3 --out $R/aspect/s$S
  python -m absa_models.pipeline --extractor $R/extractor/s$S --sentiment $R/sentiment/s$S \
    --aspect $R/aspect/s$S --out $R/pipeline/s$S
  python -m absa_models.multitask --model $M --seed $S --soft --out $R/multitask/e10-s$S
  python -m absa_models.multitask --model $M --seed $S --soft --epochs 20 --patience 4 --out $R/multitask/e20-s$S
  [ "${KEEP_CHECKPOINTS:-0}" = 1 ] || find $R -name best.pt -delete
  python -m absa_models.aggregate $R   # partial table after every seed, in case the session is cut
done
