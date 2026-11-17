#!/bin/bash
# Sensitivity to embedding dimension D and latent dimension M (1 seed, 10 folds); plus non-negative (SPoSE-like) embedding.
cd "$(dirname "$0")/.."
jobs=$(mktemp)
for f in 0 1 2 3 4 5 6 7 8 9; do
  for D in 5 16; do
    echo "uv run python scripts/04_train.py --exp 2 --kind D1 --D $D --M 4 --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --tag _sensD > logs/sens_exp2_D1_D${D}_f${f}.log 2>&1"
    echo "uv run python scripts/04_train.py --exp 1 --kind D2 --D $D --M 4 --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --tag _sensD > logs/sens_exp1_D2_D${D}_f${f}.log 2>&1"
  done
  for M in 2 8; do
    echo "uv run python scripts/04_train.py --exp 2 --kind D1 --D 8 --M $M --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --tag _sensM > logs/sens_exp2_D1_M${M}_f${f}.log 2>&1"
    echo "uv run python scripts/04_train.py --exp 1 --kind D2 --D 8 --M $M --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --tag _sensM > logs/sens_exp1_D2_M${M}_f${f}.log 2>&1"
  done
  echo "uv run python scripts/04_train.py --exp 2 --kind D1 --D 8 --M 4 --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --nonneg --tag _nonneg > logs/nonneg_exp2_D1_f${f}.log 2>&1"
  echo "uv run python scripts/04_train.py --exp 1 --kind D2 --D 8 --M 4 --fold $f --seed 0 --epochs 100 --batch 32 --lr 5e-3 --init_mds --det --nonneg --tag _nonneg > logs/nonneg_exp1_D2_f${f}.log 2>&1"
done > $jobs
echo "$(wc -l < $jobs) jobs"; cat $jobs | xargs -P 6 -I{} bash -c "{}"; echo DONE
