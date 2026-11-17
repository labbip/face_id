#!/bin/bash
cd "$(dirname "$0")/.."
for exp in 1 2; do for k in D1 D3; do for s in 0 1; do for f in 0 1 2 3 4 5 6 7 8 9; do
  echo "uv run python scripts/04_train.py --exp $exp --kind $k --M 16 --fold $f --seed $s --epochs 100 --batch 32 --lr 5e-3 --init_mds --direct --det --tag _cv > logs/cv_exp${exp}_${k}_M16_f${f}_s${s}_cv.log 2>&1"
done; done; done; done | xargs -P 3 -I{} bash -c "{}"
echo DONE
