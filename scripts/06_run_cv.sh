#!/bin/bash
# Full participant-CV runs. usage: scripts/06_run_cv.sh <exp> <kinds...>  (env: SEEDS, FOLDS, EXTRA)
cd "$(dirname "$0")/.."
EXP=$1; shift
SEEDS=${SEEDS:-"0 1 2"}; FOLDS=${FOLDS:-"0 1 2 3 4 5 6 7 8 9"}; EXTRA=${EXTRA:-"--epochs 100 --batch 32 --lr 5e-3 --init_mds --direct --det"}; TAG=${TAG:-_cv}
NPAR=${NPAR:-4}
jobs_file=$(mktemp)
for k in "$@"; do for s in $SEEDS; do for f in $FOLDS; do
  echo "uv run python scripts/04_train.py --exp $EXP --kind $k --fold $f --seed $s $EXTRA --tag $TAG > logs/cv_exp${EXP}_${k}_f${f}_s${s}${TAG}.log 2>&1" >> $jobs_file
done; done; done
echo "$(wc -l < $jobs_file) jobs, $NPAR parallel"
cat $jobs_file | xargs -P $NPAR -I{} bash -c "{}"
echo DONE
