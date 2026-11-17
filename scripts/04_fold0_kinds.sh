#!/bin/bash
# fold-0 comparison of decoder hierarchy with the tuned optimisation settings
cd "$(dirname "$0")/.."
for k in D1 D2 D3; do
  uv run python scripts/04_train.py --exp 1 --kind $k --fold 0 --seed 0 --epochs 100 --batch 32 --lr 5e-3 --tag _smoke2 2>&1 | grep -v Warning
done
