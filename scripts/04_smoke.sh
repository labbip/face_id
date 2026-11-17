#!/bin/bash
cd "$(dirname "$0")/.."
for k in D0 D1; do
  uv run python scripts/04_train.py --exp 1 --kind $k --fold 0 --seed 0 --epochs 30 --tag _smoke 2>&1 | grep -v Warning
done
