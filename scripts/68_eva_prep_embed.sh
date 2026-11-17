#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# EVA: download the 7-part image zip from GitHub (kang-gnak/eva-dataset, CC0), unzip, CLIP-embed, and write the long rating table.
set -e; cd "$ROOT"; mkdir -p data/raw/eva/images; cd data/raw/eva/images
for i in 1 2 3 4 5 6 7; do [ -s EVA_together.zip.00$i ] || curl -sL -o EVA_together.zip.00$i "https://raw.githubusercontent.com/kang-gnak/eva-dataset/master/images/EVA_together.zip.00$i"; done
ls -la; cat EVA_together.zip.001 EVA_together.zip.002 EVA_together.zip.003 EVA_together.zip.004 EVA_together.zip.005 EVA_together.zip.006 EVA_together.zip.007 > EVA_together.zip && unzip -q -o EVA_together.zip && rm EVA_together.zip && find . -name "*.jpg" | wc -l
cd "$ROOT" && uv run --with open_clip_torch --with timm python -I scripts/68b_eva_embed.py
