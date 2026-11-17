#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # repository root (2026-11-17: was an absolute path)
# LAPIS images (KU Leuven, CC BY-NC 4.0) from Hugging Face asmaerten/LAPIS -> data/raw/lapis/hf/images/
cd "$ROOT"
uv run --with huggingface_hub python -I -c "
from huggingface_hub import snapshot_download
p = snapshot_download('asmaerten/LAPIS', repo_type='dataset', allow_patterns=['images/*', 'giaa/*', 'image_attributes/*', 'README.md'], local_dir='data/raw/lapis/hf')
print(p)"
find data/raw/lapis/hf/images -name "*.jpg" | wc -l
