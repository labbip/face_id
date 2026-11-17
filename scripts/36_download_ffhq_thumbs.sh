#!/usr/bin/env bash
# FFHQ 128x128 thumbnails (NVIDIA, CC BY-NC-SA 4.0 for the dataset; individual image licences in FFHQ metadata),
# Hugging Face mirror nuwandaa/ffhq128 of the official thumbnails128x128.zip. Only the 4,977 AVFS faces are extracted.
set -e
D=data/raw/ffhq
curl -L -s -o $D/thumbnails128x128.zip "https://huggingface.co/datasets/nuwandaa/ffhq128/resolve/main/thumbnails128x128.zip"
ls -l $D/thumbnails128x128.zip
unzip -l $D/thumbnails128x128.zip | tail -1
