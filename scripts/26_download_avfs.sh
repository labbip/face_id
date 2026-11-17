#!/usr/bin/env bash
# AVFS dataset (Andrews, Joniak & Xiang, ICLR 2023; Zenodo 7878655; CC BY-NC-SA 4.0). Judgments only, no images.
set -e
D=data/raw/avfs
curl -sL -o $D/avfs-dataset-v1.zip "https://zenodo.org/api/records/7878655/files/avfs-dataset-v1.zip/content"
curl -sL -o $D/AVFS_README.md https://raw.githubusercontent.com/SonyResearch/a_view_from_somewhere/main/docs/AVFS_README.md
unzip -o -q $D/avfs-dataset-v1.zip -d $D/unzip
find $D/unzip -type f | head -30
