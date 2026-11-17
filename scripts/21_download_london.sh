#!/usr/bin/env bash
# Face Research Lab London Set (DeBruine & Jones, figshare 5047666, CC BY 4.0)
# ratings + info + neutral front images only.
set -e
D=data/raw/london
curl -sL -o $D/london_faces_ratings.csv https://ndownloader.figshare.com/files/8542045
curl -sL -o $D/london_faces_info.csv    https://ndownloader.figshare.com/files/27397184
curl -sL -o $D/facelab_london.pdf       https://ndownloader.figshare.com/files/8541955
curl -sL -o $D/neutral_front.zip        https://ndownloader.figshare.com/files/8541961
ls -l $D
