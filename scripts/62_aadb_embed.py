"""CLIP ViT-B/32 (laion2b) embeddings of the AADB photographs (data/raw/aadb/images/datasetImages_warp256/, 256x256 warped release).
Row order = sorted image_id of data/processed/aadb_ratings.csv that exist on disk (saved to results/aadb_image_ids.npy).
Run: uv run --with open_clip_torch --with timm python -I scripts/62_aadb_embed.py
"""
import os, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
root = "data/raw/aadb/images/datasetImages_warp256/"
ids = np.array(sorted(i for i in pd.read_csv("data/processed/aadb_ratings.csv").image_id.unique() if os.path.exists(root + i))); print("images with files:", len(ids))
dev = "cuda"; m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval(); out = []
with torch.no_grad():
    for s in range(0, len(ids), 256):
        x = torch.stack([pre(Image.open(root + p).convert("RGB")) for p in ids[s:s + 256]]).to(dev); out.append(m.encode_image(x).float().cpu().numpy())
E = np.concatenate(out); np.save("results/aadb_embed_clip.npy", E); np.save("results/aadb_image_ids.npy", ids); print(E.shape)
