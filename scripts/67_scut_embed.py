"""CLIP ViT-B/32 (laion2b) embeddings of the 5,500 SCUT-FBP5500 faces (data/raw/scut/images/SCUT-FBP5500_v2/Images/*.jpg).
Row order = sorted Filename of All_Ratings.xlsx (saved to results/scut_image_ids.npy). Run: uv run --with open_clip_torch --with timm --with openpyxl python -I scripts/67_scut_embed.py
"""
import os, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
root = "data/raw/scut/images/SCUT-FBP5500_v2/Images/"
ids = np.array(sorted(pd.read_excel("data/raw/scut/unzip/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL").Filename.unique())); ids = np.array([i for i in ids if os.path.exists(root + i)]); print("images", len(ids))
dev = "cuda"; m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval(); out = []
with torch.no_grad():
    for s in range(0, len(ids), 256):
        x = torch.stack([pre(Image.open(root + p).convert("RGB")) for p in ids[s:s + 256]]).to(dev); out.append(m.encode_image(x).float().cpu().numpy())
E = np.concatenate(out); np.save("results/scut_embed_clip.npy", E); np.save("results/scut_image_ids.npy", ids); print(E.shape)
