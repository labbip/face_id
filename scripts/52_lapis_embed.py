"""CLIP ViT-B/32 embeddings of the 11,723 LAPIS artworks. Row order = sorted image_id (saved to results/lapis_image_ids.npy).
Run: uv run --with open_clip_torch --with timm python -I scripts/52_lapis_embed.py
"""
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
d = pd.read_csv("data/raw/lapis/piaa.csv", low_memory=False)[["image_id", "path"]].drop_duplicates("image_id").sort_values("image_id")
ids = d.image_id.to_numpy(); paths = ("data/raw/lapis/hf/images/" + d.path).tolist(); dev = "cuda"
m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval()
out = []
with torch.no_grad():
    for s in range(0, len(paths), 256):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[s:s + 256]]).to(dev); out.append(m.encode_image(x).float().cpu().numpy())
E = np.concatenate(out); np.save("results/lapis_embed_clip.npy", E); np.save("results/lapis_image_ids.npy", ids); print(E.shape)
