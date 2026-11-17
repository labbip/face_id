"""CLIP ViT-B/32 (laion2b) embeddings of the XPASS-Vis fashion (CCP) and scenery (Sekai, one frame per clip) images, same model as 52/62.
Row order = sorted image_id (file name) of data/processed/xpass_{fashion,scenery}.csv -> results/xpass_<g>_image_ids.npy, results/xpass_<g>_embed_clip.npy
Run: uv run --with open_clip_torch --with timm python -I scripts/74b_xpass_embed.py
"""
import os, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
dev = "cuda"; m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval()
for g, sub in (("fashion", "fashion"), ("scenery", "scenery_image")):
    root = f"data/raw/xpass/data/samples/{sub}/"; ids = np.array(sorted(i for i in pd.read_csv(f"data/processed/xpass_{g}.csv").image_id.unique() if os.path.exists(root + i))); out = []
    with torch.no_grad():
        for s in range(0, len(ids), 128):
            x = torch.stack([pre(Image.open(root + p).convert("RGB")) for p in ids[s:s + 128]]).to(dev); out.append(m.encode_image(x).float().cpu().numpy())
    E = np.concatenate(out); np.save(f"results/xpass_{g}_embed_clip.npy", E); np.save(f"results/xpass_{g}_image_ids.npy", ids); print(g, "images with files:", len(ids), E.shape, flush=True)
