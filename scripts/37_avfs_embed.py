"""CLIP ViT-B/32 embeddings of the FFHQ 128px thumbnails referenced by AVFS (results/avfs_face_ids.json).
Outputs results/avfs_embed_clip.npy (rows in the order of avfs_face_ids.json).
Run: uv run --with open_clip_torch --with timm python -I scripts/37_avfs_embed.py
"""
import json, zipfile, io, numpy as np, torch, open_clip
from PIL import Image
ids = json.load(open("results/avfs_face_ids.json")); dev = "cuda"
z = zipfile.ZipFile("data/raw/ffhq/thumbnails128x128.zip"); names = set(z.namelist())
def path(i):
    for cand in (f"thumbnails128x128/{i:05d}.png", f"thumbnails128x128/{i//1000:05d}/{i:05d}.png", f"{i:05d}.png"):
        if cand in names: return cand
    raise FileNotFoundError(i)
m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval()
out = []
with torch.no_grad():
    for s in range(0, len(ids), 256):
        x = torch.stack([pre(Image.open(io.BytesIO(z.read(path(i)))).convert("RGB")) for i in ids[s:s + 256]]).to(dev)
        out.append(m.encode_image(x).float().cpu().numpy())
E = np.concatenate(out); np.save("results/avfs_embed_clip.npy", E); print(E.shape, "first path", path(ids[0]))
