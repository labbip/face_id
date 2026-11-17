"""EVA: long rating table data/processed/eva_ratings.csv (participant_id, image_id, rating; votes_filtered) and CLIP ViT-B/32 embeddings of
the images under data/raw/eva/images/ (results/eva_embed_clip.npy, results/eva_image_ids.npy; ids = image_id as string)."""
import glob, os, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
v = pd.read_csv("data/raw/eva/votes_filtered.csv", sep="=").rename(columns={"user_id": "participant_id", "score": "rating"}); v["image_id"] = v.image_id.astype(str)
v[["participant_id", "image_id", "rating"]].to_csv("data/processed/eva_ratings.csv", index=False)
files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("data/raw/eva/images/**/*.jpg", recursive=True)}
ids = np.array(sorted(i for i in v.image_id.unique() if i in files)); print("images with files", len(ids), "of", v.image_id.nunique())
dev = "cuda"; m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval(); out = []
with torch.no_grad():
    for s in range(0, len(ids), 256):
        x = torch.stack([pre(Image.open(files[p]).convert("RGB")) for p in ids[s:s + 256]]).to(dev); out.append(m.encode_image(x).float().cpu().numpy())
E = np.concatenate(out); np.save("results/eva_embed_clip.npy", E); np.save("results/eva_image_ids.npy", ids); print(E.shape)
