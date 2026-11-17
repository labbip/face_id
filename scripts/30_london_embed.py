"""Image embeddings of the 102 London Set neutral-front faces (CLIP ViT-B/32 laion2b, DINOv2 ViT-S/14).
Face order follows the rating columns (X001..). Outputs results/london_embed_{clip,dino}.npy.
Run: uv run --with open_clip_torch --with timm python -I scripts/30_london_embed.py
"""
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
r = pd.read_csv("data/raw/london/london_faces_ratings.csv", nrows=1); ids = [c[1:] for c in r.columns[3:]]
paths = [f"data/raw/london/neutral_front/neutral_front/{i}_03.jpg" for i in ids]
imgs = [Image.open(p).convert("RGB") for p in paths]; dev = "cuda"
m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m = m.to(dev).eval()
with torch.no_grad():
    x = torch.stack([pre(im) for im in imgs]).to(dev); e = m.encode_image(x).float().cpu().numpy()
np.save("results/london_embed_clip.npy", e); print("clip", e.shape)
d = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
import torchvision.transforms as T
tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
with torch.no_grad():
    x = torch.stack([tf(im) for im in imgs]).to(dev); e2 = d(x).float().cpu().numpy()
np.save("results/london_embed_dino.npy", e2); print("dino", e2.shape)
# sanity: linear read-out of population mean attractiveness and face gender from each embedding (LOO-CV ridge)
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
R = pd.read_csv("data/raw/london/london_faces_ratings.csv").iloc[:, 3:].to_numpy(float); pop = R.mean(0)
info = pd.read_csv("data/raw/london/london_faces_info.csv", dtype={"face_id": str}); fg = (info.face_gender == "male").to_numpy()
for name, E in (("clip", e), ("dino", e2)):
    E = (E - E.mean(0)) / E.std(0)
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(0, 5, 11)), E, pop, cv=10)
    acc = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), E, fg, cv=10).mean()
    print(f"{name}: pop attractiveness r (10-fold ridge) {np.corrcoef(pr, pop)[0,1]:.3f}; face gender acc {acc:.3f}")
