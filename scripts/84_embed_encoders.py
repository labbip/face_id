"""Encoder sensitivity (2026-11-12): image embeddings from encoders other than CLIP ViT-B/32 for the image-version datasets.
Encoders: dinov2b = DINOv2 ViT-B/14 (self-supervised, CLS token, 768 dims; timm vit_base_patch14_dinov2.lvd142m, resize 224 + centre crop as in 30);
          clipl   = CLIP ViT-L/14 (open_clip, laion2b_s32b_b82k, 768 dims; same family as the main encoder, larger).
Row order = the existing results/<ds>_image_ids.npy (london: rating-column order), so the fold splits are unchanged.
Outputs results/<ds>_embed_<enc>.npy.  Existing *_embed_clip.npy are not touched.
Run: uv run --with open_clip_torch --with timm python -I scripts/84_embed_encoders.py [lapis eva aadb london]
"""
import glob, os, sys, numpy as np, pandas as pd, torch, open_clip
import torchvision.transforms as T
from PIL import Image
dev = "cuda"; DSS = sys.argv[1:] or ["lapis", "eva", "aadb", "london"]
def paths_of(ds):
    if ds == "london":
        r = pd.read_csv("data/raw/london/london_faces_ratings.csv", nrows=1); return [f"data/raw/london/neutral_front/neutral_front/{c[1:]}_03.jpg" for c in r.columns[3:]]
    ids = np.load(f"results/{ds}_image_ids.npy", allow_pickle=True)
    if ds == "lapis":
        d = pd.read_csv("data/raw/lapis/piaa.csv", low_memory=False)[["image_id", "path"]].drop_duplicates("image_id").set_index("image_id").path; return ["data/raw/lapis/hf/images/" + d[i] for i in ids]
    if ds == "eva":
        files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("data/raw/eva/images/**/*.jpg", recursive=True)}; return [files[str(i)] for i in ids]
    if ds == "aadb": return ["data/raw/aadb/images/datasetImages_warp256/" + str(i) for i in ids]
ENC = {}
m, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k"); ENC["clipl"] = (m.to(dev).eval().encode_image, pre)
import timm   # same LVD-142M weights as torch.hub dinov2_vitb14, taken from the Hugging Face hub (the fbaipublicfiles download failed certificate verification on 2026-11-12)
d = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=0).to(dev).eval()   # num_classes=0 -> CLS token
ENC["dinov2b"] = (d, T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
for ds in DSS:
    paths = paths_of(ds); out = {k: [] for k in ENC}
    with torch.no_grad():
        for s in range(0, len(paths), 64):
            ims = [Image.open(p).convert("RGB") for p in paths[s:s + 64]]
            for k, (fn, tf) in ENC.items(): out[k].append(fn(torch.stack([tf(im) for im in ims]).to(dev)).float().cpu().numpy())
    for k in ENC:
        E = np.concatenate(out[k]); assert np.isfinite(E).all() and len(E) == len(paths); np.save(f"results/{ds}_embed_{k}.npy", E); print(ds, k, E.shape, flush=True)
