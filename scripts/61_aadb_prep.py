"""AADB (Kong et al. ECCV 2016): raw MTurk results (data/raw/aadb/unzip/ImageAesthetics_ECCV2016/AllinAll.csv; 10 images per HIT,
overall score + 11 attributes, anonymised WorkerId) -> long table data/processed/aadb_ratings.csv (participant_id, image_id, rating, attributes)
and a suitability screen (ratings per worker, residual variance share, split-half stability of between-worker residual similarity).
Run: uv run python -I scripts/61_aadb_prep.py
"""
import re, numpy as np, pandas as pd
d = pd.read_csv("data/raw/aadb/unzip/ImageAesthetics_ECCV2016/AllinAll.csv", low_memory=False)
att = ["overallScore", "ColorHarmony", "Content", "DoF", "MotionBlur", "ObjectEmphasis", "Repetition", "RuleOfThirds", "StrongColor", "Symmetry", "VisualBalance", "choiceLight"]
rows = []
for i in range(1, 11):
    sub = d[["WorkerId", "AssignmentStatus", "SubmitTime", f"Input.image_url{i}"] + [f"Answer.{a}{i}" for a in att]].copy()
    sub.columns = ["participant_id", "status", "submit_time", "url"] + att; rows.append(sub)
L = pd.concat(rows, ignore_index=True)
L["image_id"] = L.url.str.replace(r"^https?://", "", regex=True).str.replace(".staticflickr.com/", "_", regex=False).str.replace("/", "_", regex=False)
L = L.rename(columns={"overallScore": "rating"}); L["rating"] = pd.to_numeric(L.rating, errors="coerce"); L = L.dropna(subset=["rating"])
L = L[["participant_id", "image_id", "rating", "status"] + att[1:]]; L.to_csv("data/processed/aadb_ratings.csv", index=False)
n = L.groupby("participant_id").size(); print(f"ratings {len(L)} images {L.image_id.nunique()} workers {len(n)}; rating values {sorted(L.rating.unique())}")
print("workers with >= 50/100/200/500 ratings:", [(n >= k).sum() for k in (50, 100, 200, 500)], "| ratings covered by workers >= 100:", round(n[n >= 100].sum() / len(L), 3))
print("ratings per image:", L.groupby("image_id").size().describe()[["mean", "min", "max"]].round(2).to_dict(), "| duplicate (worker, image) pairs:", L.duplicated(["participant_id", "image_id"]).sum())
# screen: variance decomposition (image mean, rater mean, residual) on workers >= 100 ratings
g = L[L.participant_id.isin(n[n >= 100].index)].groupby(["participant_id", "image_id"], as_index=False).rating.mean()
im = g.groupby("image_id").rating.transform("mean"); rm = g.groupby("participant_id").rating.transform("mean"); res = g.rating - im - rm + g.rating.mean()
print(f"screen (workers >= 100): total var {g.rating.var():.3f}; image-mean share {im.var() / g.rating.var():.3f}; rater-mean share {rm.var() / g.rating.var():.3f}; residual share {res.var() / g.rating.var():.3f}")
# split-half stability of each worker's residual profile correlation with other workers (images split at random)
g["res"] = res; rng = np.random.default_rng(0); imgs = g.image_id.unique(); st = []
for _ in range(10):
    h = set(rng.permutation(imgs)[: len(imgs) // 2]); A = g[g.image_id.isin(h)].pivot(index="participant_id", columns="image_id", values="res"); B = g[~g.image_id.isin(h)].pivot(index="participant_id", columns="image_id", values="res")
    def sim(P):
        P = P.sub(P.mean(1), axis=0); X = P.fillna(0).to_numpy(); M = (~P.isna()).to_numpy().astype(float); num = X @ X.T; den = np.sqrt((X ** 2 @ M.T) * (M @ (X ** 2).T)) + 1e-9; return num / den
    SA, SB = sim(A), sim(B.reindex(A.index)); iu = np.triu_indices(len(SA), 1); ok = np.isfinite(SA[iu]) & np.isfinite(SB[iu]); st.append(np.corrcoef(SA[iu][ok], SB[iu][ok])[0, 1])
print(f"split-half stability of between-worker residual similarity: {np.mean(st):.3f} (workers {g.participant_id.nunique()})")
