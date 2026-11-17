"""XPASS-Vis (Hayashi et al. 2026, arXiv:2606.15629) -> sparse-rating tables for scripts/53_lapis_faceid_img.py.
Input : data/raw/xpass/data/maked/{ratings,users}.csv (129 raters x art/fashion/scenery, 87,836 interactions).
Output: data/processed/xpass_{art,fashion,scenery}.csv  (participant_id, image_id, rating=<--col>, plus all item columns)
        results/xpass_art_image_ids.npy / xpass_art_embed_clip.npy  (rows of results/lapis_embed_clip.npy for the 2,345 LAPIS test images)
        data/processed/xpass_users.csv  (users + TIPI Big Five: E=Q1,Q6R  A=Q2R,Q7  C=Q3,Q8R  N=Q4R,Q9  O=Q5,Q10R; 0-6 scale)
        results/xpass_retest.csv  (per rater x genre test-retest r over the ~12 repeated items, per item column)
Run: uv run python -I scripts/74_xpass_prep.py [--col Aesthetic]
"""
import argparse, os, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--col", default="Aesthetic"); a = ap.parse_args()
RAW = "data/raw/xpass/data/maked/"; ITEMS = ["Like", "Beautiful", "Distasteful", "Impressed", "Intellectually", "Motivated", "Nostalgic", "Sad", "Amused", "Aesthetic"]
r = pd.read_csv(RAW + "ratings.csv"); u = pd.read_csv(RAW + "users.csv")
r["image_id"] = r.sample_file.str.replace(".mp4", ".jpg", regex=False)          # scenery: one frame per clip, provided as jpg
lap = pd.read_csv("data/raw/lapis/piaa.csv", usecols=["image_id", "image_filename"], low_memory=False).drop_duplicates()
fmap = dict(zip(lap.image_filename, lap.image_id)); miss = set(r[r.genre == "art"].image_id) - set(fmap); assert not miss, miss
r.loc[r.genre == "art", "image_id"] = r.loc[r.genre == "art", "image_id"].map(fmap).astype(int).astype(str)
# test-retest over repeated (rater, item) pairs, per genre and item column
rows = []
for (g, k), q in r.groupby(["genre", "user_id"]):
    dup = q[q.duplicated("image_id", keep=False)].sort_values(["image_id", "Time"]) if "Time" in q else q[q.duplicated("image_id", keep=False)]
    first = dup.groupby("image_id").head(1).set_index("image_id"); second = dup.groupby("image_id").tail(1).set_index("image_id")
    row = dict(genre=g, participant_id=k, n_rep=len(first))
    for c in ITEMS: row[f"r_{c}"] = np.corrcoef(first[c], second.loc[first.index, c])[0, 1] if len(first) >= 5 and first[c].std() > 0 and second[c].std() > 0 else np.nan
    rows.append(row)
R = pd.DataFrame(rows); R.to_csv("results/xpass_retest.csv", index=False)
print("test-retest r (mean over raters) per genre:\n", R.groupby("genre")[[f"r_{c}" for c in ITEMS]].mean().round(3).T.to_string())
print("n repeated items per rater x genre:", R.n_rep.describe()[["min", "50%", "max"]].to_dict())
os.makedirs("data/processed", exist_ok=True); sfx = "" if a.col == "Aesthetic" else f"_{a.col}"
for g in ("art", "fashion", "scenery"):
    q = r[r.genre == g].rename(columns={"user_id": "participant_id"}); q["rating"] = q[a.col]
    q = q[["participant_id", "image_id", "rating"] + ITEMS + ["Time"]]; q.to_csv(f"data/processed/xpass_{g}{sfx}.csv", index=False)
    print(g, "rows", len(q), "raters", q.participant_id.nunique(), "items", q.image_id.nunique(), "rating col", a.col, "mean %.2f sd %.2f" % (q.rating.mean(), q.rating.std()))
ids_all = np.load("results/lapis_image_ids.npy", allow_pickle=True); E = np.load("results/lapis_embed_clip.npy"); pos = {int(i): j for j, i in enumerate(ids_all)}
art_ids = np.array(sorted(r[r.genre == "art"].image_id.astype(int).unique())); np.save("results/xpass_art_image_ids.npy", art_ids); np.save("results/xpass_art_embed_clip.npy", E[[pos[i] for i in art_ids]])
print("art ids", len(art_ids), "embed", E[[pos[i] for i in art_ids]].shape)
u["tipi_E"] = (u.Q1 + (6 - u.Q6)) / 2; u["tipi_A"] = ((6 - u.Q2) + u.Q7) / 2; u["tipi_C"] = (u.Q3 + (6 - u.Q8)) / 2; u["tipi_N"] = ((6 - u.Q4) + u.Q9) / 2; u["tipi_O"] = (u.Q5 + (6 - u.Q10)) / 2
u.rename(columns={"user_id": "participant_id"}).to_csv("data/processed/xpass_users.csv", index=False); print(u[[c for c in u if c.startswith("tipi")]].describe().loc[["mean", "std"]].round(2).to_string())
