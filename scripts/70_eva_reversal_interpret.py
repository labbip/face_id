"""What do the EVA raters with sign-reversed weights reverse? Uses results/sign_check_eva_{dims,raters}.csv (per-rater weights in each fold
model's basis, flipped-energy U) plus the image content categories (data/raw/eva/image_content_category.csv, 6 classes) and user attributes.
Outputs: correlation of each weight dimension with per-rater category preference (residual rating per category), category preferences of
the most-reversed quartile vs the rest, and U vs user attributes. Run: uv run python -I scripts/70_eva_reversal_interpret.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, kruskal
v = pd.read_csv("data/processed/eva_ratings.csv"); v["image_id"] = v.image_id.astype(str)
cat = pd.read_csv("data/raw/eva/image_content_category.csv", dtype=str).rename(columns={"sort": "cat"}); v = v.merge(cat, on="image_id", how="left")
n = v.groupby("participant_id").size(); v = v[v.participant_id.isin(n[n >= 100].index)]
v["res"] = v.rating - v.groupby("image_id").rating.transform("mean") - v.groupby("participant_id").rating.transform("mean") + v.rating.mean()
P = v.groupby(["participant_id", "cat"]).res.mean().unstack()                       # rater x category residual preference
R = pd.read_csv("results/sign_check_eva_raters.csv").groupby("rater").mean(numeric_only=True); D = pd.read_csv("results/sign_check_eva_dims.csv")
print("category preference (residual rating, 0-10 scale) spread across raters: SD per category", P.std().round(3).to_dict())
print("split-half reliability of category preference (images split at random):")
rng = np.random.default_rng(0); imgs = v.image_id.unique(); h = set(rng.permutation(imgs)[: len(imgs) // 2])
Pa = v[v.image_id.isin(h)].groupby(["participant_id", "cat"]).res.mean().unstack(); Pb = v[~v.image_id.isin(h)].groupby(["participant_id", "cat"]).res.mean().unstack()
print({c: round(Pa[c].corr(Pb[c]), 3) for c in P.columns})
# which weight dimension tracks which category preference (per fold model basis)
rows = []
for (fold, s), g in D[~D.pop_row].groupby(["fold", "seed"]):
    g = g.set_index("rater"); pop = D[(D.fold == fold) & (D.seed == s) & D.pop_row].iloc[0]
    for k in range(8):
        for c in P.columns:
            x = g[f"w{k}"]; y = P[c].reindex(x.index); ok = x.notna() & y.notna()
            if ok.sum() >= 10: rows.append(dict(fold=fold, seed=s, dim=k, cat=c, r=spearmanr(x[ok], y[ok])[0], share_pop=pop[f"w{k}"] ** 2 / sum(pop[f"w{j}"] ** 2 for j in range(8)), relflip=g[f"relflip{k}"].astype(float).mean()))
T = pd.DataFrame(rows); T.to_csv("results/eva_reversal_dims.csv", index=False)
best = T.loc[T.groupby(["fold", "seed", "dim"]).r.apply(lambda x: x.abs().idxmax())]
print("per fold model: for each dim, the category whose preference correlates most with the dim's weight (|r|), the dim's share of population weight, reliable-flip fraction")
print(best[best.fold.isin([0, 1])].round(3).to_string(index=False))
print("dims with reliable flips >= 0.15: mean |r| with their best category %.3f; other dims %.3f" % (best[best.relflip >= 0.15].r.abs().mean(), best[best.relflip < 0.15].r.abs().mean()))
# most-reversed quartile vs rest: category preferences
U = R.U_free.reindex(P.index).dropna(); q4 = U >= U.quantile(0.75)
print(f"raters with U: {len(U)}; Q4 (most reversed, n={q4.sum()}) vs rest: mean category preference and Mann-Whitney p")
for c in P.columns:
    a, b = P.loc[q4[q4].index, c].dropna(), P.loc[q4[~q4].index, c].dropna(); print(f"  cat {c}: Q4 {a.mean():+.3f} (SD {a.std():.3f}) rest {b.mean():+.3f} (SD {b.std():.3f}) p={mannwhitneyu(a, b).pvalue:.3g}")
print("|category preference| summed over categories: Q4 %.3f vs rest %.3f (p=%.3g)" % (P.abs().sum(1)[q4[q4].index].mean(), P.abs().sum(1)[q4[~q4].index].mean(), mannwhitneyu(P.abs().sum(1)[q4[q4].index], P.abs().sum(1)[q4[~q4].index]).pvalue))
print("spearman(U, |category preference| sum) = %.3f (p=%.2g)" % spearmanr(U, P.abs().sum(1).reindex(U.index))[:2])
# user attributes
u = pd.read_csv("data/raw/eva/users.csv", sep="=").set_index("id"); u = u.reindex(U.index)
for col, name in (("photographic_level_id", "photography level"), ("gender_id", "gender"), ("age", "birth year")):
    x = u[col]; ok = x.notna() & U.notna()
    if col == "age": print(f"spearman(U, {name}) = {spearmanr(U[ok], x[ok])[0]:+.3f} (p={spearmanr(U[ok], x[ok])[1]:.2g}, n={ok.sum()})")
    else: grp = [U[ok][x[ok] == g_] for g_ in sorted(x[ok].unique())]; print(f"U by {name} {dict(zip(sorted(x[ok].unique()), [round(g_.mean(), 3) for g_ in grp]))} n={[len(g_) for g_ in grp]} Kruskal p={kruskal(*[g_ for g_ in grp if len(g_) > 1]).pvalue:.2g}")
