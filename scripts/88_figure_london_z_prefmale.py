r"""london_z_prefmale.pdf (Supplementary Fig S3c), redrawn 2026-11-13 so that the figure and the quoted number come from the same analysis.
Only the main models are used (London learned-embedding version, D1, D = 8, M = 4; 10 folds x 5 seeds = 50 fold models; z at 40 estimation faces).
The z basis differs between fold models, so z is never averaged across models: within each fold model, the male-face preference index of its
fold-out raters is predicted from their z by 10-fold ridge CV (as before: RidgeCV, alphas logspace(-2, 3, 12)).  Each rater is fold-out in one
fold, so the 10 fold models of ONE seed (PLOT_SEED = 0) give every rater exactly one (prediction, observed) point; these are plotted.
Predictions are NOT averaged over the 5 seeds: the average of 5 predictions is an ensemble with less noise (R2 0.82) and would make the
scatter look tighter than the quoted per-model value (0.71).  Reported: per-model CV R2 over all 50 models (mean, range; the number quoted
in the paper), the same for the plotted seed, the R2 of the plotted points, and (for the record) the R2 of seed-averaged predictions.
(The earlier version in scripts/34 averaged z over all eval_D1_*.csv, including the D/M sensitivity models -> CV R2 0.30.)
Outputs figures/london_z_prefmale.pdf, results/london_z_prefmale.csv (per rater), results/london_z_prefmale_models.csv (per fold model).
Run: sed -i '/^- `london_z_prefmale\.pdf`/d' figures/INDEX.md; uv run python -I scripts/88_figure_london_z_prefmale.py
"""
import glob, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from face_id.plotting import plt, savefig, TAB10, figdata, FROM_FIGDATA
def compute():
    """Analysis (needs the London ratings and the per-rater evaluation files); returns the plotted points and the numbers of the index line."""
    r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(float)
    info = pd.read_csv("data/raw/london/london_faces_info.csv", dtype={"face_id": str}); fg = (info.face_gender.to_numpy() == "male")
    res = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean(); pref = res[:, fg].mean(1) - res[:, ~fg].mean(1)
    def r2_(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rows, mods = [], []
    for x in sorted(glob.glob("results/london/eval_D1_D8_M4_f*_s*.csv")):
        d = pd.read_csv(x); d = d[d.n_est == 40].dropna(subset=["z0"]).set_index("rater"); Z = d[["z0", "z1", "z2", "z3"]].to_numpy(); y = pref[d.index]
        pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Z, y, cv=10)
        mods.append(dict(fold=int(d.fold.iloc[0]), seed=int(d.seed.iloc[0]), n=len(y), cv_r2=r2_(y, pr))); rows.append(pd.DataFrame(dict(rater=d.index, fold=int(d.fold.iloc[0]), seed=int(d.seed.iloc[0]), pred=pr, obs=y)))
    M = pd.DataFrame(mods); A = pd.concat(rows); G = A.groupby("rater").agg(pred=("pred", "mean"), obs=("obs", "first"), n_models=("pred", "size")).reset_index()
    M.to_csv("results/london_z_prefmale_models.csv", index=False); G.to_csv("results/london_z_prefmale.csv", index=False)
    PLOT_SEED = 0; Pl = A[A.seed == PLOT_SEED].sort_values("rater"); assert Pl.rater.is_unique and len(Pl) == len(G); G = G.merge(Pl[["rater", "pred"]].rename(columns={"pred": "pred_plotted"}), on="rater"); G.to_csv("results/london_z_prefmale.csv", index=False)
    Ms = M[M.seed == PLOT_SEED]; r2_plot = r2_(Pl.obs.to_numpy(), Pl.pred.to_numpy()); r2_avg = r2_(G.obs.to_numpy(), G.pred.to_numpy()); r2_single = r2_(A.obs.to_numpy(), A.pred.to_numpy())
    print(f"fold models {len(M)}; per-model CV R2 mean {M.cv_r2.mean():.3f} (range {M.cv_r2.min():.3f}-{M.cv_r2.max():.3f}); raters {len(G)} (models per rater {G.n_models.min()}-{G.n_models.max()})")
    print(f"plotted: seed {PLOT_SEED}, {len(Ms)} fold models, per-model CV R2 mean {Ms.cv_r2.mean():.3f} (range {Ms.cv_r2.min():.3f}-{Ms.cv_r2.max():.3f}); R2 of the plotted points {r2_plot:.3f}, Pearson r {np.corrcoef(Pl.pred, Pl.obs)[0, 1]:.3f}")
    print(f"not plotted: R2 of predictions averaged over the 5 seeds {r2_avg:.3f}; R2 pooling all 50 models' points {r2_single:.3f}; pooled R2 per seed " + ", ".join(f"{r2_(g.obs.to_numpy(), g.pred.to_numpy()):.3f}" for _, g in A.groupby("seed")))
    return pd.DataFrame(dict(predicted=Pl.pred.to_numpy(), observed=Pl.obs.to_numpy())).assign(n_models_plotted=len(Ms), n_models=len(M), cv_r2_mean=M.cv_r2.mean(), cv_r2_min=M.cv_r2.min(), cv_r2_max=M.cv_r2.max(),
                                                                                          cv_r2_mean_plotted=Ms.cv_r2.mean(), cv_r2_min_plotted=Ms.cv_r2.min(), cv_r2_max_plotted=Ms.cv_r2.max(), r2_points=r2_plot, r2_seed_average=r2_avg)
# Source data: results/figdata/london_z_prefmale.csv holds one row per rater (predicted, observed; no identifier). This is the one panel whose source data
# are per-rater values (decision of 2026-11-17: published, London Set is CC BY and the values are derived indices, not ratings). FIGDATA=read redraws from it.
PLOT_SEED = 0; T = figdata("london_z_prefmale", compute); q = T.iloc[0]
pr, y = T.predicted.to_numpy(), T.observed.to_numpy()
fig, ax = plt.subplots(); ax.scatter(pr, y, s=3, alpha=0.3, color=TAB10[0], linewidths=0)
lim = [min(pr.min(), y.min()), max(pr.max(), y.max())]; ax.plot(lim, lim, "k:", lw=0.8)
ax.set_xlabel("Male-face preference predicted from $z$"); ax.set_ylabel("Male-face preference (rating residual)")
savefig(fig, "london_z_prefmale", w=3.0, h=2.8, index_line=f"London Set: male-face preference index (mean residual rating of male minus female faces) vs its prediction from the D1 latent z (learned-embedding version, D = 8, M = 4 main models only, 40 estimation faces). Within each fold model the index of the fold-out raters is predicted from their z by 10-fold ridge CV (z is not pooled across models because the basis differs). Plotted: the {int(q.n_models_plotted)} fold models of seed {PLOT_SEED}, so each rater appears once (n = {len(T)}), in the model where the rater is fold-out; dotted = identity. CV R2 per fold model = {q.cv_r2_mean:.2f} over all {int(q.n_models)} models (10 folds x 5 seeds; range {q.cv_r2_min:.2f}-{q.cv_r2_max:.2f}; the value to quote) and {q.cv_r2_mean_plotted:.2f} over the {int(q.n_models_plotted)} plotted models (range {q.cv_r2_min_plotted:.2f}-{q.cv_r2_max_plotted:.2f}); R2 of the plotted points pooled = {q.r2_points:.2f}. Predictions are not averaged over seeds (that ensemble gives R2 {q.r2_seed_average:.2f} and would not match the quoted value). From results/london_z_prefmale.csv, results/london_z_prefmale_models.csv (scripts/88; redrawn 2026-11-13, the earlier version mixed in the D/M sensitivity models and gave 0.30).")
