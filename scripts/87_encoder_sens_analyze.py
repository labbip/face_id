"""Encoder sensitivity (2026-11-12): the image-version results with CLIP ViT-B/32 (main), DINOv2 ViT-B/14 and CLIP ViT-L/14.
Inputs: results/<ds>_img[_<enc>]/eval_D[012]_*.csv (scripts/53, 31 via 86) and results/sign_check_<ds>[_<enc>]_{raters,gain,dims,perdim}.csv (scripts/55 with ENC=<enc>).
Outputs (one row per dataset x encoder [x n_est x side]):
  results/encoder_sens_perf.csv    image-mean baseline, D0, D1, D2 (per-rater r, seeds averaged per rater), D1 - D0 and D2 - D1 with SE and paired-t / Wilcoxon p
  results/encoder_sens_sign.csv    sign-check metrics as in the paper_numbers sign-check table (per-rater means over fold models)
  results/encoder_sens_varaxis.csv variation-axis statistics as in scripts/79 (median over fold models; frac_opposite also pooled)
Run: uv run python -I scripts/87_encoder_sens_analyze.py
"""
import glob, os, numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon, spearmanr, skew, kurtosis
ENCS = (("clipb32", "", "CLIP ViT-B/32 (main)"), ("dinov2b", "_dinov2b", "DINOv2 ViT-B/14"), ("clipl", "_clipl", "CLIP ViT-L/14"))
DSS = ("lapis", "eva", "aadb", "london")
def bimod(x): n = len(x); return (skew(x) ** 2 + 1) / (kurtosis(x) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
perf, sign, vax = [], [], []
for ds in DSS:
    for enc, suf, lab in ENCS:
        pat = f"results/london_img{suf}/eval_D[012]_D8_M4_{enc if suf else 'clip'}_f*_s*.csv" if ds == "london" else f"results/{ds}_img{suf}/eval_D[012]_D8_M4_f*_s*.csv"
        fs = sorted(glob.glob(pat))
        if not fs: continue
        ev = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True); nruns = ev.groupby("kind").apply(lambda g: len(g[["fold", "seed"]].drop_duplicates())).to_dict()
        for n_est in ((20, 40) if ds == "london" else (20, 100)):
            for side in ("seen", "unseen"):
                popc = "r_popimg_unseen" if (ds == "london" and side == "unseen") else f"r_pop_{side}"     # london unseen: image-regression baseline (r_pop_unseen there is the held-out faces' own mean)
                g = ev[ev.n_est == n_est].groupby(["kind", "rater"])[[popc, f"r_amort_{side}"]].mean().unstack(0)
                A = g[f"r_amort_{side}"]; row = dict(dataset=ds, encoder=enc, n_est=n_est, side=side, runs_per_kind=nruns.get("D2"), n_raters=int(A.dropna().shape[0]), r_image_mean=g[popc]["D0"].mean(), r_D0=A["D0"].mean(), r_D1=A["D1"].mean(), r_D2=A["D2"].mean())
                for a_, b_ in (("D1", "D0"), ("D2", "D1")):
                    d = (A[a_] - A[b_]).dropna(); k = f"{a_}_minus_{b_}"; row.update({k: d.mean(), f"{k}_se": d.std() / np.sqrt(len(d)), f"{k}_p_t": ttest_rel(A[a_][d.index], A[b_][d.index]).pvalue, f"{k}_p_w": wilcoxon(d).pvalue})
                perf.append(row)
        sc = f"results/sign_check_{ds}{suf}"
        if not os.path.exists(f"{sc}_raters.csv"): continue
        r = pd.read_csv(f"{sc}_raters.csv"); G = pd.read_csv(f"{sc}_gain.csv"); rr = r.groupby("rater").mean(numeric_only=True)
        q = pd.qcut(G.U_free.rank(method="first"), 4, labels=False); qm = G.groupby(q).d2_minus_d1.mean()
        sign.append(dict(dataset=ds, encoder=enc, n_raters=r.rater.nunique(), median_ratings=r.n_rat.median(), weight_splithalf_r=rr.halves_r.mean(), frac_cos_neg=(rr.cos_pop < 0).mean(), U_flipped_energy=rr.U_free.mean(), U_noise_unbiased=rr.U_unb.mean(),
                         gap_free_minus_D1=rr.gap_free_d1.mean(), gap_se=rr.gap_free_d1.std() / np.sqrt(len(rr)), frac_gap_gt_0_02=(rr.gap_free_d1 > 0.02).mean(), mean_D2_D1=G.d2_minus_d1.mean(),
                         spearman_U_vs_D2D1=spearmanr(G.U_free, G.d2_minus_d1, nan_policy="omit")[0], spearman_p=spearmanr(G.U_free, G.d2_minus_d1, nan_policy="omit")[1], D2D1_Q1=qm.get(0, np.nan), D2D1_Q2=qm.get(1, np.nan), D2D1_Q3=qm.get(2, np.nan), D2D1_Q4=qm.get(3, np.nan)))
        D = pd.read_csv(f"{sc}_dims.csv"); R = r.set_index(["fold", "seed", "rater"]); wc = [c for c in D.columns if c[0] == "w" and c[1:].isdigit()]; rows = []; pooled = []
        for (fold, s), g in D[D.rater.astype(str) != "-1"].groupby(["fold", "seed"]):     # as scripts/79
            W = g[wc].to_numpy(float); wpop = D[(D.fold == fold) & (D.seed == s) & (D.rater.astype(str) == "-1")][wc].to_numpy(float)[0]
            lam, V = np.linalg.eigh(np.cov(W.T)); lam, V = lam[::-1], V[:, ::-1]; v1 = V[:, 0] * np.sign(wpop @ V[:, 0]); sco = W @ v1; pooled.extend(sco / sco.std())
            row = dict(var_share1=lam[0] / lam.sum(), cos_v1_pop=abs(v1 @ wpop) / np.linalg.norm(wpop), pop_proj_over_sd=(wpop @ v1) / sco.std(), frac_opposite=(sco < 0).mean(), bimodality=bimod(sco), var_over_pop=np.sqrt(lam.sum()) / np.linalg.norm(wpop))
            if "abs_pref" in R.columns:
                ap = R.loc[list(zip(g.fold, g.seed, g.rater)), "abs_pref"].to_numpy(); row["rho_s_abs"] = spearmanr(sco, ap, nan_policy="omit")[0]; res = W - np.outer(sco, v1); row["rho_resid_abs_max"] = max(abs(spearmanr(res @ V[:, j], ap, nan_policy="omit")[0]) for j in range(1, len(wc)))
            rows.append(row)
        T = pd.DataFrame(rows); vax.append(dict(dataset=ds, encoder=enc, n_fold_models=len(T), **T.median().to_dict(), frac_opposite_pooled=float(np.mean(np.array(pooled) < 0)), frac_opposite_min=T.frac_opposite.min(), frac_opposite_max=T.frac_opposite.max(), bimodality_max=T.bimodality.max(),
                                **({"rho_s_abs_min": T.rho_s_abs.min(), "rho_s_abs_max": T.rho_s_abs.max()} if "rho_s_abs" in T else {})))
P, S, V = pd.DataFrame(perf), pd.DataFrame(sign), pd.DataFrame(vax)
P.to_csv("results/encoder_sens_perf.csv", index=False); S.to_csv("results/encoder_sens_sign.csv", index=False); V.to_csv("results/encoder_sens_varaxis.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(P.drop(columns=[c for c in P.columns if c.endswith("_p_w")]).round(4).to_string(index=False)); print(); print(S.round(4).to_string(index=False)); print(); print(V.round(3).to_string(index=False))
