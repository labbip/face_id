"""Test statistics for every model comparison quoted in the paper (2026-11-17).
For each paired contrast: n, mean difference, s.e.m., 95 % CI (t distribution), t, df, Cohen's d_z (mean / SD of the paired differences),
two-sided paired-t p and Wilcoxon signed-rank p. All values are computed from the unrounded per-rater differences (seeds averaged per rater first),
re-derived from the evaluation files exactly as the analysis scripts do (07, 33, 44, 83); where an existing results/*_paired.csv row exists, the
mean difference and n are asserted to equal it. No existing result file is rewritten.
Outputs: results/stats_table.csv         (paired contrasts; one row per contrast)
         results/stats_correlations.csv  (correlations: coefficient, n, p, number of permutations for Mantel tests)
Run: uv run python -I scripts/89_stats_table.py     (then scripts/73 adds the section "Test statistics for the model comparisons")
"""
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import t as tdist, ttest_1samp, wilcoxon, spearmanr
R = "results"; rows = []; nchk = [0]
def add(label, source, contrast, x, y=None, ref=None):
    d = (x - y if y is not None else x).dropna().astype(float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n); h = tdist.ppf(0.975, n - 1) * se
    if ref is not None: assert len(ref) == 1 and np.isclose(ref.iloc[0, 0], m, rtol=1e-9, atol=1e-12) and int(ref.iloc[0, 1]) == n, (label, contrast, m, n, ref); nchk[0] += 1
    rows.append(dict(label=label, source_file=source, contrast=contrast, n=n, mean_diff=m, sem=se, ci95_lo=m - h, ci95_hi=m + h, t=m / se, df=n - 1, cohen_dz=m / sd,
                     p_t=ttest_1samp(d, 0).pvalue, p_w=wilcoxon(d).pvalue if d.abs().sum() > 0 else 1.0, checked_against_paired_csv=ref is not None))
def wide(src, metric, n_est, idcol, pat="eval_*.csv", extra=()):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/{src}/{pat}"))], ignore_index=True)
    g = df[df.n_est == n_est].groupby(["kind", idcol])[[metric, *extra]].mean().reset_index()         # seeds averaged per rater
    return g.pivot(index=idcol, columns="kind", values=metric), g
def ref_of(src, n_est, contrast, **kw):
    P = pd.read_csv(f"{R}/{src}_paired.csv"); q = P[(P.n_est == n_est) & (P.contrast == contrast)]
    for k, v in kw.items(): q = q[q[k] == v]
    return q[["delta", "n"]]
# ---- 1. Xu et al. Experiments 1 and 2 (as scripts/07: main runs *_cv, M = 4, amortised inference, seeds averaged per participant)
XU = {}
for e in (1, 2):
    ev = []
    for k in ("D0", "D1", "D2", "D3"):
        for d in sorted(glob.glob(f"{R}/runs/exp{e}_{k}_D8_M4_f*_s*_cv")):
            if os.path.exists(f"{d}/eval.csv"): x = pd.read_csv(f"{d}/eval.csv"); x = x[(x.frac_est == 0.7) & (x.infer == "amortized")].copy(); x["kind"] = k; ev.append(x)
    pm = pd.concat(ev, ignore_index=True).groupby(["pid", "kind"])[["rho_hold", "rho_tr_unseen"]].mean().reset_index()
    bl = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/baselines/exp{e}_f*.csv")], ignore_index=True).set_index("pid"); P = pd.read_csv(f"{R}/paired_exp{e}_cv.csv")
    XU[e] = {m: pm.pivot(index="pid", columns="kind", values=m) for m in ("rho_hold", "rho_tr_unseen")}; XU[e]["bl"] = bl
    def xu(label, a_, b_, metric, e=e, P=P, bl=bl):
        w = XU[e][metric]; y = bl[f"{metric}_agg"].reindex(w.index) if b_ == "agg" else w[b_]; name = f"{a_}-M4-amortized vs " + ("agg" if b_ == "agg" else f"{b_}-M4-amortized")
        add(label, f"paired_exp{e}_cv.csv ({metric}); results/runs/exp{e}_*_cv/eval.csv", f"{a_} - " + ("population RDM" if b_ == "agg" else b_), w[a_], y, P[(P.comparison == name) & (P.metric == metric)][["mean_diff", "n"]])
    for a_, b_ in (("D1", "D0"), ("D2", "D1")): xu(f"Xu Exp. {e}, held-out arrangement trials", a_, b_, "rho_hold")
    if e == 1: xu("Xu Exp. 1, held-out arrangement trials", "D2", "D0", "rho_hold"); xu("Xu Exp. 1, held-out arrangement trials", "D3", "D2", "rho_hold"); xu("Xu Exp. 1, retest task (unmeasured face pairs)", "D2", "D0", "rho_tr_unseen"); xu("Xu Exp. 1, retest task (unmeasured face pairs)", "D2", "D1", "rho_tr_unseen")
    else: xu("Xu Exp. 2, held-out arrangement trials", "D1", "agg", "rho_hold"); xu("Xu Exp. 2, held-out arrangement trials", "D3", "D2", "rho_hold"); xu("Xu Exp. 2, retest task", "D1", "D0", "rho_tr_unseen"); xu("Xu Exp. 2, retest task", "D2", "D1", "rho_tr_unseen"); xu("Xu Exp. 2, retest task", "D1", "agg", "rho_tr_unseen")
# ---- 1. generalisation datasets (Supplementary Table 2): D1 - D0 and D2 - D1
MAIN = "eval_*_D8_M4_f*_s*.csv"
GEN = [("London Set, 20 faces", "london", "r_amort", 20, "rater", MAIN, {}), ("AVFS novel set (56 faces), 20 triplets", "avfs", "acc_amort", 20, "ann", MAIN, {}),
       ("AVFS main set, 20 triplets", "avfs_img", "acc_main", 20, "ann", "eval_*.csv", dict(metric="acc_main")), ("AVFS main set, all estimation triplets", "avfs_img", "acc_main", 1000000, "ann", "eval_*.csv", dict(metric="acc_main")),
       ("FaceSim3D 2D, 20 triplets", "facesim2D", "acc_amort", 20, "ann", "eval_*.csv", dict(metric="acc_amort")), ("FaceSim3D 3D, 20 triplets", "facesim3D", "acc_amort", 20, "ann", "eval_*.csv", dict(metric="acc_amort")),
       ("SCUT-FBP5500, 500 ratings (seen faces)", "scut", "r_amort_seen", 500, "rater", "eval_*.csv", dict(side="seen")),
       ("LAPIS, 20 ratings (seen images)", "lapis_img", "r_amort_seen", 20, "rater", "eval_*.csv", dict(side="seen")), ("LAPIS, 100 ratings (seen images)", "lapis_img", "r_amort_seen", 100, "rater", "eval_*.csv", dict(side="seen")),
       ("AADB, 100 ratings (seen images)", "aadb_img", "r_amort_seen", 100, "rater", "eval_*.csv", dict(side="seen")), ("EVA, 100 ratings (unseen images)", "eva_img", "r_amort_unseen", 100, "rater", "eval_*.csv", dict(side="unseen")),
       ("Jester, 20 ratings (seen jokes)", "jester", "r_amort_seen", 20, "rater", "eval_*.csv", dict(side="seen")), ("Jester, 50 ratings (seen jokes)", "jester", "r_amort_seen", 50, "rater", "eval_*.csv", dict(side="seen")),
       ("Keller odours, 100 ratings (seen odours)", "keller_fix", "r_amort_seen", 100, "rater", "eval_*.csv", dict(side="seen"))]
for label, src, metric, n_est, idc, pat, kw in GEN:
    w, _ = wide(src, metric, n_est, idc, pat)
    for a_, b_ in (("D1", "D0"), ("D2", "D1")):
        if a_ in w and b_ in w: add(label, f"{src}_paired.csv ({metric}, n_est {n_est}); results/{src}/{pat}", f"{a_} - {b_}", w[a_], w[b_], ref_of(src, n_est, f"{a_} - {b_}", **kw))
# ---- 2. XPASS-Vis table (results/xpass_followup_table.csv; sources as scripts/78), n_est 50 and 100, seen and unseen images
XP = [("LAPIS full (553 raters)", "lapis_img"), ("LAPIS 129 raters, all ratings", "lapis_r129_img"), ("LAPIS 129 raters x 227 ratings", "lapis_thin129_img"), ("XPASS art (own models)", "xpass_art_img"),
      ("XPASS art raters <- LAPIS models", "lapis_on_xpass"), ("LAPIS raters on XPASS images <- XPASS models", "xpass_on_lapis"), ("XPASS fashion", "xpass_fashion_img"), ("XPASS scenery", "xpass_scenery_img"),
      ("XPASS art, Like", "xpass_art_Like_img"), ("XPASS fashion, Like", "xpass_fashion_Like_img"), ("XPASS scenery, Like", "xpass_scenery_Like_img"),
      ("XPASS art, Beautiful", "xpass_art_Beautiful_img"), ("XPASS fashion, Beautiful", "xpass_fashion_Beautiful_img"), ("XPASS scenery, Beautiful", "xpass_scenery_Beautiful_img")]
for name, src in XP:
    if not os.path.exists(f"{R}/{src}_paired.csv"): continue
    for n_est in (50, 100):
        for side in ("seen", "unseen"):
            w, _ = wide(src, f"r_amort_{side}", n_est, "rater")
            for a_, b_ in (("D1", "D0"), ("D2", "D1")):
                ref = ref_of(src, n_est, f"{a_} - {b_}", side=side)
                if len(ref): add(f"XPASS table: {name}, {n_est} ratings ({side} images)", f"{src}_paired.csv (r_amort_{side}, n_est {n_est}); results/{src}/eval_*.csv", f"{a_} - {b_}", w[a_], w[b_], ref)
# ---- 3. remaining contrasts quoted with a p value
for n_est in (20, 40):      # London Set, faces held out from training (image version). All 3 seeds are used (as in the figures); results/london_img_paired.csv was written on 2026-10-21 with seeds 0-1 only (D1 - D0 0.0266 / 0.0325 there), so it is not asserted
    w, g = wide("london_img", "r_amort_unseen", n_est, "rater", extra=("r_popimg_unseen",)); s = f"results/london_img/eval_*.csv (r_amort_unseen, r_popimg_unseen, n_est {n_est}; 3 seeds)"
    add(f"London Set image version, unseen faces, {n_est} faces", s, "D1 - D0", w["D1"], w["D0"])
    add(f"London Set image version, unseen faces, {n_est} faces", s, "D1 - population image regression", w["D1"], g[g.kind == "D0"].set_index("rater").r_popimg_unseen.reindex(w.index))
for n_est in (1000000,):      # acc_novel is evaluated with all estimation triplets only
    w, _ = wide("avfs_img", "acc_novel", n_est, "ann"); add(f"AVFS, 56 unseen faces, {'all' if n_est > 1000 else n_est} estimation triplets", f"avfs_img_paired.csv (acc_novel, n_est {n_est}); results/avfs_img/eval_*.csv", "D1 - D0", w["D1"], w["D0"], ref_of("avfs_img", n_est, "D1 - D0", metric="acc_novel"))
_, g = wide("avfs_img", "acc_same", 1000000, "ann", extra=("agree_modal_loo",))
for k in ("D1", "D0"): s = g[g.kind == k].set_index("ann").dropna(subset=["acc_same", "agree_modal_loo"]); add("AVFS main set, 1,000 common triplets", "results/avfs_img/eval_*.csv (acc_same, agree_modal_loo; as scripts/44)", f"{k} - leave-one-out modal vote", s.acc_same, s.agree_modal_loo)
for e, f in ((2, "h6_text_exp2_D1_M4_cv.csv"), (1, "h6_text_exp1_D2_M4_cv.csv")):
    tx = pd.read_csv(f"{R}/{f}").set_index("pid"); add(f"Xu Exp. {e}, z decoded from free text ({f.split('_')[3]} model)", f, "z from text - population", tx.z_text, tx["pop"]); add(f"Xu Exp. {e}, z decoded from free text ({f.split('_')[3]} model)", f, "z from arrangements - population", tx.z_arr, tx["pop"])
for side in ("seen", "unseen"):
    w, _ = wide("eva_img_n68", f"r_amort_{side}", 100, "rater"); add(f"EVA reduced to 68 raters per seed, 100 ratings ({side} images)", f"eva_img_n68_paired.csv; results/eva_img_n68/eval_*.csv", "D2 - D1", w["D2"], w["D1"], ref_of("eva_img_n68", 100, "D2 - D1", side=side))
for ds in ("london", "scut", "eva", "lapis", "jester"):      # sign check: loss of the D1-constrained readout (free - D1), per rater (mean over fold models), one-sample test against 0
    rr = pd.read_csv(f"{R}/sign_check_{ds}_raters.csv").groupby("rater").mean(numeric_only=True); add(f"Sign check, {ds}", f"sign_check_{ds}_raters.csv (gap_free_d1)", "free readout - D1-constrained readout", rr.gap_free_d1)
T = pd.DataFrame(rows); T.to_csv(f"{R}/stats_table.csv", index=False)
# ---- 4. correlations
C = []
for ds in ("london", "scut", "eva", "lapis", "jester"):
    A = pd.read_csv(f"{R}/sign_check_{ds}_gain.csv"); ok = A[["U_free", "d2_minus_d1"]].dropna(); r, p = spearmanr(ok.U_free, ok.d2_minus_d1)
    C.append(dict(label=f"Sign check, {ds}: reversed weight energy vs D2 - D1", source_file=f"sign_check_{ds}_gain.csv", statistic="Spearman rho", coef=r, n=len(ok), n_unit="raters", p=p, p_kind="t approximation", n_perm=np.nan, n_models=np.nan))
for _, q in pd.read_csv(f"{R}/xpass_cross_q1.csv").iterrows():
    C.append(dict(label=f"XPASS cross-domain: {q.measure}, {q.a} vs {q.b}", source_file="xpass_cross_q1.csv", statistic="Spearman rho", coef=q.spearman, n=int(q.n), n_unit="raters", p=q.p, p_kind="t approximation", n_perm=np.nan, n_models=np.nan))
Q2 = pd.read_csv(f"{R}/xpass_cross_q2.csv"); NP = {"model-free rater similarity (Mantel)": 2000, "z distance split-half (ratings)": 50}      # permutations as coded in scripts/76 (--n_perm default 2000 for the between-domain model-free Mantel, 50 for the z split-half, 200 for the others); p = (count + 1) / (n_perm + 1)
for (test, a_, b_), q in Q2.groupby(["test", "a", "b"], sort=False):
    one = len(q) == 1
    C.append(dict(label=f"XPASS cross-domain: {test}, {a_} vs {b_}", source_file="xpass_cross_q2.csv", statistic="Mantel Spearman rho" if "Mantel" in test or "split-half" in test else "first canonical correlation", coef=q.r.mean(), n=int(q.n_pairs.iloc[0]) if "n_pairs" in q and pd.notna(q.n_pairs.iloc[0]) else np.nan, n_unit="raters" if "CCA" in test else "rater pairs",
                  p=q.p.iloc[0] if one else q.p.median(), p_kind="permutation" if one else "permutation, median over fold models", n_perm=NP.get(test, 200), n_models=len(q)))
C = pd.DataFrame(C); C.to_csv(f"{R}/stats_correlations.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 60)
print(T.drop(columns=["source_file"]).to_string(index=False, float_format=lambda v: f"{v:.4g}")); print(f"\n{len(T)} contrasts, {nchk[0]} checked against the existing *_paired.csv files"); print(C.drop(columns=["source_file"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
