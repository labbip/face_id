"""Sign check: is the D2 > D1 advantage (LAPIS art) due to raters whose preference direction is sign-flipped relative to
the population on some embedding dimension (which D1 = same-sign reweighting cannot express)? Same analysis on London faces.
For each D2 fold model (f basis of that model): (a) model-based w_i = w0 + wz(z_i) for held-out raters (z at the largest n_est);
(b) model-free per-rater ridge of z-scored ratings on f over all their training-image ratings, split-half sign reliability;
U_i = fraction of the rater's weight energy on dims whose sign is opposite to the population = part D1 cannot represent.
Relates U_i to the per-rater D2 - D1 gain. Run: uv run python -I scripts/55_sign_check.py lapis|london
"""
import os, sys, glob, warnings, numpy as np, pandas as pd, torch   # usage: lapis | london | scut ; ENC=<encoder> env var (2026-11-12): encoder-sensitivity models (results/<ds>_img_<ENC>, results/<ds>_embed_<ENC>.npy) -> results/sign_check_<ds>_<ENC>_*
warnings.filterwarnings("ignore")
from scipy.special import erf
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
DS = sys.argv[1]; ENC = os.environ.get("ENC", ""); out = f"results/sign_check_{DS}" + (f"_{ENC}" if ENC else "")
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
def ridge_w(F, y, alpha=1.0): return Ridge(alpha=alpha).fit(F, y).coef_
def d1_w(F, y, wpop, alpha=1.0):            # D1-constrained per-rater weights: w = wpop * s, s >= 0 (ridge-regularised NNLS)
    G = F * wpop[None]; A = np.vstack([G, np.sqrt(alpha) * np.eye(F.shape[1])]); s_, _ = nnls(A, np.concatenate([y, np.zeros(F.shape[1])])); return wpop * s_
def cv_gap(F, y, wpop, h1, h2):             # split-half CV: free ridge vs D1-constrained; returns (r_free, r_d1)
    rf, rd = [], []
    for a_, b_ in ((h1, h2), (h2, h1)):
        wf, wd = ridge_w(F[a_], y[a_]), d1_w(F[a_], y[a_], wpop)
        rf.append(np.corrcoef(F[b_] @ wf, y[b_])[0, 1] if (F[b_] @ wf).std() > 0 else 0.0); rd.append(np.corrcoef(F[b_] @ wd, y[b_])[0, 1] if (F[b_] @ wd).std() > 0 else 0.0)
    return np.mean(rf), np.mean(rd)
if DS in ("lapis", "eva", "aadb") or DS.startswith(("xpass_", "sim_lapis_")):   # sim_lapis_<cond>: synthetic LAPIS-like raters (81)   # sparse-rating image models from 53_lapis_faceid_img.py (xpass_<art|fashion|scenery>: XPASS-Vis, 74/75)
    CFG = {"lapis": ("results/lapis_img", "data/raw/lapis/piaa.csv", "results/lapis_image_ids.npy", "results/lapis_embed_clip.npy"), "eva": ("results/eva_img", "data/processed/eva_ratings.csv", "results/eva_image_ids.npy", "results/eva_embed_clip.npy"), "aadb": ("results/aadb_img", "data/processed/aadb_ratings.csv", "results/aadb_image_ids.npy", "results/aadb_embed_clip.npy"),
           **{f"xpass_{g}": (f"results/xpass_{g}_img", f"data/processed/xpass_{g}.csv", f"results/xpass_{g}_image_ids.npy", f"results/xpass_{g}_embed_clip.npy") for g in ("art", "fashion", "scenery")},
           **{f"sim_lapis_{c}": (f"results/sim_lapis_{c}", f"data/processed/sim_lapis_{c}.csv", "results/lapis_image_ids.npy", "results/lapis_embed_clip.npy") for c in ("cont", "disc")}}[DS]
    if ENC: CFG = (f"{CFG[0]}_{ENC}", CFG[1], CFG[2], CFG[3].replace("_clip.npy", f"_{ENC}.npy"))
    SRC, NEST, tagf = CFG[0], 100, lambda f, s: f"D2_D8_M4_f{f}_s{s}"
    d = pd.read_csv(CFG[1], low_memory=False)
    sty = d.groupby("image_id")["2_styles"].first() if DS == "lapis" else (pd.read_csv("data/raw/lapis/piaa.csv", usecols=["image_id", "2_styles"], low_memory=False).drop_duplicates("image_id").set_index("image_id")["2_styles"] if DS == "xpass_art" or DS.startswith("sim_lapis_") else None)   # xpass_art / sim_lapis images are LAPIS images
    d = d.groupby(["participant_id", "image_id"], as_index=False).rating.mean()
    n = d.groupby("participant_id").size(); d = d[d.participant_id.isin(n[n >= 100].index)].reset_index(drop=True)
    ids = np.load(CFG[2], allow_pickle=True); imap = {str(m): i for i, m in enumerate(ids)}; d["i"] = d.image_id.astype(str).map(imap); d = d.dropna(subset=["i"]); d["i"] = d.i.astype(int); Fn_ = len(ids)
    raters = sorted(d.participant_id.unique()); N = len(raters); d["k"] = d.participant_id.map({r: i for i, r in enumerate(raters)})
    rfold = np.random.default_rng(0).permutation(N) % 10; ifold = np.random.default_rng(1).permutation(Fn_) % 5
    d["rf"] = rfold[d.k]; d["if"] = ifold[d.i]; E = np.load(CFG[3])
    abstract = (sty.reindex(ids).to_numpy() == "ABSTRACT") if sty is not None else None
    def fold_data(fold, sd):
        itr = np.where(ifold != fold % 5)[0]; pca = PCA(64, random_state=0).fit(E[itr]); X = pca.transform(E); X = (X - X[itr].mean(0)) / X[itr].std(0)
        f = X @ sd["P.weight"].numpy().T + sd["P.bias"].numpy()
        tr = d[(d.rf != fold) & (d["if"] != fold % 5)]; te = d[(d.rf == fold) & (d["if"] != fold % 5)]
        def rz(g): return (g.rating - g.rating.mean()) / (g.rating.std() + 1e-6)
        tr = tr.assign(rz=tr.groupby("k", group_keys=False).apply(rz)); te = te.assign(rz=te.groupby("k", group_keys=False).apply(rz))
        per = {raters[k]: (g.i.to_numpy(), g.rz.to_numpy()) for k, g in te.groupby("k")}
        return f, (tr.i.to_numpy(), tr.rz.to_numpy()), per, itr
elif DS == "scut":   # SCUT-FBP5500: 60 raters x 5,500 faces (dense), models from 65_dense_faceid.py (random item split)
    SRC, NEST, tagf = "results/scut", 500, lambda f, s: f"D2_D8_M4_random_f{f}_s{s}"
    M_ = pd.read_excel("data/raw/scut/unzip/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL").pivot(index="Rater", columns="Filename", values="Rating"); R = M_.to_numpy(np.float32); rid = M_.index.to_numpy(); N, Fn_ = R.shape
    fid = np.random.default_rng(0).permutation(N) % 10; ffid = np.random.default_rng(1).permutation(Fn_) % 5; abstract = None
    def fold_data(fold, sd):
        f = np.load(f"{SRC}/items_{tagf(fold, 0)}.npy"); ftr = np.where(ffid != fold % 5)[0]
        tr = np.where(fid != fold)[0]; te = np.where(fid == fold)[0]
        Z = (R[:, ftr] - R[:, ftr].mean(1, keepdims=True)) / (R[:, ftr].std(1, keepdims=True) + 1e-6)
        per = {int(rid[k]): (ftr, Z[k]) for k in te}
        return f, (np.tile(ftr, len(tr)), Z[tr].ravel()), per, ftr
elif DS == "jester":   # Jester: 11,067 raters x 150 jokes (61% observed), models from 65_dense_faceid.py
    SRC, NEST, tagf = "results/jester", 50, lambda f, s: f"D2_D8_M4_random_f{f}_s{s}"
    d_ = pd.read_excel("data/raw/jester/jokes_ratings.xls", header=None); R = d_.iloc[:, 1:].to_numpy(np.float32); R[R == 99] = np.nan; keep = (~np.isnan(R)).sum(1) >= 50; R = R[keep]; rid = np.where(keep)[0]; N, Fn_ = R.shape
    fid = np.random.default_rng(0).permutation(N) % 10; ffid = np.random.default_rng(1).permutation(Fn_) % 5; abstract = None
    def fold_data(fold, sd):
        f = np.load(f"{SRC}/items_{tagf(fold, 0)}.npy"); ftr = np.where(ffid != fold % 5)[0]; tr = np.where(fid != fold)[0]; te = np.where(fid == fold)[0]
        V = ~np.isnan(R[:, ftr]); R0 = np.where(V, R[:, ftr], 0.0); cnt = np.maximum(V.sum(1, keepdims=True), 1); mu = R0.sum(1, keepdims=True) / cnt; sd_ = np.sqrt((((R0 - mu) ** 2) * V).sum(1, keepdims=True) / cnt) + 1e-6; Z = np.where(V, (R0 - mu) / sd_, np.nan)
        per = {int(rid[k]): (ftr[V[k]], Z[k][V[k]]) for k in te if V[k].sum() >= 40}
        ii = np.concatenate([ftr[V[k]] for k in tr]); yy = np.concatenate([Z[k][V[k]] for k in tr]); return f, (ii, yy), per, ftr
else:
    SRC, NEST, tagf = "results/london_img" + (f"_{ENC}" if ENC else ""), 40, lambda f, s: f"D2_D8_M4_{ENC or 'clip'}_f{f}_s{s}"
    r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(np.float32); N, Fn_ = R.shape
    fid = np.random.default_rng(0).permutation(N) % 10; ffid = np.random.default_rng(1).permutation(Fn_) % 5; abstract = None
    def fold_data(fold, sd):
        f = np.load(f"{SRC}/faces_{tagf(fold, 0)}.npy"); ftr = np.where(ffid != fold % 5)[0]
        tr = np.where((fid != fold) & (R.std(1) > 0))[0]; te = np.where(fid == fold)[0]
        Z = (R[:, ftr] - R[:, ftr].mean(1, keepdims=True)) / (R[:, ftr].std(1, keepdims=True) + 1e-6)
        per = {int(k): (ftr, Z[k]) for k in te if R[k].std() > 0}
        return f, (np.tile(ftr, len(tr)), Z[tr].ravel()), per, ftr
ev = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f"{SRC}/eval_D[12]_*.csv"))], ignore_index=True)
ev = ev[ev.n_est == NEST].groupby(["kind", "rater"])[["r_amort_seen", "r_amort_unseen"]].mean().unstack(0)
gain = (ev["r_amort_seen"]["D2"] - ev["r_amort_seen"]["D1"]).rename("d2_minus_d1"); gain_u = (ev["r_amort_unseen"]["D2"] - ev["r_amort_unseen"]["D1"]).rename("d2_minus_d1_unseen")
rows, dims = [], []
rng = np.random.default_rng(0)
for fold in range(10):
    for s in range(10):
        pt = f"{SRC}/model_{tagf(fold, s)}.pt"
        if not glob.glob(pt): continue
        sd = torch.load(pt, map_location="cpu"); f, (itr_i, itr_rz), per, itr = fold_data(fold, sd)
        mu, sdv = f[itr].mean(0), f[itr].std(0); fs = (f - mu) / sdv                    # standardised basis (signs unchanged)
        w0 = sd["dec.w0"].numpy() * sdv; wpop = ridge_w(fs[itr_i], itr_rz)             # model w0 and model-free population weight, same basis
        z = pd.read_csv(f"{SRC}/eval_{tagf(fold, s)}.csv"); z = z[z.n_est == NEST].groupby("rater")[[f"z{j}" for j in range(4)]].mean()
        W1, b1, W2, b2 = (sd[k].numpy() for k in ("dec.wz.0.weight", "dec.wz.0.bias", "dec.wz.2.weight", "dec.wz.2.bias"))
        wz = gelu(z.to_numpy() @ W1.T + b1) @ W2.T + b2; wm = (sd["dec.w0"].numpy()[None] + wz) * sdv      # model-based per-rater weights
        for j, (rt, (ii, yy)) in enumerate(per.items()):
            w = ridge_w(fs[ii], yy); pm = rng.permutation(len(ii)); h1, h2 = pm[: len(ii) // 2], pm[len(ii) // 2:]
            wa, wb = ridge_w(fs[ii[h1]], yy[h1]), ridge_w(fs[ii[h2]], yy[h2]); r_free, r_d1 = cv_gap(fs[ii], yy, wpop, h1, h2)
            flip = np.sign(w) != np.sign(wpop); rel_flip = (np.sign(wa) != np.sign(wpop)) & (np.sign(wb) != np.sign(wpop)); rel_same = (np.sign(wa) == np.sign(wpop)) & (np.sign(wb) == np.sign(wpop))
            wmi = wm[z.index.get_loc(rt)] if rt in z.index else np.full(len(w), np.nan); flip_m = np.sign(wmi) != np.sign(wpop)   # reference = model-free population weight (same basis)
            row = dict(ds=DS, fold=fold, seed=s, rater=rt, n_rat=len(ii), U_free=(w[flip] ** 2).sum() / (w ** 2).sum(), U_rel=(w[rel_flip] ** 2).sum() / (w ** 2).sum(),
                       U_model=(wmi[flip_m] ** 2).sum() / (wmi ** 2).sum(), U_unb=(wa[flip] * wb[flip]).sum() / (wa * wb).sum() if (wa * wb).sum() > 0 else np.nan, cos_pop=w @ wpop / np.linalg.norm(w) / np.linalg.norm(wpop), n_flip=int(flip.sum()), n_relflip=int(rel_flip.sum()),
                       halves_r=pearsonr(wa, wb)[0], r_free=r_free, r_d1=r_d1, gap_free_d1=r_free - r_d1)
            if abstract is not None: row["abs_pref"] = yy[abstract[ii]].mean() - yy[~abstract[ii]].mean()
            rows.append(row); dims.append(dict(fold=fold, seed=s, rater=rt, pop_row=False, **{f"w{k}": w[k] for k in range(len(w))}, **{f"wm{k}": wmi[k] for k in range(len(w))}, **{f"relflip{k}": rel_flip[k] for k in range(len(w))}, **{f"relsame{k}": rel_same[k] for k in range(len(w))}))
        dims.append(dict(fold=fold, seed=s, rater=-1, pop_row=True, **{f"w{k}": wpop[k] for k in range(len(w))}, **{f"wm{k}": w0[k] for k in range(len(w))}))
Rw = pd.DataFrame(rows); Dm = pd.DataFrame(dims); Rw.to_csv(f"{out}_raters.csv", index=False); Dm.to_csv(f"{out}_dims.csv", index=False)
print(f"{DS}: {Rw.rater.nunique()} held-out raters, {len(Rw)} rater x model rows, ratings/rater median {Rw.n_rat.median():.0f}")
print("per-rater weights split-half r (model-free): mean %.3f" % Rw.halves_r.mean())
print("mean U (energy on sign-flipped dims): free %.3f | reliable-flip %.3f | noise-unbiased (split-half product) %.3f (median %.3f) | D2-model %.3f ; cos(w_i, w_pop) mean %.3f, frac<0 %.3f" % (Rw.U_free.mean(), Rw.U_rel.mean(), Rw.U_unb.mean(), Rw.U_unb.median(), Rw.U_model.mean(), Rw.cos_pop.mean(), (Rw.cos_pop < 0).mean()))
print("model-free split-half CV per rater: free linear r %.3f | D1-constrained (w_pop * s, s>=0) r %.3f | gap %+.4f (SE %.4f), raters with gap>0.02: %.3f" % (Rw.r_free.mean(), Rw.r_d1.mean(), Rw.gap_free_d1.mean(), Rw.gap_free_d1.std() / np.sqrt(len(Rw)), (Rw.gap_free_d1 > 0.02).mean()))
print("raters with >=1 reliably flipped dim: %.3f ; mean #reliable flips %.2f of %d dims" % ((Rw.n_relflip > 0).mean(), Rw.n_relflip.mean(), Dm.filter(like="relflip").shape[1]))
# per-dim summary in each fold model (basis differs across folds, so report per fold then the range)
per_dim = []
for (fold, s), g in Dm[~Dm.pop_row].groupby(["fold", "seed"]):
    pop = Dm[(Dm.fold == fold) & (Dm.seed == s) & Dm.pop_row].iloc[0]
    for k in range(8):
        per_dim.append(dict(fold=fold, seed=s, dim=k, w_pop=pop[f"w{k}"], share_pop=pop[f"w{k}"] ** 2 / sum(pop[f"w{j}"] ** 2 for j in range(8)), relflip=g[f"relflip{k}"].astype(float).mean(), relsame=g[f"relsame{k}"].astype(float).mean(),
                            flip_model=(np.sign(g[f"wm{k}"]) != np.sign(pop[f"w{k}"])).mean(), abs_r=(spearmanr(g[f"w{k}"], Rw.set_index(["fold", "seed", "rater"]).loc[list(zip(g.fold, g.seed, g.rater)), "abs_pref"])[0] if abstract is not None else np.nan)))
PD = pd.DataFrame(per_dim); PD.to_csv(f"{out}_perdim.csv", index=False)
print("per-dim (fold 0 seed 0): share of population weight energy, reliable-flip fraction, reliable-same fraction, model flip fraction" + (", corr(w_k, abstract pref)" if abstract is not None else ""))
print(PD[(PD.fold == 0) & (PD.seed == 0)].round(3).to_string(index=False))
print("across fold models: energy-weighted reliable-flip fraction mean %.3f (range %.3f-%.3f)" % tuple([(PD.share_pop * PD.relflip).groupby([PD.fold, PD.seed]).sum().mean()] + list((PD.share_pop * PD.relflip).groupby([PD.fold, PD.seed]).sum().agg(["min", "max"]))))
# relation to D2 - D1 per rater
A = Rw.groupby("rater")[["U_free", "U_rel", "U_unb", "U_model", "cos_pop", "n_relflip", "gap_free_d1"] + (["abs_pref"] if abstract is not None else [])].mean().join(gain).join(gain_u).dropna(subset=["d2_minus_d1"])
print(f"n raters with D2-D1 at n_est={NEST}: {len(A)}; mean D2-D1 seen {A.d2_minus_d1.mean():+.4f} unseen {A.d2_minus_d1_unseen.mean():+.4f}")
for c in ["U_free", "U_rel", "U_unb", "U_model", "cos_pop", "gap_free_d1"]:
    rs, ps = spearmanr(A[c], A.d2_minus_d1, nan_policy="omit"); ru, pu = spearmanr(A[c], A.d2_minus_d1_unseen, nan_policy="omit"); print(f"spearman({c}, D2-D1): seen {rs:+.3f} (p={ps:.1e}) unseen {ru:+.3f} (p={pu:.1e})")
q = pd.qcut(A.U_free.rank(method="first"), 4, labels=["Q1 (least flipped)", "Q2", "Q3", "Q4 (most flipped)"])
print(A.groupby(q, observed=True)[["U_free", "gap_free_d1", "d2_minus_d1", "d2_minus_d1_unseen"]].agg(["mean", "count"]).round(4).to_string())
A.to_csv(f"{out}_gain.csv")
