"""Basis dependence of the sign-check metrics (2026-11-12).
55_sign_check.py computes the D1-constrained readout as w = w_pop * s with ONE NON-NEGATIVE SCALE PER DIMENSION (ridge-regularised NNLS, alpha 1)
in the standardised 8-dim item basis of each D2 fold model.  The per-dimension sign constraint depends on the basis inside that 8-dim space.
Here, with the same data, the same split halves (same rng sequence as 55) and the same free ridge readout, the constrained readout is recomputed in
  model : the fold model's standardised basis (= 55; must reproduce gap_free_d1)
  rot   : N_ROT random orthogonal rotations of that basis (w_pop rotated with it; ridge is rotation-equivariant)
  pca   : principal axes of the item features (orthogonal rotation, a basis defined without the decoder)
  scalar: w = c * w_pop, c >= 0 (one scalar; invariant to any orthogonal change of basis)
and the cosine between the per-rater and the population weight is recomputed
  cos_model (= 55), cos_rot (orthogonal rotation of the standardised basis; should be identical),
  cos_rawrot (rotate the model's raw basis, then standardise each new dim: not an orthogonal map of the standardised basis),
  cos_white (whitened basis), cos_pred (correlation of the two readouts' predictions over the training items = cosine in the item-covariance metric).
The data-loading part of 55 is executed as is (no copy).  Outputs results/basis_dep_<ds>_raters.csv; summary printed (and built by 73).
Run: uv run python -I scripts/85_basis_dependence.py lapis|eva|london|scut|jester   (scut/jester: add --with openpyxl --with xlrd)
"""
import sys, glob, numpy as np, pandas as pd, torch
src = open("scripts/55_sign_check.py").read().split("\nev = pd.concat(")[0]; exec(src)     # DS, SRC, tagf, fold_data, ridge_w, d1_w (same definitions as 55)
N_ROT = 5; out = f"results/basis_dep_{DS}"
def r_(p, y): return np.corrcoef(p, y)[0, 1] if p.std() > 0 else 0.0
def gap_d1(F, y, wpop, h1, h2):        # split-half CV r of the per-dimension constrained readout in the basis of F
    return np.mean([r_(F[b_] @ d1_w(F[a_], y[a_], wpop), y[b_]) for a_, b_ in ((h1, h2), (h2, h1))])
def r_scalar(F, y, wpop, h1, h2):      # w = c * w_pop, c >= 0 (ridge on c does not change r); c = 0 -> r = 0, as for an all-zero NNLS solution
    g = F @ wpop; return np.mean([r_(g[b_], y[b_]) if g[a_] @ y[a_] > 0 else 0.0 for a_, b_ in ((h1, h2), (h2, h1))])
def cosv(a, b): return a @ b / np.linalg.norm(a) / np.linalg.norm(b)
rows = []; rng = np.random.default_rng(0); rq = np.random.default_rng(12345)
for fold in range(10):
    for s in range(10):
        pt = f"{SRC}/model_{tagf(fold, s)}.pt"
        if not glob.glob(pt): continue
        sd = torch.load(pt, map_location="cpu"); f, (itr_i, itr_rz), per, itr = fold_data(fold, sd)
        mu, sdv = f[itr].mean(0), f[itr].std(0); fs = (f - mu) / sdv; D = fs.shape[1]; wpop = ridge_w(fs[itr_i], itr_rz)
        Qs = [np.linalg.qr(rq.standard_normal((D, D)))[0] for _ in range(N_ROT)]
        lam, V = np.linalg.eigh(np.cov(fs[itr].T)); V = V[:, ::-1]; lam = lam[::-1]            # principal axes of the item features
        bases = {f"rot{j}": (fs @ Q, Q.T @ wpop) for j, Q in enumerate(Qs)}; bases["pca"] = (fs @ V, V.T @ wpop)
        fr = (f - mu) @ Qs[0]; fr = fr / fr[itr].std(0); wpop_rr = ridge_w(fr[itr_i], itr_rz)   # rotate the raw basis, then standardise
        fw = (fs @ V) / np.sqrt(lam); wpop_w = ridge_w(fw[itr_i], itr_rz)                        # whitened
        C = np.cov(fs[itr].T)
        for rt, (ii, yy) in per.items():
            w = ridge_w(fs[ii], yy); pm = rng.permutation(len(ii)); h1, h2 = pm[: len(ii) // 2], pm[len(ii) // 2:]      # same halves as 55
            r_free = np.mean([r_(fs[ii][b_] @ ridge_w(fs[ii][a_], yy[a_]), yy[b_]) for a_, b_ in ((h1, h2), (h2, h1))])
            row = dict(ds=DS, fold=fold, seed=s, rater=rt, n_rat=len(ii), r_free=r_free, gap_model=r_free - gap_d1(fs[ii], yy, wpop, h1, h2), gap_scalar=r_free - r_scalar(fs[ii], yy, wpop, h1, h2))
            for k, (Fb, wb) in bases.items(): row[f"gap_{k}"] = r_free - gap_d1(Fb[ii], yy, wb, h1, h2)
            row.update(cos_model=cosv(w, wpop), cos_rot=cosv(Qs[0].T @ w, Qs[0].T @ wpop), cos_rawrot=cosv(ridge_w(fr[ii], yy), wpop_rr), cos_white=cosv(ridge_w(fw[ii], yy), wpop_w), cos_pred=(w @ C @ wpop) / np.sqrt((w @ C @ w) * (wpop @ C @ wpop)))
            rows.append(row)
        print(f"fold {fold} seed {s} done", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{out}_raters.csv", index=False)
old = pd.read_csv(f"results/sign_check_{DS}_raters.csv") if glob.glob(f"results/sign_check_{DS}_raters.csv") else None
if old is not None and len(old) == len(R): print("reproduces 55: max |gap_model - gap_free_d1| = %.2e, max |cos_model - cos_pop| = %.2e" % (np.abs(R.gap_model.to_numpy() - old.gap_free_d1.to_numpy()).max(), np.abs(R.cos_model.to_numpy() - old.cos_pop.to_numpy()).max()))
rr = R.groupby("rater").mean(numeric_only=True); n = len(rr)
print(f"{DS}: {n} raters (rater x model rows {len(R)}); per-rater means over fold models/seeds")
for c in ["gap_model"] + [f"gap_rot{j}" for j in range(N_ROT)] + ["gap_pca", "gap_scalar"]: print("%-11s mean %+.4f (SE %.4f)  frac > 0.02: %.3f" % (c, rr[c].mean(), rr[c].std() / np.sqrt(n), (rr[c] > 0.02).mean()))
for c in ["cos_model", "cos_rot", "cos_rawrot", "cos_white", "cos_pred"]: print("%-11s mean %+.3f  frac < 0: %.3f" % (c, rr[c].mean(), (rr[c] < 0).mean()))
print("max |cos_rot - cos_model| = %.2e ; per-rater corr(gap_model, gap_rot0) = %.3f, corr(gap_model, gap_scalar) = %.3f" % (np.abs(R.cos_rot - R.cos_model).max(), np.corrcoef(rr.gap_model, rr.gap_rot0)[0, 1], np.corrcoef(rr.gap_model, rr.gap_scalar)[0, 1]))
