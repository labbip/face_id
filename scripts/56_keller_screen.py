"""Screen Keller & Vosshall (2016) olfactory psychophysics (BMC Neurosci; data/raw/keller/*.xlsx, 55 subjects x 480 odorants x 2 dilutions,
20 odorants repeated) for Face-ID suitability: variance decomposition of pleasantness (odorant main effect, subject main effect, residual),
test-retest of residual (idiosyncratic) preferences from the repeated odorants, and split-half stability of per-subject residual profiles.
Run: uv run --with openpyxl python -I scripts/56_keller_screen.py
"""
import numpy as np, pandas as pd
w = pd.read_excel("data/raw/keller/keller_vosshall_2016_MOESM1.xlsx", header=2)
w.columns = [str(c).strip() for c in w.columns]; w = w.rename(columns={"Subject # (this study)": "subj", "Odor dilution": "dil", "HOW PLEASANT IS THE SMELL?": "pleas", "HOW STRONG IS THE SMELL?": "inten"})
w["pleas"] = pd.to_numeric(w.pleas, errors="coerce"); w["inten"] = pd.to_numeric(w.inten, errors="coerce")
print("rows", len(w), "subjects", w.subj.nunique(), "CIDs", w.CID.nunique(), "dilutions", w.dil.value_counts().to_dict())
print("smell-detected fraction:", (w["CAN OR CAN'T SMELL"] == "I smell something").mean().round(3), "| pleasantness NaN:", w.pleas.isna().mean().round(3))
for dil, g in w.groupby("dil"):
    g = g.dropna(subset=["pleas"]); rep = g.groupby(["subj", "CID"]).pleas.agg(["mean", "count", list])
    rr = rep[rep["count"] >= 2]; a = np.array([x[0] for x in rr["list"]]); b = np.array([x[1] for x in rr["list"]])
    M = rep["mean"].unstack()                                        # subject x odorant
    Mc = M - M.mean(0).to_numpy()[None] - M.mean(1).to_numpy()[:, None] + np.nanmean(M.to_numpy())
    v_tot = np.nanvar(M.to_numpy()); v_odor = np.nanvar(M.mean(0)); v_subj = np.nanvar(M.mean(1)); v_res = np.nanvar(Mc.to_numpy())
    # residual test-retest from repeated odorants (same subject, same odorant, two presentations)
    ia = a - M.mean(0).reindex(rr.index.get_level_values(1)).to_numpy() - M.mean(1).reindex(rr.index.get_level_values(0)).to_numpy(); ib = b - M.mean(0).reindex(rr.index.get_level_values(1)).to_numpy() - M.mean(1).reindex(rr.index.get_level_values(0)).to_numpy()
    r_raw = np.corrcoef(a, b)[0, 1]; r_res = np.corrcoef(ia, ib)[0, 1]
    # split-half (over odorants) stability of each subject's residual profile, correlated across subjects via odorant halves
    rng = np.random.default_rng(0); cols = M.columns.to_numpy(); sh = []
    for _ in range(20):
        p = rng.permutation(len(cols)); h1, h2 = cols[p[: len(p) // 2]], cols[p[len(p) // 2:]]
        # subject-pair similarity of residual profiles computed on half 1 vs half 2
        S1 = np.corrcoef(np.nan_to_num(Mc[h1].to_numpy())); S2 = np.corrcoef(np.nan_to_num(Mc[h2].to_numpy())); iu = np.triu_indices(len(S1), 1); sh.append(np.corrcoef(S1[iu], S2[iu])[0, 1])
    print(f"dilution {dil}: matrix {M.shape}, missing {np.isnan(M.to_numpy()).mean():.3f}; variance share odorant {v_odor / v_tot:.3f} subject {v_subj / v_tot:.3f} residual {v_res / v_tot:.3f}; "
          f"repeat pairs {len(rr)}: test-retest raw r {r_raw:.3f}, residual r {r_res:.3f}; split-half stability of between-subject residual similarity {np.mean(sh):.3f}")
