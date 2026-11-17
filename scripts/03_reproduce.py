"""Reproduce the paper's Exp1 population-level quantities from trial-level data:
  - agg-RDM (two-level inverse MDS: per participant, then across participants)
  - test-retest reliability (mean within-set Spearman among the 3 repetitions)
  - split-half intra-participant consistency (n_split random halves of trials)
Results -> results/repro_exp{EXP}_*.csv ; comparison with the authors' files when present.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RAW, PROC, RESULTS
from face_id.invmds import inverse_mds, pair_lookup, trial_distances
from face_id.rdm import nan_spearman, rms_to_one

import argparse; EXP = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = 100 if EXP == 1 else 96
LUT = pair_lookup(N)
P = N * (N - 1) // 2
pl = pd.read_parquet(os.path.join(PROC, f"exp{EXP}_placements.parquet"))
part = pd.read_csv(os.path.join(PROC, f"exp{EXP}_participants.csv"), low_memory=False)
keep = set(part[part.keep].pid)
ma = pl[(pl.task == "MA") & (~pl.excluded) & pl.pid.isin(keep)]
print("valid MA trials:", ma.groupby(["pid", "trial"]).ngroups, "participants", ma.pid.nunique())


def trial_matrix(g):
    """rows = trials of one participant, columns = all pairs (NaN unobserved)."""
    rows = []
    for t, tg in g.groupby("trial"):
        idx = tg.face_idx.to_numpy(); xy = tg[["x", "y"]].to_numpy()
        if len(idx) < 2: continue
        d = trial_distances(xy)
        ii, jj = np.triu_indices(len(idx), k=1)
        row = np.full(P, np.nan); row[LUT[idx[ii], idx[jj]]] = d
        rows.append(row)
    return np.array(rows)

rng = np.random.default_rng(0)
n_split = 20
ind_rdms, rel = [], []
for pid, g in tqdm(ma.groupby("pid")):
    Dm = trial_matrix(g)
    rdm, w, it = inverse_mds(Dm)
    ind_rdms.append(rdm)
    # split-half
    T = len(Dm); half = T // 2; rhos = []
    for _ in range(n_split):
        perm = rng.permutation(T)
        r1, _, _ = inverse_mds(Dm[perm[:half]]); r2, _, _ = inverse_mds(Dm[perm[half:]])
        rhos.append(nan_spearman(r1, r2))
    rho = np.nanmean(rhos)
    rel.append({"pid": pid, "n_valid_trials": T, "split_half_rho": rho,
                "split_half_rho_sb": 2 * rho / (1 + rho)})
ind = np.array(ind_rdms)
rel = pd.DataFrame(rel)
agg, w, it = inverse_mds(ind)
agg_m = np.zeros((N, N)); iu = np.triu_indices(N, k=1); agg_m[iu] = agg; agg_m += agg_m.T
np.save(os.path.join(RESULTS, f"repro_exp{EXP}_aggRDM_v.npy"), agg)
np.save(os.path.join(RESULTS, f"repro_exp{EXP}_individual_RDMs.npy"), ind)
pd.Series(sorted(keep)).to_csv(os.path.join(RESULTS, f"repro_exp{EXP}_individual_RDMs_pids.csv"), index=False, header=["pid"])

# leave-one-out consensus: each participant's 20-face RDM vs agg of all others
# (approximate the 'all others' aggregate by the full aggregate; exact LOO is costly)
cons = [nan_spearman(ind[i], agg) for i in range(len(ind))]
rel["consensus_vs_agg"] = cons

# test-retest from TR trials (within-set correlation over 3 reps; mean of 15 pairs)
tr = pl[pl.task.str.startswith("TR_") & pl.pid.isin(keep)]
trr = []
for pid, g in tr.groupby("pid"):
    cs = []
    for s in range(1, 6 if EXP == 1 else 4):
        reps = []
        for rep in range(1, 4):
            tg = g[g.task == f"TR_set{s}_rep{rep}"].sort_values("face")
            if len(tg) < 3: reps.append(None); continue
            reps.append(trial_distances(tg[["x", "y"]].to_numpy()))
        for a in range(3):
            for b in range(a + 1, 3):
                if reps[a] is not None and reps[b] is not None:
                    cs.append(spearmanr(reps[a], reps[b]).statistic)
    trr.append({"pid": pid, "tr_rho": np.nanmean(cs)})
trr = pd.DataFrame(trr)
rel = rel.merge(trr, on="pid", how="left")
rel.to_csv(os.path.join(RESULTS, f"repro_exp{EXP}_reliability.csv"), index=False)
print("split-half rho mean %.3f (SB %.3f); test-retest rho mean %.3f (n=%d; paper 0.51)" % (
    rel.split_half_rho.mean(), rel.split_half_rho_sb.mean(), rel.tr_rho.mean(), rel.tr_rho.notna().sum()))
print("consensus (indiv vs agg) mean %.3f" % rel.consensus_vs_agg.mean())
print("corr split-half vs test-retest: %.3f" % rel[["split_half_rho", "tr_rho"]].corr(method="spearman").iloc[0, 1])

# compare with authors' agg-RDM if available
f = os.path.join(RAW, "analysis_code_and_data/results_Experiment1/rsatoolboxRDM_v.csv")
if EXP == 1 and os.path.exists(f):
    ref = pd.read_csv(f, header=None).to_numpy().ravel()
    print("agg-RDM vs authors': spearman %.4f pearson %.4f maxabs %.2e" % (
        spearmanr(agg, ref).statistic, np.corrcoef(agg, ref)[0, 1], np.max(np.abs(rms_to_one(agg) - rms_to_one(ref)))))
f = os.path.join(RAW, "analysis_code_and_data/results_Experiment1/stage2_alldata_reliability_with_null_test_df_v2_20240805.csv")
if EXP == 1 and os.path.exists(f):
    ref = pd.read_csv(f, index_col=0)
    m = rel.set_index("pid").join(ref, how="inner")
    print("split-half vs authors' (n=%d): r=%.3f, means %.3f vs %.3f" % (len(m), m[["split_half_rho", "spearman"]].corr().iloc[0, 1], m.split_half_rho.mean(), m.spearman.mean()))
