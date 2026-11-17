"""Summary of the simulated-participant control at the measured reliability (scripts/88; 2026-11-17).
Inputs: results/sim_recovery_<tag>/exp<e>_s<seed>_sim<D1|D2>{.json,_raters.csv}, tag = matched (noise calibrated) and matched_id (noise and size of individual differences calibrated).
Outputs: results/sim_recovery_matched_summary.csv  one row per tag x exp x seed x generating type x fitted model (columns as results/sim_recovery_n03_summary.csv plus the calibration)
         results/sim_recovery_matched_stats.csv    paired contrasts on the held-out rho over the fold-0 simulated participants: model - population (z = 0 of the same model), D2 - D1, oracle - population;
                                                   unit = participant slot with the two seeds averaged (seeds differ in the simulated individuals and the model initialisation), and each seed separately
Run: uv run python -I scripts/88b_sim_recovery_matched_summary.py
"""
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import t as tdist, ttest_1samp, wilcoxon
S, T = [], []
def st(d):
    d = d.dropna(); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n); h = tdist.ppf(0.975, n - 1) * se
    return dict(n=n, mean_diff=m, sem=se, ci95_lo=m - h, ci95_hi=m + h, t=m / se, df=n - 1, cohen_dz=m / sd, p_t=ttest_1samp(d, 0).pvalue, p_w=wilcoxon(d).pvalue)
for tag in ("matched", "matched_id"):
    for f in sorted(glob.glob(f"results/sim_recovery_{tag}/exp*_s*_sim*.json")):
        j = json.load(open(f))
        for kind in ("D1", "D2"):
            if f"hold_pop_{kind}" not in j: continue
            S.append(dict(tag=tag, exp=j["exp"], seed=j["seed"], sim=j["sim"], kind=kind, wsd=j["wsd"], noise=j["noise"], eps=j["eps"], rho_true_pop=j["target_rho"] if j["sim"] == "D1" else j["rho_d2"], eps_at_bound=bool(j["sim"] == "D2" and j["eps"] > 7.99), n_test=j["n_test"],
                          sim_splithalf=j["sim_splithalf_all"], real_splithalf=j["real_splithalf_all"], sim_splithalf_fold0=j["sim_splithalf_fold0"], real_splithalf_fold0=j["real_splithalf_fold0"],
                          sim_hold_truepop=j["sim_hold_pop_all"], real_hold_pop=j["real_hold_pop_all"], sim_hold_truepop_fold0=j["sim_hold_pop_fold0"], real_hold_pop_fold0=j["real_hold_pop_fold0"],
                          hold_pop=j[f"hold_pop_{kind}"], hold_amortized=j[f"hold_amortized_{kind}"], hold_direct=j[f"hold_direct_{kind}"], hold_oracle=j["hold_oracle"], true_pop=j[f"true_pop_{kind}"], true_amortized=j[f"true_amortized_{kind}"], true_direct=j[f"true_direct_{kind}"]))
    fs = sorted(glob.glob(f"results/sim_recovery_{tag}/exp*_s*_sim*_raters.csv"))
    if not fs: continue
    Rr = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for (e, sim), g in Rr.groupby(["exp", "sim"]):
        num = [c for c in g.columns if c.startswith("hold_")]
        for unit, gg in [("slot, seeds averaged", g.groupby("slot")[num].mean())] + [(f"seed {s}", x.set_index("slot")[num]) for s, x in g.groupby("seed")]:
            for name, d in (("D1 model - population", gg.hold_amortized_D1 - gg.hold_pop_D1), ("D2 model - population", gg.hold_amortized_D2 - gg.hold_pop_D2), ("D2 - D1 (amortised)", gg.hold_amortized_D2 - gg.hold_amortized_D1),
                            ("D2 - D1 (direct)", gg.hold_direct_D2 - gg.hold_direct_D1), ("oracle - population (D1 model's)", gg.hold_oracle - gg.hold_pop_D1)):
                T.append(dict(tag=tag, exp=e, sim=sim, unit=unit, contrast=name, **st(d)))
S, T = pd.DataFrame(S), pd.DataFrame(T); S.to_csv("results/sim_recovery_matched_summary.csv", index=False); T.to_csv("results/sim_recovery_matched_stats.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
print(S.groupby(["tag", "exp", "sim", "kind"])[["wsd", "noise", "eps", "eps_at_bound", "rho_true_pop", "sim_splithalf", "real_splithalf", "sim_hold_truepop", "real_hold_pop", "hold_pop", "hold_amortized", "hold_direct", "hold_oracle", "true_pop", "true_amortized"]].mean().round(4).to_string())
print(T[T.unit == "slot, seeds averaged"].drop(columns=["unit"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
