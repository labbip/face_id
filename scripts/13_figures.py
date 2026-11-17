"""Paper figures of the Xu et al. experiments (one panel per PDF): methods_exp{1,2}_tr_unseen, learning_curve_exp{1,2},
reliability_strata_exp{1,2}, cross_individual_*. usage: uv run python -I scripts/13_figures.py --tag _cv
Since 2026-11-17: (1) figures/INDEX.md is no longer rewritten from scratch (each panel's line is replaced in place); (2) methods_exp{1,2}_hold.pdf are
not drawn here any more (their paper versions, full width with grouped rows, are drawn by scripts/46_figures_summary.py; this script used to overwrite
them with the old layout); (3) every panel is drawn from the table of its plotted values, written to results/figdata/<panel>.csv, and
FIGDATA=read redraws the panels from those tables without the per-participant files."""
import argparse, os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, FIGURES
from face_id.plotting import plt, savefig, esc, TAB10
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_cv"); args = ap.parse_args()
os.makedirs(FIGURES, exist_ok=True)
ONLY = os.environ.get("ONLY")                      # ONLY=<substring>: write only matching panels
from face_id import plotting
_savefig = savefig
def savefig(fig, name, *a, **k):                     # a = (size in pt,) -> fixed canvas (learning curves)
    if ONLY and ONLY not in name: return plt.close(fig)
    return plotting.savefig_fixed(fig, name, *a, **k) if a else _savefig(fig, name, **k)
LABEL = {"agg": "Population RDM", "mds5": "5-D MDS", "estrdm": "Own measured RDM", "blend": "Shrinkage blend", "rw5": "5-D reweighting",
         "D0": "D0 (shared only)", "D1": "D1 (reweighting)", "D2": "D2 (linear transform)", "D3": "D3 (+ face offsets)"}

def pooled(exp):
    """per-participant table: columns = method -> metric values (seeds averaged)."""
    import json as _json
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{exp}_D*_D8_M*_f*_s*{args.tag}"))):
        if not os.path.exists(os.path.join(d, "eval.csv")): continue
        a = _json.load(open(os.path.join(d, "args.json"))); e = pd.read_csv(os.path.join(d, "eval.csv")); e = e[e.frac_est == 0.7].copy()
        e["method"] = a["kind"] + "-M" + str(a["M"]) + "-" + e.infer; rows.append(e[["pid", "method", "rho_hold", "rho_tr_unseen"]])
    ev = pd.concat(rows).groupby(["pid", "method"]).mean().reset_index()
    bl = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, "baselines", f"exp{exp}_f*.csv"))])
    for b in ["agg", "mds5", "estrdm", "blend", "rw5"]:
        ev = pd.concat([ev, bl[["pid", f"rho_hold_{b}", f"rho_tr_unseen_{b}"]].rename(columns={f"rho_hold_{b}": "rho_hold", f"rho_tr_unseen_{b}": "rho_tr_unseen"}).assign(method=b)])
    return ev

def fig_methods(exp, metric, fname):
    def table():
        sm = pd.read_csv(os.path.join(RESULTS, f"summary_exp{exp}{args.tag}.csv")); sem = pooled(exp).groupby("method")[metric].sem()
        return pd.DataFrame(dict(method=sm.method, value=sm[metric], sem=[sem.get(n, 0) for n in sm.method]))
    T = plotting.figdata(fname, table); names = list(T.method); val = dict(zip(T.method, T.value)); sem = dict(zip(T.method, T["sem"]))
    order = [n for n in names if not n.startswith("D")] + [n for n in names if n.startswith("D") and "amortized" in n] + [n for n in names if n.startswith("D") and "direct" in n]
    fig, ax = plt.subplots()
    y = np.arange(len(order))
    for i, n in enumerate(order):
        c = TAB10[0] if not n.startswith("D") else (TAB10[1] if "amortized" in n else TAB10[2])
        ax.errorbar(val[n], i, xerr=sem[n], fmt="o", color=c, ms=4, capsize=2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([esc(LABEL.get(n.split("-")[0], n.split("-")[0]) + ("" if not n.startswith("D") else f" M={n.split('-')[1][1:]}, {n.split('-')[2]}")) for n in order])
    ax.invert_yaxis(); ax.set_xlabel(esc({"rho_hold": "Spearman rho, held-out arrangements", "rho_tr_unseen": "Spearman rho, test-retest (unseen faces)"}[metric]))
    savefig(fig, fname, w=3.6, h=0.22 * len(order) + 0.8, index_line=f"Exp{exp} {metric}: mean ± s.e.m. over fold-out participants (seeds averaged) per method; values in results/summary_exp{exp}{args.tag}.csv")

FD = lambda name: os.path.exists(os.path.join(RESULTS, "figdata", name + ".csv"))        # FIGDATA=read: draw the panels whose tables exist
for exp in (1, 2):
    if FD(f"methods_exp{exp}_tr_unseen") if plotting.FROM_FIGDATA else os.path.exists(os.path.join(RESULTS, f"summary_exp{exp}{args.tag}.csv")):
        fig_methods(exp, "rho_tr_unseen", f"methods_exp{exp}_tr_unseen")      # methods_exp{exp}_hold: scripts/46

# learning curves (request 6, 2026-11-17: series at the same x shifted by STEP pt each, legend above the axes, fixed canvas = page size of the existing PDF;
# points are joined in order of x. Before, the all-trials point (mean n about 34, between 20 and 40) came first in the line, which drew a spurious segment back to x = 3.)
STEP = 4.0
for exp in (1, 2):
    fname = f"learning_curve_exp{exp}"; files = sorted(glob.glob(os.path.join(RESULTS, f"learning_curve_exp{exp}_*{args.tag}_s*.csv")))
    if not (FD(fname) if plotting.FROM_FIGDATA else files): continue
    def table():
        out = []
        for ci, f in enumerate(files):
            d = pd.read_csv(f); parts_ = os.path.basename(f).replace(".csv", "").split("_"); kind = parts_[3]; M = parts_[4]
            for inf, g in d.groupby("infer"):
                gg = g.groupby("n_est_req").rho_hold.agg(["mean", "sem", "count"]).reset_index(); gg["x"] = gg.n_est_req.where(gg.n_est_req > 0, g[g.n_est_req < 0].n_est.mean())
                out.append(gg.sort_values("x").assign(file_index=ci, kind=kind, M=M, inference=inf))
        s_ = pd.read_csv(os.path.join(RESULTS, f"summary_exp{exp}{args.tag}.csv"))
        return pd.concat(out + [pd.DataFrame([dict(kind="D0", inference="population (dotted line)", mean=s_[s_.method == "D0-M4-amortized"].rho_hold.iloc[0])])], ignore_index=True)[["file_index", "kind", "M", "inference", "n_est_req", "x", "mean", "sem", "count"]]
    T = plotting.figdata(fname, table); pop = T[T.kind == "D0"]["mean"].iloc[0]; T = T[T.kind != "D0"]
    size = plotting.pdf_size(fname) or (3.4 * 72, 2.6 * 72)
    fig, ax = plt.subplots(layout="constrained"); fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72); fig.set_size_inches(size[0] / 72, size[1] / 72)
    ser = [(gg, "o-" if inf == "amortized" else "s--", TAB10[int(ci) % 10], esc(f"{gg.kind.iloc[0]} {inf}")) for (ci, inf), gg in T.groupby(["file_index", "inference"], sort=True)]
    for k, (gg, f_, c, lab) in enumerate(ser):
        ax.errorbar(gg.x, gg["mean"], gg["sem"], fmt=f_, color=c, ms=3, capsize=1.5, elinewidth=1, capthick=1, label=lab, transform=plotting.dodge(ax, k, len(ser), STEP))
    ax.axhline(pop, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20, 40]); ax.set_xticklabels(["3", "5", "10", "20", "40"]); ax.minorticks_off()
    # limits by hand (shifted artists are not autoscaled): y = default 5% margin around data incl. error bars and the dotted line (as before); x = 9% margin in log units (was 5%)
    lo = min([(g_["mean"] - g_["sem"]).min() for g_, *_ in ser] + [pop]); hi = max([(g_["mean"] + g_["sem"]).max() for g_, *_ in ser] + [pop]); ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    l0, l1 = np.log10(min(g_.x.min() for g_, *_ in ser)), np.log10(max(g_.x.max() for g_, *_ in ser)); ax.set_xlim(10 ** (l0 - 0.09 * (l1 - l0)), 10 ** (l1 + 0.09 * (l1 - l0)))
    nall = np.mean([g_.x[g_.n_est_req < 0].iloc[0] for g_, *_ in ser])
    ax.set_xlabel("Number of estimation trials"); ax.set_ylabel(r"Spearman $\rho$, held-out arrangements")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, loc="outside upper center", handlelength=1.5, ncol=2, columnspacing=0.8, labelspacing=0.2, borderaxespad=0)
    savefig(fig, fname, size, index_line=f"Exp{exp} learning curve (fixed 30% held-out set), mean ± s.e.m. over fold-out participants; the point at x = {nall:.0f} = all estimation trials (mean n; the points at 40 use at most 40 trials); dotted = D0 population model; series at the same x are shifted horizontally by {STEP:g} pt each for visibility (same values), legend above the axes")

# reliability strata
for exp in (1, 2):
    f = os.path.join(RESULTS, f"reliability_strata_exp{exp}{args.tag}.csv")
    if not os.path.exists(f): continue
    d = pd.read_csv(f); best = "D1-M4" if exp == 2 else "D2-M4"
    fig, ax = plt.subplots(); x = np.arange(4)
    for i, (mth, lab, mk) in enumerate([("agg", "Population RDM", "o"), ("D0-M4", "D0", "s"), (best, best.split("-")[0], "^"), ("blend", "Shrinkage blend", "v")]):
        g = d[d.method == mth].sort_values("stratum"); ax.plot(x, g.rho_hold, marker=mk, color=TAB10[i], label=esc(lab), ms=4)
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"]); ax.set_xlabel("Split-half reliability quartile (Q4 = most reliable)"); ax.set_ylabel(r"Spearman $\rho$, held-out arrangements")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, f"reliability_strata_exp{exp}", index_line=f"Exp{exp} held-out rho by participant reliability quartile for population RDM, D0, {best} and shrinkage blend (values in results/reliability_strata_exp{exp}{args.tag}.csv)")

# cross-individual
CI = sorted(glob.glob(os.path.join(RESULTS, "figdata" if plotting.FROM_FIGDATA else "", "cross_individual_*.csv")))
for f in CI:
    name = os.path.basename(f).replace(".csv", "")
    def table():
        d = pd.read_csv(f); d["bin"] = pd.qcut(d.dz, 10, labels=False)
        return d.groupby("bin").agg(dz=("dz", "mean"), rho=("rho", "mean"), sem=("rho", "sem"), n_pairs=("rho", "size")).reset_index().assign(population_mean=d.popr.mean(), own_z_mean=d.own.mean())
    g = plotting.figdata(name, table)
    fig, ax = plt.subplots(); ax.errorbar(g.dz, g.rho, g["sem"], fmt="o", color=TAB10[0], ms=3)
    ax.axhline(g.population_mean.iloc[0], color="k", ls=":", lw=0.8); ax.axhline(g.own_z_mean.iloc[0], color=TAB10[1], ls="--", lw=0.8)
    ax.set_xlabel(r"$\|z_k - z_l\|$ (decile mean)"); ax.set_ylabel(r"Spearman $\rho$ (other's $z$)")
    savefig(fig, name, index_line=f"H3: transfer rho vs z distance (deciles, mean ± sem); dotted = population, dashed = own z; from {name}.csv")
