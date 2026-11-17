"""Figures for the sign check (results/sign_check_{lapis,london}_*.csv from 55_sign_check.py).
sign_check_gap.pdf: per-rater split-half CV gain of a free linear readout over the D1-constrained readout (w_pop * s, s >= 0), LAPIS vs London.
sign_check_quartile.pdf: amortized D2 - D1 by quartile of the rater's weight energy on sign-flipped dimensions.
Each panel is drawn from the table of its plotted values (histogram densities per bin, quartile means), written to results/figdata/<panel>.csv;
FIGDATA=read redraws the panels from those tables without the per-rater files.
Run: uv run python -I scripts/57_figures_sign_check.py      (FIGDATA=read uv run python -I scripts/57_figures_sign_check.py)
"""
import numpy as np, pandas as pd
import os
from face_id.plotting import plt, esc, TAB10
from face_id import plotting
ONLY = os.environ.get("ONLY")                      # ONLY=<substring>: write only matching panels
def savefig(fig, name, *a, **k):                     # a = (size in pt,) -> fixed canvas
    if ONLY and ONLY not in name: return plt.close(fig)
    return plotting.savefig_fixed(fig, name, *a, **k) if a else plotting.savefig(fig, name, **k)
BINS = np.linspace(-0.3, 0.4, 36)
def gap_table(dss):
    """Histogram of the per-rater loss (clipped to the bin range, density) per dataset; mean of the clipped values (legend) and SD of the raw values (drawing order)."""
    def table():
        out = []
        for ds in dss:
            raw = pd.read_csv(f"results/sign_check_{ds}_raters.csv").groupby("rater").mean(numeric_only=True).gap_free_d1; g = raw.clip(BINS[0], BINS[-1]); h, _ = np.histogram(g, bins=BINS, density=True)
            out.append(pd.DataFrame(dict(dataset=ds, bin_left=BINS[:-1], bin_right=BINS[1:], density=h, n_raters=len(g), mean=g.mean(), sd_raw=raw.std())))
        return pd.concat(out, ignore_index=True)
    return table
def quart_table(dss, rank):
    def table():
        out = []
        for ds in dss:
            A = pd.read_csv(f"results/sign_check_{ds}_gain.csv", index_col=0); q = pd.qcut(A.U_free.rank(method="first") if rank else A.U_free, 4, labels=False)
            out.append(A.groupby(q).d2_minus_d1.agg(["mean", "sem", "count"]).rename_axis("quartile").reset_index().assign(dataset=ds))
        return pd.concat(out, ignore_index=True)[["dataset", "quartile", "mean", "sem", "count"]]
    return table
def step_hist(ax, T, ds, **k):      # the same outline as ax.hist(values, bins=BINS, density=True, histtype="step"), from the saved densities
    t = T[T.dataset == ds]; return ax.hist((t.bin_left + t.bin_right) / 2, bins=BINS, weights=t.density, histtype="step", **k)[2][0]      # bin centres as x: an edge value can fall into the neighbouring bin by rounding
lab = {"lapis": "LAPIS artworks", "london": "London Set faces"}
T = plotting.figdata("sign_check_gap", gap_table(("lapis", "london"))); fig, ax = plt.subplots()
for i, ds in enumerate(("lapis", "london")): step_hist(ax, T, ds, color=TAB10[i], lw=1.2, label=esc(f"{lab[ds]} ({T[T.dataset == ds]['mean'].iloc[0]:+.3f})"))
ax.axvline(0, color="k", lw=0.6, ls=":"); ax.set_xlabel(esc("Free linear readout minus D1-constrained readout (split-half CV r)")); ax.set_ylabel("Density"); ax.set_ylim(0, 19.5); ax.legend(frameon=False, loc="upper right", handlelength=1.2)
savefig(fig, "sign_check_gap", w=3.4, h=2.6, index_line="Sign check: per-rater split-half cross-validated gain (Pearson r) of a free linear readout over a readout constrained to same-sign reweighting of the population direction (w_pop * s, s >= 0; both ridge-regularised, fitted on each rater's own ratings in the fold model's embedding basis), LAPIS artworks (553 raters, median 383 ratings) vs London Set faces (2,501 raters, 82 faces); values in results/sign_check_{lapis,london}_raters.csv")
Q = plotting.figdata("sign_check_quartile", quart_table(("lapis", "london"), rank=False)); fig, ax = plt.subplots(); x = np.arange(4)
for i, ds in enumerate(("lapis", "london")):
    m = Q[Q.dataset == ds].sort_values("quartile")
    ax.errorbar(x + (i - 0.5) * 0.12, m["mean"], yerr=m["sem"], fmt="o-", color=TAB10[i], ms=4, lw=1, capsize=2, label=esc(lab[ds]))
ax.axhline(0, color="k", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels(["Q1 (least)", "Q2", "Q3", "Q4 (most)"]); ax.set_xlabel(esc("Quartile of weight energy on sign-flipped dimensions")); ax.set_ylabel(esc("D2 $-$ D1 (held-out r)")); ax.legend(frameon=False, loc="upper left", handlelength=1.5)
savefig(fig, "sign_check_quartile", w=3.4, h=2.6, index_line="Sign check: per-rater amortized D2 - D1 gain (held-out ratings of training stimuli, LAPIS 100 / London 40 estimation ratings, seeds averaged) by quartile of the rater's model-free weight energy on dimensions whose sign is opposite to the population weight (mean ± s.e.m.); LAPIS Q1 +0.009 to Q4 +0.035, London all ≈ 0; values in results/sign_check_{lapis,london}_gain.csv")
print("done")

# four-dataset versions (faces: London, SCUT; photographs: EVA; artworks: LAPIS)
lab4 = {"lapis": "LAPIS artworks", "eva": "EVA photographs", "london": "London Set faces", "scut": "SCUT-FBP5500 faces"}
# request 6 (2026-11-17): legend above the axes (it covered the peak at 0), fixed canvas = page size of the existing PDF; outlines are drawn from the widest
# distribution (back, thickest line) to the narrowest (front, thinnest) so that every outline stays visible near 0. Same bins, colours, legend order and y limit.
size = plotting.pdf_size("sign_check_gap_all") or (274.421, 188.224)
AXR = (40.0, 32.0, 196.0, 126.0)   # axes rectangle in pt (left, bottom, width, height): centred so that the long x label (wider than the axes) fits on the canvas
fig = plt.figure(figsize=(size[0] / 72, size[1] / 72)); ax = fig.add_axes([AXR[0] / size[0], AXR[1] / size[1], AXR[2] / size[0], AXR[3] / size[1]])
dss = ("lapis", "eva", "london", "scut"); T = plotting.figdata("sign_check_gap_all", gap_table(dss)); st = lambda ds, c: T[T.dataset == ds][c].iloc[0]
back = sorted(dss, key=lambda ds: -st(ds, "sd_raw")); LW = dict(zip(back, (1.8, 1.5, 1.2, 0.9))); H = {}
for i, ds in enumerate(dss): H[ds] = step_hist(ax, T, ds, color=TAB10[i], lw=LW[ds], zorder=2 + back.index(ds), label=esc(f"{lab4[ds]} ({st(ds, 'mean'):+.3f})"))
ax.axvline(0, color="k", lw=0.6, ls=":", zorder=1); ax.set_ylim(0, 20); ax.set_xlabel(esc("Free linear readout minus D1-constrained readout (split-half CV r)")); ax.set_ylabel("Density")
fig.legend([H[ds] for ds in dss], [H[ds].get_label() for ds in dss], frameon=False, loc="upper center", bbox_to_anchor=((AXR[0] + AXR[2] / 2) / size[0], 1 - 3 / size[1]), handlelength=1.2, ncol=2, columnspacing=1.0, labelspacing=0.2, borderaxespad=0, borderpad=0)
savefig(fig, "sign_check_gap_all", size, index_line="Sign check, four datasets: per-rater split-half cross-validated gain (Pearson r) of a free linear readout over the D1-constrained readout (w_pop * s, s >= 0), fitted on each rater's own ratings in the fold model's embedding basis: LAPIS artworks (553 raters, median 383 ratings), EVA photographs (175 raters, median 256), London Set faces (2,501 raters, 82 faces), SCUT-FBP5500 faces (60 raters, 4,400 faces); legend (above the axes) gives the mean; the SCUT peak (density 38 at 0) is truncated at 20; outlines drawn from the widest distribution (back, line width " + ", ".join(f"{lab4[ds].split()[0]} {LW[ds]}" for ds in back) + " pt) to the narrowest (front); values in results/sign_check_{lapis,eva,london,scut}_raters.csv")
DX = 0.15   # request 6: shift between neighbouring series in quartile units (was 0.10), about 7.5 pt; marker 3.5 pt (was 4), cap width 4 pt
Q = plotting.figdata("sign_check_quartile_all", quart_table(dss, rank=True)); fig, ax = plt.subplots(); x = np.arange(4)
for i, ds in enumerate(dss):
    m = Q[Q.dataset == ds].sort_values("quartile")
    ax.errorbar(x + (i - 1.5) * DX, m["mean"], yerr=m["sem"], fmt="o-", color=TAB10[i], ms=3.5, lw=1, capsize=2, label=esc(lab4[ds]))
ax.axhline(0, color="k", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels(["Q1 (least)", "Q2", "Q3", "Q4 (most)"]); ax.set_xlabel(esc("Quartile of weight energy on sign-flipped dimensions")); ax.set_ylabel(esc("D2 $-$ D1 (held-out r)")); ax.legend(frameon=False, loc="upper left", handlelength=1.5)
savefig(fig, "sign_check_quartile_all", w=3.4, h=2.6, index_line="Sign check, four datasets: per-rater amortized D2 - D1 gain (held-out ratings of training stimuli; estimation ratings LAPIS 100, EVA 100, London 40, SCUT 500; seeds averaged) by quartile of the rater's model-free weight energy on dimensions whose sign is opposite to the population weight (mean ± s.e.m.); LAPIS and EVA rise with the flipped energy, the two face datasets stay near 0; the four series are shifted horizontally by 0.15 quartile units each for visibility (same values); values in results/sign_check_{lapis,eva,london,scut}_gain.csv")
print("done all")
