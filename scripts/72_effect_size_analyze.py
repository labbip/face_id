"""Aggregate the effect-size views (results/<src>/eval2_*.csv from 53/65 --eval_only; London retrained in 65 -> results/london65):
  all-item r (from eval_*.csv) vs contested-item r (top-tercile disagreement items), residual r, contested-pair accuracy; pop vs D1 vs D2.
Writes results/effect_size_summary.csv and figures effect_contested.pdf, effect_pair_accuracy.pdf. Run: uv run python -I scripts/72_effect_size_analyze.py
"""
import glob, numpy as np, pandas as pd
from scipy.stats import ttest_rel
from face_id.plotting import plt, esc, TAB10, _index, FROM_FIGDATA
SETS = [  # label, results dir, n_est values to report (last = used in figures)
    ("London Set (faces)", "london65", [5, 20, 40]), ("SCUT-FBP5500 (faces)", "scut", [20, 100, 500]), ("AADB (photographs)", "aadb_img", [20, 50, 100]),
    ("EVA (photographs)", "eva_img", [20, 50, 100]), ("LAPIS (artworks)", "lapis_img", [20, 50, 100]), ("Jester (jokes)", "jester", [10, 20, 50])]
rows = []
for lab, src, nests in ([] if FROM_FIGDATA else SETS):      # FIGDATA=read: the two panels are redrawn from results/effect_size_summary.csv (their source data)
    e2 = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f"results/{src}/eval2_*.csv"))], ignore_index=True)
    e1 = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f"results/{src}/eval_*.csv"))], ignore_index=True)
    e1 = e1[e1.kind.isin(["D0", "D1", "D2"])]; e2 = e2[e2.kind.isin(["D0", "D1", "D2"])]
    for n in nests:
        g2 = e2[e2.n_est == n].groupby(["kind", "rater"]).mean(numeric_only=True); g1 = e1[e1.n_est == n].groupby(["kind", "rater"]).mean(numeric_only=True)
        for side in ("seen", "unseen"):
            rec = dict(dataset=lab, src=src, n_est=n, side=side)
            P1 = g1.unstack(0); P2 = g2.unstack(0)
            def col(P, c, k): return P[c][k] if (c in P.columns.get_level_values(0) and k in P[c].columns) else None
            allpop = col(P1, f"r_pop_{side}", "D0"); d1 = col(P1, f"r_amort_{side}", "D1"); d2 = col(P1, f"r_amort_{side}", "D2"); d0 = col(P1, f"r_amort_{side}", "D0")
            hipop = col(P2, f"rhi_pop_{side}", "D0"); h1 = col(P2, f"rhi_amort_{side}", "D1"); h2 = col(P2, f"rhi_amort_{side}", "D2")
            rr1 = col(P2, f"rr_amort_{side}", "D1"); rr2 = col(P2, f"rr_amort_{side}", "D2"); rr0 = col(P2, f"rr_amort_{side}", "D0")
            pp = col(P2, f"pacc_pop_{side}", "D0"); p1 = col(P2, f"pacc_amort_{side}", "D1"); p2 = col(P2, f"pacc_amort_{side}", "D2")
            def ms(x): return (np.nan, np.nan, 0) if x is None else (x.mean(), x.std() / np.sqrt(x.notna().sum()), int(x.notna().sum()))
            def gain(a, b):
                if a is None or b is None: return (np.nan, np.nan, np.nan)
                d = (a - b).dropna(); return (d.mean(), d.std() / np.sqrt(len(d)), ttest_rel(a[d.index], b[d.index]).pvalue if len(d) > 2 else np.nan)
            for name, x in (("r_all_pop", allpop), ("r_all_D0", d0), ("r_all_D1", d1), ("r_all_D2", d2), ("r_hi_pop", hipop), ("r_hi_D1", h1), ("r_hi_D2", h2), ("rr_D0", rr0), ("rr_D1", rr1), ("rr_D2", rr2), ("pacc_pop", pp), ("pacc_D1", p1), ("pacc_D2", p2)):
                rec[name], rec[name + "_se"], rec[name + "_n"] = ms(x)
            for name, (x, y) in (("gain_all_D1", (d1, allpop)), ("gain_hi_D1", (h1, hipop)), ("gain_pacc_D1", (p1, pp)), ("gain_all_D2", (d2, allpop)), ("gain_hi_D2", (h2, hipop)), ("gain_pacc_D2", (p2, pp))):
                rec[name], rec[name + "_se"], rec[name + "_p"] = gain(x, y)
            rows.append(rec)
if FROM_FIGDATA: S = pd.read_csv("results/effect_size_summary.csv")
else:
    S = pd.DataFrame(rows); S.to_csv("results/effect_size_summary.csv", index=False)
    show = S[S.side == "seen"][["dataset", "n_est", "r_all_pop", "r_all_D1", "gain_all_D1", "r_hi_pop", "r_hi_D1", "gain_hi_D1", "rr_D1", "rr_D2", "pacc_pop", "pacc_D1", "pacc_D2", "gain_pacc_D1_p"]]
    print(show.round(3).to_string(index=False)); print("unseen:"); print(S[S.side == "unseen"][["dataset", "n_est", "r_all_pop", "r_all_D1", "gain_all_D1", "r_hi_pop", "r_hi_D1", "gain_hi_D1", "rr_D1", "pacc_pop", "pacc_D1", "pacc_D2"]].round(3).to_string(index=False))
# figures: largest n_est per dataset, seen side
F = S[S.apply(lambda r: r.n_est == max(n for l, s, nn in SETS if l == r.dataset for n in nn), axis=1)]
F = F[F.side == np.where(F.dataset.str.startswith("EVA"), "unseen", "seen")].reset_index(drop=True)      # EVA: the 30-rating image mean dominates the seen side, report the unseen side
F.loc[F.dataset.str.startswith("AADB"), ["gain_hi_D1", "gain_hi_D1_se"]] = np.nan                          # AADB: 5 ratings per image, contested-item subset unreliable
# Request 7 (2026-11-17): both panels full text width in the format of Fig 2 (scripts/46): fixed canvas 455 pt wide, plot frame x = 123.6 .. 444.9 pt,
# tick/legend 7 pt, axis label 8 pt, markers 4 pt, caps 2, line 0.8; same height and frame for the two (stacked in Supplementary Fig 3); legend above the axes.
# Series, colours, values, the shift between series (in row units) and the axis ranges (autoscaled from the same data) are as before.
W, H, LEFT, RIGHT, BOTTOM, TOP, FS, FS_LAB = 455.0, 2.4 * 72, 123.6, 444.9, 0.42 * 72, 0.22 * 72, 7, 8
y = np.arange(len(F))[::-1]; ylab = [esc(f"{r.dataset}, {r.n_est} est.") for _, r in F.iterrows()]
def panel(xlabel, vline):
    fig = plt.figure(figsize=(W / 72, H / 72)); ax = fig.add_axes([LEFT / W, BOTTOM / H, (RIGHT - LEFT) / W, (H - BOTTOM - TOP) / H])
    ax.axvline(vline, color="k", lw=0.6, ls=":"); ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=FS); ax.tick_params(axis="x", labelsize=FS); ax.set_xlabel(xlabel, fontsize=FS_LAB); return fig, ax
def finish(fig, ax, name, index_line):
    ax.legend(frameon=False, fontsize=FS, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.5, borderaxespad=0.2)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); wl = max(t.get_window_extent(r).width for t in ax.get_yticklabels()) * 72 / fig.dpi; assert wl < LEFT - 9, wl   # row labels fit left of the frame
    path = f"figures/{name}.pdf"
    with plt.rc_context({"savefig.bbox": "standard"}): fig.savefig(path)
    plt.close(fig); _index(path, f"({W / 72:.2f}x{H / 72:.2f} in, fixed canvas {W:.0f}x{H:.1f} pt, plot frame x = {LEFT} .. {RIGHT} pt as in methods_exp2_hold.pdf): {index_line}")
fig, ax = panel(esc("Gain of D1 over population mean (Pearson r)"), 0)
ax.errorbar(F.gain_all_D1, y + 0.15, xerr=F.gain_all_D1_se, fmt="o", color=TAB10[0], ms=4, capsize=2, lw=0.8, label="all items")
ax.errorbar(F.gain_hi_D1, y - 0.15, xerr=F.gain_hi_D1_se, fmt="s", color=TAB10[1], ms=4, capsize=2, lw=0.8, label=esc("contested items (top tercile of disagreement)"))
finish(fig, ax, "effect_contested", "Effect-size view: per-rater gain of the D1 model over the population item mean on all held-out items (circles) and on contested items only (squares; items in the top tercile of between-rater disagreement among training raters), mean ± s.e.m., seeds averaged; largest estimation set per dataset (London 40 faces, SCUT 500, AADB/EVA/LAPIS 100 ratings, Jester 50 jokes); EVA shown on model-unseen photographs (its seen-side baseline is a 30-rater image mean), AADB contested subset omitted (5 ratings per image); London from the dense-matrix reimplementation (results/london65); the two series of a row are shifted vertically by 0.15 row; same height and plot frame as effect_pair_accuracy.pdf (stacked in Supplementary Fig 3); values in results/effect_size_summary.csv")
fig, ax = panel(esc("Accuracy on contested pairs (population means within 0.1 SD)"), 0.5)
for j, (c, m, lab) in enumerate((("pacc_pop", "o", "population mean"), ("pacc_D1", "s", "D1"), ("pacc_D2", "^", "D2"))):
    ax.errorbar(F[c], y + (1 - j) * 0.2, xerr=F[c + "_se"], fmt=m, color=TAB10[j], ms=4, capsize=2, lw=0.8, label=esc(lab))
finish(fig, ax, "effect_pair_accuracy", "Effect-size view: for pairs of a rater's held-out items whose population means differ by less than 0.1 SD (the population model cannot order them) and which the rater rated differently, the fraction ordered correctly by the population mean, D1 and D2 (mean ± s.e.m. over raters, seeds averaged; up to 4,000 sampled pairs per rater, raters with >= 20 such pairs; a tied prediction counts 0.5); largest estimation set per dataset, EVA on model-unseen photographs; the three series of a row are shifted vertically by 0.2 row; same height and plot frame as effect_contested.pdf; values in results/effect_size_summary.csv")
print("done")
