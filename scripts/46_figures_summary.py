"""Cross-dataset summary figures (one panel per PDF):
  methods_exp2_hold.pdf        : Xu Exp. 2 held-out rho per method (baselines, amortised, direct)
  generalization_gain.pdf      : gain of the reweighting model over the population model (D1 - D0), per dataset and evaluation
  generalization_structure.pdf : D1-D0, D2-D1, D3-D2 on held-out data per dataset (structure of individual differences)
Per-participant deltas are computed from the raw eval files (seeds averaged per participant), error bars = s.e.m.
2026-11-12 (requests 1 and 2 from the writing side): the three result panels of the paper are drawn here at full text width (455 pt):
methods_exp2_hold.pdf (Fig 2), generalization_gain.pdf (Fig 3a) and generalization_structure.pdf (Fig 3b, stacked below 3a); since 2026-11-17 also
methods_exp1_hold.pdf (Supplementary Fig 6a, same format as Fig 2; do not overwrite it with scripts/13). Height = margins +
rows x 0.15 in; left margin and plot-frame edges are common; rows are grouped by domain (faces, artworks, photographs, jokes) in the same order;
the structure panel uses the largest estimation set of each dataset (EVA on model-unseen photographs). Colours and markers as in earlier versions.
2026-11-17 (request 6): in generalization_structure.pdf the three series of a row are shifted vertically (no overlap); its row pitch is 0.18 in (PITCH_B), so it is taller.
The INDEX.md lines of the panels are replaced in place.
Source data: generalization_{gain,structure} = results/generalization_summary_deltas.csv; methods_exp{1,2}_hold = results/figdata/<panel>.csv.
FIGDATA=read redraws the four panels from these tables without the per-participant evaluation files.
Run: uv run python -I scripts/46_figures_summary.py      (FIGDATA=read uv run python -I scripts/46_figures_summary.py)
"""
import os, glob, numpy as np, pandas as pd
from face_id.plotting import plt, savefig, esc, TAB10, _index, figdata, FROM_FIGDATA
ONLY = os.environ.get("ONLY")                      # ONLY=<substring>: write only matching panels (2026-11-17); the common left margin is still computed from all three
from face_id.paths import RESULTS, FIGURES
def per_part(files, idcol, metric, filt=None):
    df = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    if filt: df = filt(df)
    return df.groupby(["kind", idcol])[metric].mean().unstack(0)
def xu(exp, metric):
    fs = glob.glob(os.path.join(RESULTS, f"runs/exp{exp}_D?_D8_M4_f*_s*_cv/eval.csv"))
    return per_part(fs, "pid", metric, lambda d: d[d.infer == "amortized"])
def gen(sub, idcol, metric, n_est, pat="eval_*.csv"):
    # NB: results/london and results/avfs also hold the D = 5/16 and M = 2/8 sensitivity models (10 folds x 1 seed each), which the default
    # pattern would average in. All London / AVFS 56 rows therefore pass MAIN (D = 8, M = 4, all 5 seeds), the same selection as the tables
    # results/{london,avfs}_{summary,paired}.csv (scripts/83). Until 2026-11-12 the two gain rows used the default pattern (D1 - D0 differed by <= 0.0006).
    return per_part(glob.glob(os.path.join(RESULTS, f"{sub}/{pat}")), idcol, metric, lambda d: d[d.n_est == n_est])
MAIN = "eval_*_D8_M4_f*.csv"
rows = []
ALL = 10 ** 6
# (dataset, evaluation, metric symbol, unseen?, loader, panel: "A" gain / "B" structure / "AB", row label in the gain panel, row label in the structure panel)
# The first 19 rows are the rows of the earlier version (same dataset/evaluation keys in the CSV); their order here fixes the dataset colours.
specs = [
    ("Xu Exp. 2 (diverse faces, arrangement)", "held-out trials", r"$\rho$", False, lambda: xu(2, "rho_hold"), "AB", "Xu Exp. 2: held-out", "Xu Exp. 2 (all trials)"),
    ("Xu Exp. 2 (diverse faces, arrangement)", "retest, unseen faces", r"$\rho$", True, lambda: xu(2, "rho_tr_unseen"), "A", "Xu Exp. 2: retest", None),
    ("Xu Exp. 1 (White faces, arrangement)", "held-out trials", r"$\rho$", False, lambda: xu(1, "rho_hold"), "AB", "Xu Exp. 1: held-out", "Xu Exp. 1 (all trials)"),
    ("Xu Exp. 1 (White faces, arrangement)", "retest, unseen faces", r"$\rho$", True, lambda: xu(1, "rho_tr_unseen"), "A", "Xu Exp. 1: retest", None),
    ("London Set (attractiveness ratings)", "held-out faces, 20 est.", "$r$", False, lambda: gen("london", "rater", "r_amort", 20, MAIN), "A", "London Set: held-out, 20", None),
    ("London Set (attractiveness ratings)", "unseen faces (image), 20 est.", "$r$", True, lambda: gen("london_img", "rater", "r_amort_unseen", 20), "A", "London Set: unseen, 20", None),
    ("AVFS 56 faces (similarity triplets)", "held-out triplets, 20 est.", "acc.", False, lambda: gen("avfs", "ann", "acc_amort", 20, MAIN), "A", "AVFS 56: held-out, 20", None),
    ("AVFS 4,921 faces (similarity, image)", "held-out triplets, 20 est.", "acc.", False, lambda: gen("avfs_img", "ann", "acc_main", 20), "A", "AVFS 4,921: held-out, 20", None),
    ("AVFS 4,921 faces (similarity, image)", "unseen 56 faces, all est.", "acc.", True, lambda: gen("avfs_img", "ann", "acc_novel", ALL), "A", "AVFS 4,921: unseen 56, all", None),
    ("LAPIS (art aesthetics ratings, image)", "held-out artworks, 20 est.", "$r$", False, lambda: gen("lapis_img", "rater", "r_amort_seen", 20), "A", "LAPIS: held-out, 20", None),
    ("LAPIS (art aesthetics ratings, image)", "unseen artworks, 20 est.", "$r$", True, lambda: gen("lapis_img", "rater", "r_amort_unseen", 20), "A", "LAPIS: unseen, 20", None),
    ("AADB (photo aesthetics ratings, image)", "held-out photographs, 20 est.", "$r$", False, lambda: gen("aadb_img", "rater", "r_amort_seen", 20), "A", "AADB: held-out, 20", None),
    ("AADB (photo aesthetics ratings, image)", "unseen photographs, 20 est.", "$r$", True, lambda: gen("aadb_img", "rater", "r_amort_unseen", 20), "A", "AADB: unseen, 20", None),
    ("EVA (photo aesthetics ratings, image)", "held-out photographs, 20 est.", "$r$", False, lambda: gen("eva_img", "rater", "r_amort_seen", 20), "A", "EVA: held-out, 20", None),
    ("EVA (photo aesthetics ratings, image)", "unseen photographs, 20 est.", "$r$", True, lambda: gen("eva_img", "rater", "r_amort_unseen", 20), "A", "EVA: unseen, 20", None),
    ("Jester (joke ratings, text)", "held-out jokes, 20 est.", "$r$", False, lambda: gen("jester", "rater", "r_amort_seen", 20), "A", "Jester: held-out, 20", None),
    ("SCUT-FBP5500 (face attractiveness, image)", "held-out faces, 20 est.", "$r$", False, lambda: gen("scut", "rater", "r_amort_seen", 20), "A", "SCUT-FBP5500: held-out, 20", None),
    ("FaceSim3D 2D (CFD faces, triplets)", "held-out triplets, 20 est.", "acc.", False, lambda: gen("facesim2D", "ann", "acc_amort", 20), "A", "FaceSim3D 2D: held-out, 20", None),
    ("FaceSim3D 3D (CFD faces, triplets)", "held-out triplets, 20 est.", "acc.", False, lambda: gen("facesim3D", "ann", "acc_amort", 20), "A", "FaceSim3D 3D: held-out, 20", None),
    # structure panel: largest estimation set of each dataset
    ("London Set (attractiveness ratings)", "held-out faces, 40 est.", "$r$", False, lambda: gen("london", "rater", "r_amort", 40, MAIN), "B", None, "London Set (40 faces)"),
    ("AVFS 56 faces (similarity triplets)", "held-out triplets, 60 est.", "acc.", False, lambda: gen("avfs", "ann", "acc_amort", 60, MAIN), "B", None, "AVFS 56 (60 triplets)"),
    ("AVFS 4,921 faces (similarity, image)", "held-out triplets, all est.", "acc.", False, lambda: gen("avfs_img", "ann", "acc_main", ALL), "B", None, "AVFS 4,921 (all triplets)"),
    ("SCUT-FBP5500 (face attractiveness, image)", "held-out faces, 500 est.", "$r$", False, lambda: gen("scut", "rater", "r_amort_seen", 500), "B", None, "SCUT-FBP5500 (500 faces)"),
    ("FaceSim3D 2D (CFD faces, triplets)", "held-out triplets, 60 est.", "acc.", False, lambda: gen("facesim2D", "ann", "acc_amort", 60), "B", None, "FaceSim3D 2D (60 triplets)"),
    ("FaceSim3D 3D (CFD faces, triplets)", "held-out triplets, 60 est.", "acc.", False, lambda: gen("facesim3D", "ann", "acc_amort", 60), "B", None, "FaceSim3D 3D (60 triplets)"),
    ("LAPIS (art aesthetics ratings, image)", "held-out artworks, 100 est.", "$r$", False, lambda: gen("lapis_img", "rater", "r_amort_seen", 100), "B", None, "LAPIS (100 ratings)"),
    ("AADB (photo aesthetics ratings, image)", "held-out photographs, 100 est.", "$r$", False, lambda: gen("aadb_img", "rater", "r_amort_seen", 100), "B", None, "AADB (100 ratings)"),
    ("EVA (photo aesthetics ratings, image)", "unseen photographs, 100 est.", "$r$", True, lambda: gen("eva_img", "rater", "r_amort_unseen", 100), "B", None, "EVA (100 ratings, unseen)"),
    ("Jester (joke ratings, text)", "held-out jokes, 50 est.", "$r$", False, lambda: gen("jester", "rater", "r_amort_seen", 50), "B", None, "Jester (50 jokes)"),
]
# row order shared by both panels: faces, artworks, photographs, jokes
ORDER = ["Xu Exp. 2", "Xu Exp. 1", "London Set", "AVFS 56 faces", "AVFS 4,921 faces", "SCUT-FBP5500", "FaceSim3D 2D", "FaceSim3D 3D", "LAPIS", "AADB", "EVA", "Jester"]
DOMAIN = {"LAPIS": 1, "AADB": 2, "EVA": 2, "Jester": 3}                                                             # default 0 = faces
short = lambda ds: ds.split(" (")[0]
for ds, ev, sym, unseen, fn, panel, labA, labB in ([] if FROM_FIGDATA else specs):      # FIGDATA=read: the plotted values are read from results/generalization_summary_deltas.csv
    try: W = fn()
    except ValueError: continue          # dataset not available yet
    for a_, b_ in (("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D2", "D0")):
        if a_ in W and b_ in W:
            d = (W[a_] - W[b_]).dropna(); rows.append(dict(dataset=ds, evaluation=ev, metric=sym, unseen=unseen, contrast=f"{a_}-{b_}", delta=d.mean(), sem=d.std() / np.sqrt(len(d)), n=len(d)))
if FROM_FIGDATA: S = pd.read_csv(os.path.join(RESULTS, "generalization_summary_deltas.csv"))
else: S = pd.DataFrame(rows); S.to_csv(os.path.join(RESULTS, "generalization_summary_deltas.csv"), index=False); print(S.round(4).to_string(index=False))
datasets = list(dict.fromkeys(s[0] for s in specs)); col = {d: TAB10[i % 10] for i, d in enumerate(datasets)}   # tab10 repeats after 10 datasets; rows are identified by their labels
# ---- common geometry of the three full-width result panels (request 2 of 2026-11-12: Fig 2 = methods_exp2_hold, Fig 3a = gain, Fig 3b = structure)
# Width 455 pt for all; height = fixed top/bottom margins + (rows + 0.6 per group gap) x PITCH; the left margin (row labels) and the plot frame's
# left/right edges are identical in the three PDFs. Fixed canvas (no bbox tight), explicit axes rectangle (no automatic layout).
W_IN, PITCH, FS, FS_LAB, GAP, YPAD = 455 / 72, 0.15, 7, 8, 0.6, 0.7
TOP, BOTTOM, RIGHT, LABPAD = 0.05, 0.42, 0.14, 0.13            # inches; LABPAD = tick length + tick pad + outer margin left of the row labels
def layout(panel):
    """Rows of a panel in the shared order, with y positions (top to bottom, a gap between domains)."""
    sp = sorted([s for s in specs if panel in s[5] and (s[0], s[1]) in set(zip(S.dataset, S.evaluation))], key=lambda s: ORDER.index(short(s[0])))   # stable: keeps held-out before unseen
    y, cur, prev = [], 0.0, None
    for s in sp:
        dom = DOMAIN.get(short(s[0]), 0)
        if prev is not None: cur -= 1 + (GAP if dom != prev else 0)
        y.append(cur); prev = dom
    return sp, np.array(y)
PANELS = []                                                      # (fig, ax, y, name, index_line)
def panel(y, labels, xlabel, name, index_line, zero=True, pitch=None):
    fig = plt.figure(); ax = fig.add_axes([0.3, 0.2, 0.6, 0.7])
    if zero: ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=FS); ax.tick_params(axis="x", labelsize=FS); ax.set_ylim(y.min() - YPAD, y.max() + YPAD)
    ax.set_xlabel(xlabel, fontsize=FS_LAB); PANELS.append((fig, ax, y, name, index_line, pitch or PITCH)); return ax
def save_all():
    wl = []
    for fig, ax, y, name, _, _ in PANELS:                           # pass 1: widest row label over the three panels (inches)
        fig.set_size_inches(W_IN, 3); fig.canvas.draw(); r = fig.canvas.get_renderer(); wl.append(max(t.get_window_extent(r).width for t in ax.get_yticklabels()) / fig.dpi)
    left = max(wl) + LABPAD
    for fig, ax, y, name, index_line, pitch in PANELS:                  # pass 2: same left/right frame edges, height from the number of rows
        ph = (y.max() - y.min() + 2 * YPAD) * pitch; h = TOP + ph + BOTTOM
        fig.set_size_inches(W_IN, h); ax.set_position([left / W_IN, BOTTOM / h, (W_IN - left - RIGHT) / W_IN, ph / h])
        if ONLY and ONLY not in name: plt.close(fig); continue
        with plt.rc_context({"savefig.bbox": "standard"}): fig.savefig(os.path.join(FIGURES, name + ".pdf"))
        plt.close(fig); print(f"{name}: {W_IN * 72:.0f} x {h * 72:.1f} pt ({W_IN:.2f} x {h:.2f} in), plot frame x = {left * 72:.1f} .. {(W_IN - RIGHT) * 72:.1f} pt, label width {wl[[q[3] for q in PANELS].index(name)] * 72:.1f} pt")
        _index(name + ".pdf", f"({W_IN:.2f}x{h:.2f} in): {index_line}")
get = lambda s, c: S[(S.dataset == s[0]) & (S.evaluation == s[1]) & (S.contrast == c)]
# Fig 2: methods_exp2_hold (same values, colours, markers and row labels as scripts/13_figures.py fig_methods(2, "rho_hold"); only this panel is drawn here
# because 13 rewrites figures/INDEX.md and all its panels). Groups: model-free baselines / amortised inference / direct optimisation.
import json
LABEL = {"agg": "Population RDM", "mds5": "5-D MDS", "estrdm": "Own measured RDM", "blend": "Shrinkage blend", "rw5": "5-D reweighting",
         "D0": "D0 (shared only)", "D1": "D1 (reweighting)", "D2": "D2 (linear transform)", "D3": "D3 (+ face offsets)"}
def pooled(exp, tag="_cv"):
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{exp}_D*_D8_M*_f*_s*{tag}"))):
        if not os.path.exists(os.path.join(d, "eval.csv")): continue
        a = json.load(open(os.path.join(d, "args.json"))); e = pd.read_csv(os.path.join(d, "eval.csv")); e = e[e.frac_est == 0.7].copy()
        e["method"] = a["kind"] + "-M" + str(a["M"]) + "-" + e.infer; rows.append(e[["pid", "method", "rho_hold", "rho_tr_unseen"]])
    ev = pd.concat(rows).groupby(["pid", "method"]).mean().reset_index()
    bl = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS, "baselines", f"exp{exp}_f*.csv"))])
    for b in ["agg", "mds5", "estrdm", "blend", "rw5"]:
        ev = pd.concat([ev, bl[["pid", f"rho_hold_{b}", f"rho_tr_unseen_{b}"]].rename(columns={f"rho_hold_{b}": "rho_hold", f"rho_tr_unseen_{b}": "rho_tr_unseen"}).assign(method=b)])
    return ev
grp = lambda n: 0 if not n.startswith("D") else (1 if "amortized" in n else 2)
def methods_panel(exp, name, index_line):
    def table():
        sm = pd.read_csv(os.path.join(RESULTS, f"summary_exp{exp}_cv.csv")); se = pooled(exp).groupby("method").rho_hold.sem()
        return pd.DataFrame(dict(method=sm.method, value=sm.rho_hold, sem=[se.get(n, 0) for n in sm.method]))
    T = figdata(name, table); val = dict(zip(T.method, T.value)); sem = dict(zip(T.method, T["sem"])); names = list(T.method)      # plotted values: results/figdata/<name>.csv
    order = sorted(names, key=grp)                                  # stable: same order as before within each group
    y, cur = [], 0.0
    for i, n in enumerate(order):
        if i: cur -= 1 + (GAP if grp(n) != grp(order[i - 1]) else 0)
        y.append(cur)
    y = np.array(y)
    ax = panel(y, [esc(LABEL.get(n.split("-")[0], n.split("-")[0]) + ("" if not n.startswith("D") else f" M={n.split('-')[1][1:]}, {n.split('-')[2]}")) for n in order],
               esc("Spearman rho, held-out arrangements"), name, index_line, zero=False)
    for n, yi in zip(order, y):
        ax.errorbar(val[n], yi, xerr=sem[n], fmt="o", color=TAB10[grp(n)], ms=4, capsize=2, lw=0.8)
methods_panel(2, "methods_exp2_hold", "Exp2 rho_hold: mean ± s.e.m. over fold-out participants (seeds averaged) per method, in three groups (model-free baselines: population RDM, 5-D MDS, own measured RDM, shrinkage blend, 5-D reweighting; Face-ID with amortised inference; Face-ID with direct optimisation of z); values in results/summary_exp2_cv.csv; full-width panel with the same width, row pitch and plot-frame edges as generalization_{gain,structure}.pdf; drawn by scripts/46 (same values, colours and markers as the scripts/13 version)")
# Supplementary Fig 6a (request 7, 2026-11-17): the Exp. 1 panel in the same format as Fig 2 (was drawn by scripts/13 at 3.6 x 4.32 in)
methods_panel(1, "methods_exp1_hold", "Exp1 rho_hold: mean ± s.e.m. over fold-out participants (seeds averaged) per method, in three groups (model-free baselines: population RDM, 5-D MDS, own measured RDM, shrinkage blend, 5-D reweighting; Face-ID with amortised inference; Face-ID with direct optimisation of z); values in results/summary_exp1_cv.csv; same format as methods_exp2_hold.pdf (width, row pitch, plot-frame edges, colours, markers, fonts); drawn by scripts/46 since 2026-11-17 (same methods and values as the earlier scripts/13 version)")
# Fig 3a: D1 - D0 per dataset x evaluation
sp, y = layout("A")
ax = panel(y, [f"{esc(s[6])} ({s[2]})" for s in sp], esc("Gain over population model (D1 $-$ D0)"), "generalization_gain",
           "Cross-dataset summary: per-participant gain of the D1 reweighting model over the D0 population model (mean ± s.e.m.; seeds averaged per participant) for each dataset and evaluation, rows grouped by domain (faces: Xu Exp. 2 to FaceSim3D; artworks: LAPIS; photographs: AADB, EVA; jokes: Jester); filled = held-out judgments on the participant's own stimuli, open = stimuli unseen by the participant (and by the model for image-based rows; Xu retest = retest task, Exp. 1 mostly unseen faces); the number after the comma = number of estimation items (Xu: all estimation trials; all = all estimation triplets); metric in parentheses (Spearman rho, Pearson r, odd-one-out accuracy); London and AVFS 56 rows use the D = 8, M = 4 models with all 5 seeds, as results/{london,avfs}_paired.csv; same width, row pitch, plot-frame left/right edges and row order as generalization_structure.pdf (stacked above it in Fig 3); values in results/generalization_summary_deltas.csv")
for s, yi in zip(sp, y):
    r = get(s, "D1-D0").iloc[0]
    ax.errorbar(r.delta, yi, xerr=r["sem"], fmt="o", color=col[s[0]], mfc="white" if s[3] else col[s[0]], ms=5, capsize=2, lw=1)
# Fig 3b: structure at the largest estimation set: D1-D0, D2-D1, D3-D2
mk = {"D1-D0": ("o", TAB10[0], "D1 $-$ D0 (reweighting)"), "D2-D1": ("s", TAB10[1], "D2 $-$ D1 (linear transform)"), "D3-D2": ("^", TAB10[2], "D3 $-$ D2 (stimulus-specific offsets)")}
PITCH_B, OFF = 0.18, 0.27     # request 6 (2026-11-17): row pitch 0.18 in (13 pt) for this panel and series shifted by 0.27 row (3.5 pt) so that markers (3 pt) and caps (3 pt) of the three series do not overlap
sp, y = layout("B"); off = {"D1-D0": OFF, "D2-D1": 0.0, "D3-D2": -OFF}
ax = panel(y, [esc(s[7]) for s in sp], esc("Difference in performance (metric units)"), "generalization_structure",
           "Cross-dataset structure of individual differences at the largest estimation set of each dataset (in parentheses in the row labels): per-participant differences D1-D0 (reweighting of shared dimensions), D2-D1 (adding a linear transform), D3-D2 (adding stimulus-specific offsets; fitted only for the Xu, London and AVFS rows), mean ± s.e.m., seeds averaged per participant; held-out judgments on the participant's own stimuli, except EVA = photographs unseen by the model and the rater (its seen-side baseline is a 30-rater image mean); metric units: Spearman rho (Xu), Pearson r (ratings), odd-one-out accuracy (AVFS, FaceSim3D); AADB at 100 ratings has 55 raters; London and AVFS 56 rows use the D = 8, M = 4 models with all 5 seeds, as results/{london,avfs}_paired.csv; same width, plot-frame left/right edges and row order as generalization_gain.pdf (stacked below it in Fig 3); row pitch 0.18 in (gain: 0.15 in) and the three series of a row shifted vertically by 3.5 pt each for visibility (D1-D0 above, D3-D2 below the row position; same values); values in results/generalization_summary_deltas.csv (rows with the estimation size named in the evaluation column)", pitch=PITCH_B)
for c, (m, cc, lab) in mk.items():
    first = True
    for s, yi in zip(sp, y):
        r = get(s, c)
        if not len(r): continue
        r = r.iloc[0]; ax.errorbar(r.delta, yi + off[c], xerr=r["sem"], fmt=m, color=cc, ms=3, capsize=1.5, lw=1, label=esc(lab) if first else None); first = False
yl = ax.get_ylim(); yleg = y[[short(s[0]) for s in sp].index("FaceSim3D 2D")]            # empty area right of the SCUT / FaceSim3D rows
ax.legend(frameon=False, fontsize=FS, loc="center right", bbox_to_anchor=(1.0, (yleg - yl[0]) / (yl[1] - yl[0])), handlelength=1.2, borderaxespad=0, labelspacing=0.3)
save_all()
print("done")
