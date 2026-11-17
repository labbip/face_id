"""Figures for the generalization tests (one panel per PDF): learning curves for London Set (learned embedding,
image embedding / unseen faces) and AVFS; z vs male-face preference; residual PC1 vs face gender.
Every panel is drawn from the table of its plotted values, written to results/figdata/<panel>.csv; FIGDATA=read redraws from those tables without the evaluation files.
Run: uv run python -I scripts/34_figures_generalization.py      (FIGDATA=read uv run python -I scripts/34_figures_generalization.py)
"""
import os, glob, numpy as np, pandas as pd
from numpy.linalg import svd
from face_id.plotting import plt, esc, TAB10
from face_id import plotting
ONLY = os.environ.get("ONLY")                      # ONLY=<substring>: write only matching panels (others are computed but not saved)
def savefig(fig, name, *a, **k):                     # learning curves: fixed canvas (size in pt as 3rd argument); other panels: plotting.savefig
    if ONLY and ONLY not in name: return plt.close(fig)
    return plotting.savefig_fixed(fig, name, *a, **k) if a else plotting.savefig(fig, name, **k)
from face_id.paths import RESULTS
KC = {"D1": TAB10[0], "D2": TAB10[1], "D3": TAB10[2]}

STEP = 4.0   # pt between neighbouring series at the same x (request 6, 2026-11-17): cap width 3 pt, marker 3 pt
def curve(load, metric_amort, metric_direct, pop_col, ylabel, xticks, fname, index_line, legend_loc="lower right", kinds=("D1", "D2", "D3")):
    """Learning curve. Series at the same x are shifted by STEP pt each (order: D1 amortized, D1 direct, D2 amortized, ...), legend above the axes,
    fixed canvas = the page size of the existing PDF (so the physical size does not change). legend_loc is kept for old calls and ignored.
    load() -> per-rater evaluation rows (called only when the plotted values are computed; with FIGDATA=read they come from results/figdata/<fname>.csv).
    A None in xticks and "{nall}" in index_line stand for the largest n_est of the table (AVFS main set: mean number of all estimation triplets)."""
    def table():
        df = load(); g = df.groupby(["kind", "n_est", "rater"])[[c for c in (metric_amort, metric_direct, pop_col) if c in df]].mean().reset_index(); out = []
        for kind in kinds:
            for m, lab in ((metric_amort, "amortized"), (metric_direct, "direct")):
                if m in g: a = g[g.kind == kind].groupby("n_est")[m].agg(["mean", "sem", "count"]).reset_index(); out.append(a.assign(kind=kind, inference=lab))
        return pd.concat(out + [pd.DataFrame([dict(kind="D0", inference="population (dotted line)", mean=g[g.kind == "D0"][pop_col].mean())])], ignore_index=True)[["kind", "inference", "n_est", "mean", "sem", "count"]]
    if plotting.FROM_FIGDATA and not os.path.exists(os.path.join(RESULTS, "figdata", fname + ".csv")): return
    T = plotting.figdata(fname, table); pop = T[T.kind == "D0"]["mean"].iloc[0]; T = T[T.kind != "D0"]; nall = int(T.n_est.max()); xticks = [nall if x is None else x for x in xticks]
    fig, ax = plt.subplots(layout="constrained"); fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72)
    size = plotting.pdf_size(fname) or (3.4 * 72, 2.6 * 72); fig.set_size_inches(size[0] / 72, size[1] / 72)
    ser = [(kind, lab, f) for kind in kinds for lab, f in (("amortized", "o-"), ("direct", "s--")) if ((T.kind == kind) & (T.inference == lab)).any()]
    lo, hi, xs = [], [], []
    for k, (kind, lab, f) in enumerate(ser):
        a = T[(T.kind == kind) & (T.inference == lab)].set_index("n_est"); lo.append((a["mean"] - a["sem"]).min()); hi.append((a["mean"] + a["sem"]).max()); xs += list(a.index)
        ax.errorbar(a.index, a["mean"], a["sem"], fmt=f, color=KC[kind], ms=3, capsize=1.5, elinewidth=1, capthick=1, label=esc(f"{kind} {lab}"), transform=plotting.dodge(ax, k, len(ser), STEP))
    ax.axhline(pop, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(xticks); ax.set_xticklabels([str(x) for x in xticks]); ax.minorticks_off()
    # limits set by hand (artists with a shifted transform are not autoscaled): y = matplotlib's default 5% margin around the data incl. error bars and the dotted line,
    # i.e. the same range as before; x = 9% margin in log units (was 5%) to leave room for the shifted end points
    lo, hi = min(lo + [pop]), max(hi + [pop]); ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    l0, l1 = np.log10(min(xs)), np.log10(max(xs)); ax.set_xlim(10 ** (l0 - 0.09 * (l1 - l0)), 10 ** (l1 + 0.09 * (l1 - l0)))
    ax.set_xlabel("Number of estimation faces" if "london" in fname else "Number of estimation ratings" if any(k in fname for k in ("lapis", "aadb", "eva", "jester", "scut")) else "Number of estimation triplets"); ax.set_ylabel(ylabel)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, loc="outside upper center", handlelength=1.5, ncol=len(kinds), columnspacing=0.8, labelspacing=0.2, borderaxespad=0)
    savefig(fig, fname, size, index_line=index_line.replace("{nall}", str(nall)) + f"; series at the same x are shifted horizontally by {STEP:g} pt each for visibility (same values), legend above the axes")
def ev(sub, ren, pat="eval_*.csv"):
    """Loader of results/<sub>/<pat> with columns renamed (ren)."""
    return lambda: pd.concat([pd.read_csv(x) for x in sorted(glob.glob(os.path.join(RESULTS, sub, pat)))]).rename(columns=ren)
def has(sub): return plotting.FROM_FIGDATA or bool(glob.glob(os.path.join(RESULTS, sub, "eval_*.csv")))

# 1. London Set, learned face embedding
if has("london"):
    curve(ev("london", {"r_amort": "amort", "r_direct": "direct", "r_pop": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out faces", [5, 10, 20, 40], "learning_curve_london",
          "London Set (attractiveness, learned face embedding): Pearson r on the rater's remaining faces vs number of estimation faces, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = population mean (training raters); values in results/london_summary.csv")
# 2. London Set, image embedding, unseen faces
if has("london_img"):
    curve(ev("london_img", {"r_amort_unseen": "amort", "r_direct_unseen": "direct", "r_popimg_unseen": "pop"}), "amort", "direct", "pop", "Pearson $r$, unseen faces", [5, 10, 20, 40], "learning_curve_london_unseen",
          "London Set (CLIP image embedding, faces held out from training): Pearson r on faces unseen by the model and the rater vs number of estimation faces, mean ± s.e.m. over fold-out raters; dotted = image-based population ridge baseline; values in results/london_img_summary.csv", legend_loc="center right")
# 3. AVFS
if has("avfs"):
    curve(lambda: ev("avfs", {"acc_amort": "amort", "acc_direct": "direct", "ann": "rater"})().assign(pop=lambda d: d["amort"]), "amort", "direct", "pop", "Odd-one-out accuracy, held-out triplets", [10, 20, 40, 60], "learning_curve_avfs",
          "AVFS novel-stimuli set (56 faces, 504 annotators): odd-one-out accuracy on the annotator's held-out triplets vs number of estimation triplets, mean ± s.e.m. over fold-out annotators (seeds averaged); dotted = D0 population model; values in results/avfs_summary.csv")
# 3b. AVFS main set (CLIP image embedding, 4,921 faces): accuracy on the annotator's held-out triplets
def avfs_main():
    df = ev("avfs_img", {"acc_main": "amort", "acc_main_direct": "direct", "ann": "rater"})(); df["pop"] = df["amort"]
    nall = int(round(df[df.n_est == 10 ** 6].n_est_used.mean())); df["n_est"] = df.n_est.where(df.n_est != 10 ** 6, nall); return df
if has("avfs_img"):
    curve(avfs_main, "amort", "direct", "pop", "Odd-one-out accuracy, held-out triplets", [20, 50, 100, None], "learning_curve_avfs_main",
          "AVFS main set (4,921 FFHQ faces, CLIP image embedding, 1,283 annotators): odd-one-out accuracy on the annotator's held-out triplets vs number of estimation triplets (rightmost = all estimation triplets, mean {nall}), mean ± s.e.m. over fold-out annotators (seeds averaged); dotted = D0 population model; D3 (face-specific offsets over 4,921 faces) overfits and lies below D0, see results/avfs_img_summary.csv; values in results/avfs_img_summary.csv", kinds=("D1", "D2"))
# 3c. LAPIS (art aesthetics, CLIP image embedding): seen and unseen artworks
if has("lapis_img"):
    curve(ev("lapis_img", {"r_amort_unseen": "amort", "r_direct_unseen": "direct", "r_pop_unseen": "pop"}), "amort", "direct", "pop", "Pearson $r$, unseen artworks", [10, 20, 50, 100], "learning_curve_lapis_unseen",
          "LAPIS art aesthetics (11,723 artworks, 553 raters, CLIP image embedding, artworks held out from training): Pearson r on artworks unseen by the model and the rater vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = image-based population ridge baseline; values in results/lapis_img_summary.csv", kinds=("D1", "D2"))
    curve(ev("lapis_img", {"r_amort_seen": "amort", "r_direct_seen": "direct", "r_pop_seen": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out artworks", [10, 20, 50, 100], "learning_curve_lapis",
          "LAPIS art aesthetics: Pearson r on the rater's remaining ratings of training artworks vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = artwork mean over training raters; values in results/lapis_img_summary.csv", kinds=("D1", "D2"))
# 3d. AADB (photo aesthetics, CLIP image embedding): seen and unseen photographs
if has("aadb_img"):
    curve(ev("aadb_img", {"r_amort_unseen": "amort", "r_direct_unseen": "direct", "r_pop_unseen": "pop"}), "amort", "direct", "pop", "Pearson $r$, unseen photographs", [10, 20, 50, 100], "learning_curve_aadb_unseen",
          "AADB photo aesthetics (9,958 Flickr photographs, 68 raters with >= 100 ratings, CLIP image embedding, photographs held out from training): Pearson r on photographs unseen by the model and the rater vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = image-based population ridge baseline; values in results/aadb_img_summary.csv", kinds=("D1", "D2"))
    curve(ev("aadb_img", {"r_amort_seen": "amort", "r_direct_seen": "direct", "r_pop_seen": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out photographs", [10, 20, 50, 100], "learning_curve_aadb",
          "AADB photo aesthetics: Pearson r on the rater's remaining ratings of training photographs vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = photograph mean over training raters; values in results/aadb_img_summary.csv", kinds=("D1", "D2"))
# 3e. EVA (photo aesthetics, CLIP image embedding, 175 raters)
if has("eva_img"):
    curve(ev("eva_img", {"r_amort_unseen": "amort", "r_direct_unseen": "direct", "r_pop_unseen": "pop"}), "amort", "direct", "pop", "Pearson $r$, unseen photographs", [10, 20, 50, 100], "learning_curve_eva_unseen",
          "EVA photo aesthetics (4,070 AVA photographs, 175 raters with >= 100 ratings, CLIP image embedding, photographs held out from training): Pearson r on photographs unseen by the model and the rater vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = image-based population ridge baseline; D2 (linear transform) exceeds D1 from 50 ratings; values in results/eva_img_summary.csv", kinds=("D1", "D2"))
    curve(ev("eva_img", {"r_amort_seen": "amort", "r_direct_seen": "direct", "r_pop_seen": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out photographs", [10, 20, 50, 100], "learning_curve_eva",
          "EVA photo aesthetics: Pearson r on the rater's remaining ratings of training photographs vs number of estimation ratings, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = photograph mean over training raters (about 30 ratings per photograph, above the 8-dimensional image model); values in results/eva_img_summary.csv", kinds=("D1", "D2"))
# 3f. Jester (jokes, learned item embedding + text embedding, 11,067 raters)
if has("jester"):
    curve(ev("jester", {"r_amort_seen": "amort", "r_direct_seen": "direct", "r_pop_seen": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out jokes", [10, 20, 50], "learning_curve_jester",
          "Jester joke ratings (150 jokes, 11,067 raters with >= 50 ratings, learned joke embedding plus sentence embedding): Pearson r on the rater's remaining ratings of training jokes vs number of estimation jokes, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = joke mean over training raters; values in results/jester_summary.csv", kinds=("D1", "D2"))
# 3g. SCUT-FBP5500 (dense face attractiveness, CLIP image embedding, 60 raters)
if has("scut"):
    curve(ev("scut", {"r_amort_seen": "amort", "r_direct_seen": "direct", "r_pop_seen": "pop"}), "amort", "direct", "pop", "Pearson $r$, held-out faces", [10, 20, 50, 100, 500], "learning_curve_scut",
          "SCUT-FBP5500 face attractiveness (5,500 faces, 60 raters rating every face, CLIP image embedding plus learned face embedding): Pearson r on the rater's remaining ratings of training faces vs number of estimation faces, mean ± s.e.m. over fold-out raters (seeds averaged); dotted = face mean over training raters; individualisation adds nothing (shared taste dominates), D2 exceeds D1 by 0.006 only at 500 faces; values in results/scut_summary.csv", kinds=("D1", "D2"))
# 4. (moved to scripts/88) z vs male-face preference; the block below now only prepares the residuals for 5.
# london_z_prefmale.pdf is drawn by scripts/88_figure_london_z_prefmale.py since 2026-11-13 (main D = 8, M = 4 models only, per fold model; the code that was here
# averaged z over all eval_D1_*.csv including the D/M sensitivity models and gave CV R2 0.30). Do not draw it here.
# 5. residual PC1 loading vs face gender (one point per face; the plotted values incl. the horizontal jitter are in results/figdata/london_resid_pc1.csv)
def resid_pc1():
    r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(float)
    info = pd.read_csv("data/raw/london/london_faces_info.csv", dtype={"face_id": str}); fg = (info.face_gender.to_numpy() == "male")
    res = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
    U, S, Vt = svd(res, full_matrices=False); pc1 = Vt[0] * np.sign(np.corrcoef(Vt[0], fg)[0, 1]); rng = np.random.default_rng(0); out = []
    for i, m in enumerate((~fg, fg)): out.append(pd.DataFrame(dict(group=i, x=i + rng.uniform(-0.15, 0.15, m.sum()), pc1_loading=pc1[m])))
    return pd.concat(out, ignore_index=True).assign(abs_r_face_gender=abs(np.corrcoef(Vt[0], fg)[0, 1]), pc1_variance_share=S[0] ** 2 / (S ** 2).sum())
if plotting.FROM_FIGDATA or os.path.exists("data/raw/london/london_faces_ratings.csv"):
    P = plotting.figdata("london_resid_pc1", resid_pc1); fig, ax = plt.subplots()
    for i in (0, 1): ax.scatter(P[P.group == i].x, P[P.group == i].pc1_loading, s=8, color=TAB10[i], linewidths=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Female faces", "Male faces"]); ax.set_ylabel("Residual PC1 loading"); ax.axhline(0, color="k", lw=0.5, ls=":")
    savefig(fig, "london_resid_pc1", w=2.6, h=2.6, index_line=f"London Set: loading of each face on the first principal component of the rater x face residual matrix (ratings minus rater and face means), by face gender; |r| with face gender = {P.abs_r_face_gender.iloc[0]:.2f}; PC1 carries {P.pc1_variance_share.iloc[0]:.1%} of residual variance")
print("done")
