"""T2 (claim revision, 2026-10-29): can the learned D1 basis express a reversal of a content axis through duplicated dimensions of opposite sign?
For each D1 fold model (LAPIS, EVA; results/<src>/model_D1_D8_M4_f<fold>_s<seed>.pt) in the fold's stimulus PCA-64 basis (as in 53):
  content direction u = unit ridge direction predicting the content indicator from X (LAPIS: abstract vs figurative; EVA: category c in 1..6);
  a_j = w0_j * (P_j . u): contribution of embedding dim j to the readout along u when its D1 multiplier is 1.
  pos = sum max(a_j, 0), neg = sum max(-a_j, 0); dup = neg / (pos + neg)  (0 = no opposite-sign capacity; 0.5 = fully duplicated).
  Realised per-rater D1 weights w_k = w0 * softplus(sz(z_k) + 0.5413) for held-out raters (z at n_est = 100, eval_D1 csv) -> readout along u
  p_k = (w_k @ P) . u; fraction of raters with sign(p_k) != sign(p_pop), p_pop = (w0 @ P) . u.  Same for the D2 model (w_k = w0 + wz(z_k)) for comparison.
Outputs results/dup_axis_diag.csv.  Run: uv run python -I scripts/80_duplicate_axis_diag.py
"""
import numpy as np, pandas as pd, torch
from scipy.special import erf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2))); softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
CFG = {"lapis": ("results/lapis_img", "results/lapis_image_ids.npy", "results/lapis_embed_clip.npy"), "eva": ("results/eva_img", "results/eva_image_ids.npy", "results/eva_embed_clip.npy")}
sty = pd.read_csv("data/raw/lapis/piaa.csv", usecols=["image_id", "2_styles"], low_memory=False).drop_duplicates("image_id").set_index("image_id")["2_styles"]
cat = pd.read_csv("data/raw/eva/image_content_category.csv", dtype=str).rename(columns={"sort": "cat"}).set_index("image_id")["cat"]
rows = []
for ds, (SRC, IDS, EMB) in CFG.items():
    ids = np.load(IDS, allow_pickle=True); E = np.load(EMB); Fn = len(ids); ifold = np.random.default_rng(1).permutation(Fn) % 5
    content = {"abstract": (sty.reindex(ids).to_numpy() == "ABSTRACT").astype(float)} if ds == "lapis" else {f"cat{c}": (cat.reindex(ids.astype(str)).to_numpy() == c).astype(float) for c in "123456"}
    for fold in range(10):
        itr = np.where(ifold != fold % 5)[0]; pca = PCA(64, random_state=0).fit(E[itr]); X = pca.transform(E); X = (X - X[itr].mean(0)) / X[itr].std(0)
        U = {c: Ridge(alpha=10.0).fit(X[itr], y[itr]).coef_ for c, y in content.items()}; U = {c: u / np.linalg.norm(u) for c, u in U.items()}
        for s in (0, 1):
            out = {}
            for kind in ("D1", "D2"):
                sd = torch.load(f"{SRC}/model_{kind}_D8_M4_f{fold}_s{s}.pt", map_location="cpu"); P = sd["P.weight"].numpy(); w0 = sd["dec.w0"].numpy()
                z = pd.read_csv(f"{SRC}/eval_{kind}_D8_M4_f{fold}_s{s}.csv"); z = z[z.n_est == 100].groupby("rater")[[f"z{j}" for j in range(4)]].mean().to_numpy()
                W1, b1, W2, b2 = (sd[f"dec.{'sz' if kind == 'D1' else 'wz'}.{i}"].numpy() for i in ("0.weight", "0.bias", "2.weight", "2.bias")); h = gelu(z @ W1.T + b1) @ W2.T + b2
                Wk = w0[None] * softplus(h + 0.5413) if kind == "D1" else w0[None] + h; Wpc = Wk @ P; wpop_pc = w0 @ P
                for c, u in U.items():
                    a = w0 * (P @ u); pos, neg = a[a > 0].sum(), -a[a < 0].sum(); pk = Wpc @ u; pp = wpop_pc @ u
                    out[(c, kind)] = dict(dup=neg / (pos + neg), pos=pos, neg=neg, p_pop=pp, frac_rev=(np.sign(pk) != np.sign(pp)).mean(), sd_pk=pk.std(), n=len(pk), pop_over_sd=abs(pp) / pk.std())
            for c in U:
                r = dict(ds=ds, fold=fold, seed=s, content=c); r.update({f"D1_{k}": v for k, v in out[(c, "D1")].items()}); r.update({f"D2_{k}": v for k, v in out[(c, "D2")].items() if k in ("frac_rev", "pop_over_sd", "sd_pk")}); rows.append(r)
T = pd.DataFrame(rows); T.to_csv("results/dup_axis_diag.csv", index=False)
pd.set_option("display.width", 200); print(T.groupby(["ds", "content"])[["D1_dup", "D1_pos", "D1_neg", "D1_pop_over_sd", "D1_frac_rev", "D2_pop_over_sd", "D2_frac_rev"]].agg(["median", "min", "max"]).round(3).to_string())
