"""Face-ID (ratings version) on LAPIS art aesthetics with CLIP image embeddings, sparse ratings (rater x image triples).
f_i = P e_i (CLIP -> PCA --pcs -> D). Decoders: D0 population, D1 same-sign reweighting, D2 free linear direction.
Double hold-out: rater 10-fold x image 5-fold (image fold = rater fold % 5). Training uses training raters x training images.
Evaluation on held-out raters, z from n_est of their ratings of training images:
  seen  : Pearson r on their remaining ratings of training images   (baseline: image mean over training raters)
  unseen: Pearson r on their ratings of held-out images             (baseline: ridge image -> training-rater mean)
Run: uv run python -I scripts/53_lapis_faceid_img.py --kind D1 --fold 0
"""
import argparse, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.effect import item_disagreement, contested_items, metrics_long
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
p = argparse.ArgumentParser()
p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4); p.add_argument("--pcs", type=int, default=64)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--img_folds", type=int, default=5); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3); p.add_argument("--p_l2", type=float, default=1e-4)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--min_rat", type=int, default=100); p.add_argument("--out", default="results/lapis_img")
p.add_argument("--max_raters", type=int, default=0); p.add_argument("--eval_only", action="store_true"); p.add_argument("--eval_write", action="store_true");   # eval_write: also write eval_*.csv in --eval_only mode (transfer evaluation of existing models on another data file)
p.add_argument("--data", default="data/raw/lapis/piaa.csv"); p.add_argument("--ids", default="results/lapis_image_ids.npy"); p.add_argument("--embed", default="results/lapis_embed_clip.npy")   # generic sparse-rating datasets (e.g. AADB)
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
d = pd.read_csv(a.data, low_memory=False).groupby(["participant_id", "image_id"], as_index=False).rating.mean()
n = d.groupby("participant_id").size(); d = d[d.participant_id.isin(n[n >= a.min_rat].index)].reset_index(drop=True)
ids = np.load(a.ids, allow_pickle=True); imap = {str(m): i for i, m in enumerate(ids)}; d["i"] = d.image_id.astype(str).map(imap); d = d.dropna(subset=["i"]); d["i"] = d.i.astype(int); F = len(ids)
raters = sorted(d.participant_id.unique())
if a.max_raters and a.max_raters < len(raters):   # power analysis: random subset of raters (depends on --seed)
    raters = sorted(np.random.default_rng(100 + a.seed).choice(raters, a.max_raters, replace=False)); d = d[d.participant_id.isin(raters)].reset_index(drop=True)
N = len(raters); d["k"] = d.participant_id.map({r: i for i, r in enumerate(raters)})
rng0 = np.random.default_rng(0); rfold = rng0.permutation(N) % a.folds; ifold = np.random.default_rng(1).permutation(F) % a.img_folds
d["rf"] = rfold[d.k]; d["if"] = ifold[d.i]; itr = np.where(ifold != a.fold % a.img_folds)[0]; ite = np.where(ifold == a.fold % a.img_folds)[0]
tr = d[(d.rf != a.fold) & (d["if"] != a.fold % a.img_folds)]; te = d[d.rf == a.fold]
E = np.load(a.embed); pca = PCA(a.pcs, random_state=0).fit(E[itr]); X = pca.transform(E); X = (X - X[itr].mean(0)) / X[itr].std(0); Xt = torch.tensor(X, device=dev, dtype=torch.float32)
# per-rater z-scores over the rater's training ratings
rs = tr.groupby("k").rating.agg(["mean", "std"]); tr = tr.assign(rz=(tr.rating - rs["mean"].reindex(tr.k).to_numpy()) / (rs["std"].reindex(tr.k).to_numpy() + 1e-6))
img_mean = tr.groupby("i").rating.mean(); pop = img_mean.reindex(range(F)).to_numpy()          # NaN for held-out images
pop_img = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(X[itr][~np.isnan(pop[itr])], pop[itr][~np.isnan(pop[itr])]).predict(X)
print(f"fold {a.fold}: raters train {tr.k.nunique()} test {te.k.nunique()}; images train {len(itr)} test {len(ite)}; train ratings {len(tr)}", flush=True)

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * D + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, i, rz, kidx, n):     # sets of (image, z-scored rating) per rater slot
        fe = f[i]; x = torch.cat([fe, rz[:, None], rz[:, None] * fe], -1); h = s.net(x)
        al = s.att(h).squeeze(-1); al = al - al.max(); w = torch.exp(al); den = torch.zeros(n, device=f.device).index_add_(0, kidx, w) + 1e-12; w = w / den[kidx]
        return s.head(torch.zeros(n, h.shape[1], device=f.device).index_add_(0, kidx, h * w[:, None]))
class Dec(nn.Module):
    def __init__(s, D, M, kind, h):
        super().__init__(); s.kind = kind; s.w0 = nn.Parameter(torch.randn(D) * 0.1); s.b0 = nn.Parameter(torch.zeros(1))
        if kind != "D0": s.bz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, 1))
        if kind == "D1": s.sz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind == "D2": s.wz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
    def forward(s, f, z, i, kidx):           # -> predicted z-scored rating per (rater, image) pair
        if s.kind == "D0": return f[i] @ s.w0 + s.b0
        w = s.w0[None] * Fn.softplus(s.sz(z) + 0.5413) if s.kind == "D1" else s.w0[None] + s.wz(z)
        return (w[kidx] * f[i]).sum(-1) + s.bz(z)[kidx, 0]
class Model(nn.Module):
    def __init__(s): super().__init__(); s.P = nn.Linear(a.pcs, a.D); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, a.kind, a.hidden)
    def f(s): return s.P(Xt)
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr); tag = f"{a.kind}_D{a.D}_M{a.M}_f{a.fold}_s{a.seed}"
if a.eval_only: m.load_state_dict(torch.load(f"{a.out}/model_{tag}.pt", map_location=dev)); a.epochs = 0
# contested items: between-rater SD of rater-centred ratings over training raters (all images they rated), top tercile
_tr_all = d[d.rf != a.fold]; _c = _tr_all.rating - _tr_all.groupby("k").rating.transform("mean"); HI = contested_items(item_disagreement(_c.to_numpy(), _tr_all.i.to_numpy(), _tr_all.k.to_numpy()))
tr_k = np.sort(tr.k.unique()); slot = {k: j for j, k in enumerate(tr_k)}; n_tr = len(tr_k)
i_t = torch.tensor(tr.i.to_numpy(), device=dev); r_t = torch.tensor(tr.rz.to_numpy(), device=dev, dtype=torch.float32); k_t = torch.tensor(tr.k.map(slot).to_numpy(), device=dev)
gen = torch.Generator(device=dev).manual_seed(a.seed)
for ep in range(a.epochs):
    rate = (torch.rand(n_tr, device=dev, generator=gen) * 0.5 + 0.05)[k_t]; enc = torch.rand(len(k_t), device=dev, generator=gen) < rate
    f = m.f(); z = m.enc(f, i_t[enc], r_t[enc], k_t[enc], n_tr) if a.kind != "D0" else torch.zeros(n_tr, a.M, device=dev)
    pred = m.dec(f, z, i_t[~enc], k_t[~enc]); loss = ((pred - r_t[~enc]) ** 2).mean() + a.z_l2 * (z ** 2).sum(1).mean() + a.p_l2 * (m.P.weight ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {loss.item():.4f}", flush=True)
def fit_direct(f, i, rz, kidx, n, steps=200, lr=0.05, l2=1e-2):
    z = torch.zeros(n, a.M, device=dev, requires_grad=True); op = torch.optim.Adam([z], lr=lr); cnt = torch.bincount(kidx, minlength=n).clamp(min=1)
    for _ in range(steps):
        e = (m.dec(f, z, i, kidx) - rz) ** 2; loss = torch.zeros(n, device=dev).index_add_(0, kidx, e) / cnt + l2 * (z ** 2).sum(1)
        op.zero_grad(); loss.sum().backward(); op.step()
    return z.detach()
def per_rater_r(pred, y, kidx, n):
    out = np.full(n, np.nan)
    for k in range(n):
        mk = kidx == k
        if mk.sum() >= 10 and y[mk].std() > 0 and pred[mk].std() > 0: out[k] = np.corrcoef(pred[mk], y[mk])[0, 1]
    return out
m.eval(); rows = []
te_k = np.sort(te.k.unique()); slot = {k: j for j, k in enumerate(te_k)}; n_te = len(te_k); ted = te.assign(k=te.k.map(slot))
seen_pool = ted[ted["if"] != a.fold % a.img_folds]; unseen = ted[ted["if"] == a.fold % a.img_folds]
g = np.random.default_rng(1000 + a.seed)
with torch.no_grad():
    f = m.f()
    for n_est in (10, 20, 50, 100):
        sh = seen_pool.sample(frac=1, random_state=int(g.integers(1e9))); est = sh[sh.groupby("k").cumcount() < n_est]; rest = sh[sh.groupby("k").cumcount() >= n_est]
        mu_s = est.groupby("k").rating.mean(); sd_s = est.groupby("k").rating.std() + 1e-6
        rz_e = torch.tensor(((est.rating - mu_s.reindex(est.k).to_numpy()) / sd_s.reindex(est.k).to_numpy()).to_numpy(), device=dev, dtype=torch.float32)
        i_e = torch.tensor(est.i.to_numpy(), device=dev); k_e = torch.tensor(est.k.to_numpy(), device=dev)
        z = m.enc(f, i_e, rz_e, k_e, n_te) if a.kind != "D0" else torch.zeros(n_te, a.M, device=dev)
        row = dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, rater=[raters[k] for k in te_k], n_est=n_est)
        for name, sub in (("seen", rest), ("unseen", unseen)):
            i_s = torch.tensor(sub.i.to_numpy(), device=dev); k_s = torch.tensor(sub.k.to_numpy(), device=dev); y = sub.rating.to_numpy(); kk = sub.k.to_numpy()
            base = pop[sub.i.to_numpy()] if name == "seen" else pop_img[sub.i.to_numpy()]
            row[f"r_pop_{name}"] = per_rater_r(base, y, kk, n_te); preds = {"pop": base, "amort": m.dec(f, z, i_s, k_s).cpu().numpy()}
            row[f"r_amort_{name}"] = per_rater_r(preds["amort"], y, kk, n_te)
            if a.kind != "D0":
                with torch.enable_grad(): zd = fit_direct(f, i_e, rz_e, k_e, n_te)
                preds["direct"] = m.dec(f, zd, i_s, k_s).cpu().numpy(); row[f"r_direct_{name}"] = per_rater_r(preds["direct"], y, kk, n_te)
            for nm, pr in preds.items():
                ex = metrics_long(pr, y, base, kk, sub.i.to_numpy(), HI, n_te, seed=a.seed * 7 + n_est)
                for key, arr in ex.items(): row[f"{key}_{nm}_{name}"] = arr
        if a.kind != "D0":
            for j in range(a.M): row[f"z{j}"] = z[:, j].cpu().numpy()
        rows.append(pd.DataFrame(row))
df = pd.concat(rows); ex_cols = [c for c in df.columns if c.split("_")[0] in ("rr", "rhi", "pacc")]
df.drop(columns=ex_cols).to_csv(f"{a.out}/eval_{tag}.csv", index=False) if (not a.eval_only or a.eval_write) else None; df[["kind", "fold", "seed", "rater", "n_est"] + ex_cols].to_csv(f"{a.out}/eval2_{tag}.csv", index=False)
if not a.eval_only: torch.save(m.state_dict(), f"{a.out}/model_{tag}.pt")
print(df.groupby("n_est")[[c for c in df.columns if c.startswith("r_")]].mean().round(3).to_string())
