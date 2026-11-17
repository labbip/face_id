"""Face-ID (hybrid ratings version) for dense rater x item matrices with missing entries: Jester jokes (--src jester, text embedding) and
SCUT-FBP5500 faces (--src scut, CLIP embedding; --item_split random|asian2cauc|cauc2asian for cross-race transfer).
f_i = g_i + P e_i (free per-item embedding with dropout --g_drop, e_i = stimulus embedding, PCA --pcs on training items).
Double hold-out: rater 10-fold x item split. Evaluation on held-out raters (batched): z from n_est of their training items ->
Pearson r on their remaining training items ("seen") and held-out items ("unseen"); baselines: item mean over training raters (pop),
embedding ridge -> pop (pop_img), D0, direct z. Run: uv run --with openpyxl --with xlrd python -I scripts/65_dense_faceid.py --src jester --kind D1 --fold 0
"""
import argparse, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.effect import contested_items, rater_metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
p = argparse.ArgumentParser()
p.add_argument("--src", default="jester"); p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4); p.add_argument("--pcs", type=int, default=32)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--item_folds", type=int, default=5); p.add_argument("--item_split", default="random"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3); p.add_argument("--p_l2", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--g_drop", type=float, default=0.5); p.add_argument("--g_l2", type=float, default=1e-3); p.add_argument("--batch", type=int, default=2048)
p.add_argument("--n_est", default="10,20,50"); p.add_argument("--eval_only", action="store_true"); p.add_argument("--min_rat", type=int, default=50); p.add_argument("--out", default=None)
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; out = a.out or f"results/{a.src}"; os.makedirs(out, exist_ok=True); NEST = [int(x) for x in a.n_est.split(",")]
if a.src == "london":   # London Set attractiveness (2,513 raters x 102 faces, dense) with CLIP embedding; same folds as scripts/31
    r_ = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r_.iloc[:, 3:].to_numpy(np.float32); keep = R.std(1) > 0; R = R[keep]; rid = np.where(keep)[0]; E = np.load("results/london_embed_clip.npy"); groups = None
elif a.src == "jester":
    d = pd.read_excel("data/raw/jester/jokes_ratings.xls", header=None); R = d.iloc[:, 1:].to_numpy(np.float32); R[R == 99] = np.nan; keep = (~np.isnan(R)).sum(1) >= a.min_rat; R = R[keep]; rid = np.where(keep)[0]
    E = np.load("results/jester_embed_text.npy")[: R.shape[1]]; groups = None
elif a.src == "scut":
    d = pd.read_excel("data/raw/scut/unzip/SCUT-FBP5500_v2/All_Ratings.xlsx", sheet_name="ALL"); M_ = d.pivot(index="Rater", columns="Filename", values="Rating"); R = M_.to_numpy(np.float32); rid = M_.index.to_numpy()
    ids = np.load("results/scut_image_ids.npy", allow_pickle=True); assert list(ids) == list(M_.columns), "embedding order mismatch"; E = np.load("results/scut_embed_clip.npy"); groups = np.asarray(M_.columns.str[:1])   # A = Asian, C = Caucasian
V = ~np.isnan(R); N, F = R.shape; print(f"{a.src}: raters {N} items {F} observed {V.mean():.3f}")
fid = np.random.default_rng(0).permutation(N) % a.folds
if a.item_split == "random": ffid = np.random.default_rng(1).permutation(F) % a.item_folds; ftr, fte = np.where(ffid != a.fold % a.item_folds)[0], np.where(ffid == a.fold % a.item_folds)[0]
elif a.item_split == "asian2cauc": ftr, fte = np.where(groups == "A")[0], np.where(groups == "C")[0]
elif a.item_split == "cauc2asian": ftr, fte = np.where(groups == "C")[0], np.where(groups == "A")[0]
tr, te = np.where(fid != a.fold)[0], np.where(fid == a.fold)[0]
pca = PCA(min(a.pcs, len(ftr) - 1, E.shape[1]), random_state=0).fit(E[ftr]); X = pca.transform(E); X = (X - X[ftr].mean(0)) / (X[ftr].std(0) + 1e-6); Xt = torch.tensor(X, device=dev, dtype=torch.float32)
R0 = np.where(V, R, 0.0)
def zscore_rows(Rsub, Vsub):
    cnt = np.maximum(Vsub.sum(1, keepdims=True), 1); mu = (Rsub * Vsub).sum(1, keepdims=True) / cnt; sd = np.sqrt((((Rsub - mu) ** 2) * Vsub).sum(1, keepdims=True) / cnt) + 1e-6
    return np.where(Vsub, (Rsub - mu) / sd, 0.0).astype(np.float32)
Rz_tr = zscore_rows(R0[tr][:, ftr], V[tr][:, ftr]); V_tr = V[tr][:, ftr]
pop = R0[tr].sum(0) / np.maximum(V[tr].sum(0), 1); pop = np.where(V[tr].sum(0) > 0, pop, np.nanmean(pop))
pop_img = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(X[ftr], pop[ftr]).predict(X)
print(f"fold {a.fold} ({a.item_split}): raters train {len(tr)} test {len(te)}; items train {len(ftr)} test {len(fte)}; pop_img r on unseen items {np.corrcoef(pop_img[fte], pop[fte])[0,1]:.3f}", flush=True)

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * D + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, rz, mask):
        B = rz.shape[0]; fe = f[None].expand(B, -1, -1); x = torch.cat([fe, rz[..., None], rz[..., None] * fe], -1)
        h = s.net(x); al = s.att(h).squeeze(-1).masked_fill(~mask, -1e9); wt = torch.softmax(al, 1)
        return s.head((h * wt[..., None]).sum(1))
class Dec(nn.Module):
    def __init__(s, D, M, F, kind, h):
        super().__init__(); s.kind = kind; s.w0 = nn.Parameter(torch.randn(D) * 0.1); s.b0 = nn.Parameter(torch.zeros(1))
        if kind != "D0": s.bz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, 1))
        if kind == "D1": s.sz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind in ("D2", "D3"): s.wz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind == "D3": s.uz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, F))
    def forward(s, f, z):
        if s.kind == "D0": return (f @ s.w0)[None] + s.b0
        wv = s.w0[None] * Fn.softplus(s.sz(z) + 0.5413) if s.kind == "D1" else s.w0[None] + s.wz(z)
        o = wv @ f.T + s.bz(z); return o + s.uz(z) if s.kind == "D3" else o
class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.P = nn.Linear(X.shape[1], a.D); s.g = nn.Parameter(torch.zeros(F, a.D)); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, F, a.kind, a.hidden); s.g_mask = torch.ones(F, 1, device=dev)
    def f(s, drop=None):
        g = s.g * s.g_mask
        if drop is not None: g = g * (torch.rand(F, 1, device=dev) >= drop)
        return s.P(Xt) + g
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr); tag = f"{a.kind}_D{a.D}_M{a.M}_{a.item_split}_f{a.fold}_s{a.seed}"
if a.eval_only: m.load_state_dict(torch.load(f"{out}/model_{tag}.pt", map_location=dev)); a.epochs = 0
# contested items: between-rater SD of rater-centred ratings over training raters (all items they rated), top tercile
_Rc = R[tr] - np.nanmean(R[tr], 1, keepdims=True); _n = (~np.isnan(_Rc)).sum(0); _sd = np.nanstd(_Rc, 0); HI = contested_items({i: _sd[i] for i in range(F) if _n[i] >= 5})
Rt = torch.tensor(Rz_tr, device=dev); Vt = torch.tensor(V_tr, device=dev); ntr = len(tr); ftr_t = torch.tensor(ftr, device=dev); Ft = len(ftr)
for ep in range(a.epochs):
    perm = torch.randperm(ntr, device=dev); tot = 0.0
    for i in range(0, ntr, a.batch):
        idx = perm[i:i + a.batch]; rz = Rt[idx]; vv = Vt[idx]; B = len(idx)
        frac = torch.rand(B, 1, device=dev) * 0.5 + 0.05; mask = (torch.rand(B, Ft, device=dev) < frac) & vv
        f = m.f(drop=a.g_drop); f_tr = f[ftr_t]
        z = m.enc(f_tr, rz * mask, mask) if a.kind != "D0" else torch.zeros(B, a.M, device=dev)
        pred = m.dec(f, z)[:, ftr_t]; tgt = vv & ~mask
        loss = (((pred - rz) ** 2) * tgt).sum() / tgt.sum().clamp(min=1) + a.z_l2 * (z ** 2).sum(1).mean() + a.p_l2 * (m.P.weight ** 2).sum() + a.g_l2 * (m.g ** 2).sum(1).mean()      # per-item penalty (independent of the number of items)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * B
    if ep % 100 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {tot / ntr:.4f}", flush=True)

def fit_direct(f, rz, mask, steps=200, lr=0.05, l2=1e-2):          # batched over raters (independent z per rater)
    z = torch.zeros(rz.shape[0], a.M, device=dev, requires_grad=True); o = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        pred = m.dec(f, z)[:, ftr_t]; loss = ((((pred - rz) ** 2) * mask).sum(1) / mask.sum(1).clamp(min=1) + l2 * (z ** 2).sum(1)).sum()
        o.zero_grad(); loss.backward(); o.step()
    return z.detach()
def rows_r(P, Y, Mk):                                                # per-row Pearson r over masked entries (nan if < 10)
    Mk = Mk.astype(float); n = Mk.sum(1); P = np.where(Mk > 0, P, 0); Y = np.where(Mk > 0, Y, 0)
    mp, my = P.sum(1) / np.maximum(n, 1), Y.sum(1) / np.maximum(n, 1); dp, dy = (P - mp[:, None]) * Mk, (Y - my[:, None]) * Mk
    r = (dp * dy).sum(1) / (np.sqrt((dp ** 2).sum(1) * (dy ** 2).sum(1)) + 1e-12); r[n < 10] = np.nan; return r
m.eval(); rows = []
with torch.no_grad():
    m.g_mask[torch.tensor(fte, device=dev)] = 0; f = m.f(); f_tr = f[ftr_t]; fnp = f.cpu().numpy()
    Y = R0[te]; Vte = V[te]; n_te = len(te); g = np.random.default_rng(1000 + a.seed)
    for n_est in NEST:
        avail = Vte[:, ftr]; ok = avail.sum(1) >= n_est + 20
        if ok.sum() == 0: continue
        sc = np.where(avail, g.random(avail.shape), 2.0); order = np.argsort(sc, 1); est = np.zeros_like(avail); est[np.arange(n_te)[:, None], order[:, :n_est]] = True; est &= avail; est[~ok] = False
        ytr = Y[:, ftr]; cnt = np.maximum(est.sum(1, keepdims=True), 1); mu = (ytr * est).sum(1, keepdims=True) / cnt; sd = np.sqrt((((ytr - mu) ** 2) * est).sum(1, keepdims=True) / cnt) + 1e-6
        rz = torch.tensor(np.where(est, (ytr - mu) / sd, 0.0).astype(np.float32), device=dev); mk = torch.tensor(est, device=dev)
        seen = avail & ~est; uns = Vte[:, fte]
        row = dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, rater=rid[te], n_est=n_est, ok=ok, n_seen=seen.sum(1), n_unseen=uns.sum(1),
                   r_pop_seen=rows_r(np.broadcast_to(pop[ftr], ytr.shape), ytr, seen), r_pop_unseen=rows_r(np.broadcast_to(pop[fte], Y[:, fte].shape), Y[:, fte], uns), r_popimg_unseen=rows_r(np.broadcast_to(pop_img[fte], Y[:, fte].shape), Y[:, fte], uns))
        preds = {}
        z = m.enc(f_tr, rz * mk, mk) if a.kind != "D0" else torch.zeros(n_te, a.M, device=dev); preds["amort"] = np.broadcast_to(m.dec(f, z).cpu().numpy(), (n_te, F))   # D0 returns one row
        if a.kind != "D0":
            with torch.enable_grad(): zd = fit_direct(f, rz, mk.float())
            preds["direct"] = m.dec(f, zd).cpu().numpy()
            for j in range(a.M): row[f"z{j}"] = z[:, j].cpu().numpy()
        preds["pop"] = np.broadcast_to(np.where(np.arange(F)[None, :] < 0, 0, pop[None, :]), Y.shape)
        for nm, P in preds.items(): row[f"r_{nm}_seen"] = rows_r(P[:, ftr], ytr, seen); row[f"r_{nm}_unseen"] = rows_r(P[:, fte], Y[:, fte], uns)
        rng_ex = np.random.default_rng(a.seed * 7 + n_est)
        for nm, P in preds.items():
            for side, cols, mk_ in (("seen", ftr, seen), ("unseen", fte, uns)):
                base = pop[cols]; ex = {k_: np.full(n_te, np.nan) for k_ in ("rr", "rhi", "pacc")}   # dense data: training raters rated the model-unseen items too, so the item mean is the population baseline on both sides
                for r_i in range(n_te):
                    sel = mk_[r_i]
                    if sel.sum() < 10: continue
                    e_ = rater_metrics(P[r_i, cols][sel], Y[r_i, cols][sel], base[sel], cols[sel], HI, rng_ex)
                    for k_ in ex: ex[k_][r_i] = e_[k_]
                for k_, arr in ex.items(): row[f"{k_}_{nm}_{side}"] = arr
        rows.append(pd.DataFrame(row)[lambda d_: d_.ok].drop(columns="ok"))
df = pd.concat(rows); ex_cols = [c for c in df.columns if c.split("_")[0] in ("rr", "rhi", "pacc")] + ["r_pop_seen_dup"]; ex_cols = [c for c in ex_cols if c in df.columns]
df[["kind", "fold", "seed", "rater", "n_est"] + ex_cols].to_csv(f"{out}/eval2_{tag}.csv", index=False)
if not a.eval_only: df.drop(columns=ex_cols).to_csv(f"{out}/eval_{tag}.csv", index=False); torch.save(m.state_dict(), f"{out}/model_{tag}.pt"); np.save(f"{out}/items_{tag}.npy", fnp)
print(df.groupby("n_est")[[c for c in df.columns if c.startswith("r_")]].mean().round(3).to_string())
