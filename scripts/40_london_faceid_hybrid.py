"""Hybrid Face-ID on the London Set: f_i = g_i + P e_i, free per-face embedding g_i (dropped out with prob --g_drop per face
and batch during training; set to 0 for unseen faces) plus CLIP image projection P e_i (PCA --pcs, fitted on training faces).
Double hold-out: rater fold (10) x face fold (5; face fold = rater fold % 5). Training uses training raters x training faces only.
Evaluation on held-out raters: z from n_est training faces -> Pearson r on (a) remaining training faces ("seen") and (b) held-out faces ("unseen").
Baselines: population image regression (ridge E -> training-rater mean, fitted on training faces), D0, direct z.
Run: uv run python -I scripts/40_london_faceid_hybrid.py --kind D1 --fold 0
"""
import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
p = argparse.ArgumentParser()
p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4); p.add_argument("--pcs", type=int, default=32)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--face_folds", type=int, default=5); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3); p.add_argument("--p_l2", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--embed", default="clip"); p.add_argument("--out", default="results/london_hyb"); p.add_argument("--g_drop", type=float, default=0.5); p.add_argument("--g_l2", type=float, default=1e-3)
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(np.float32); N, F = R.shape
E = np.load(f"results/london_embed_{a.embed}.npy")
fid = np.random.default_rng(0).permutation(N) % a.folds; ffid = np.random.default_rng(1).permutation(F) % a.face_folds
tr, te = np.where(fid != a.fold)[0], np.where(fid == a.fold)[0]; ftr, fte = np.where(ffid != a.fold % a.face_folds)[0], np.where(ffid == a.fold % a.face_folds)[0]
sd = R.std(1); keep_tr = tr[sd[tr] > 0]
pca = PCA(a.pcs, random_state=0).fit(E[ftr]); X = pca.transform(E); X = (X - X[ftr].mean(0)) / X[ftr].std(0); Xt = torch.tensor(X, device=dev, dtype=torch.float32)
Rz_tr = R[keep_tr][:, ftr]; Rz_tr = (Rz_tr - Rz_tr.mean(1, keepdims=True)) / (Rz_tr.std(1, keepdims=True) + 1e-6)   # z-score over training faces
pop = R[keep_tr].mean(0)
pop_img = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(X[ftr], pop[ftr]).predict(X)                  # image -> population baseline
print(f"fold {a.fold}: raters train {len(keep_tr)} test {len(te)}; faces train {len(ftr)} test {len(fte)}; pop_img r on unseen faces {np.corrcoef(pop_img[fte], pop[fte])[0,1]:.3f}")

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * D + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, rz, mask):
        B = rz.shape[0]; fe = f[None].expand(B, -1, -1); x = torch.cat([fe, rz[..., None], rz[..., None] * fe], -1)
        h = s.net(x); al = s.att(h).squeeze(-1).masked_fill(~mask, -1e9); w = torch.softmax(al, 1)
        return s.head((h * w[..., None]).sum(1))
class Dec(nn.Module):
    def __init__(s, D, M, F, kind, h):
        super().__init__(); s.kind = kind; s.w0 = nn.Parameter(torch.randn(D) * 0.1); s.b0 = nn.Parameter(torch.zeros(1))
        if kind != "D0": s.bz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, 1))
        if kind == "D1": s.sz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind in ("D2", "D3"): s.wz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind == "D3": s.uz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, F))
    def forward(s, f, z):
        if s.kind == "D0": return (f @ s.w0)[None] + s.b0
        w = s.w0[None] * Fn.softplus(s.sz(z) + 0.5413) if s.kind == "D1" else s.w0[None] + s.wz(z)
        out = w @ f.T + s.bz(z)
        return out + s.uz(z) if s.kind == "D3" else out
class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.P = nn.Linear(a.pcs, a.D); s.g = nn.Parameter(torch.zeros(F, a.D)); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, F, a.kind, a.hidden)
        s.g_mask = torch.ones(F, 1, device=dev)
    def f(s, drop=None):
        g = s.g * s.g_mask
        if drop is not None: g = g * (torch.rand(F, 1, device=dev) >= drop)
        return s.P(Xt) + g
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
Rt = torch.tensor(Rz_tr, device=dev); ntr = len(keep_tr); ftr_t = torch.tensor(ftr, device=dev); Ft = len(ftr)
for ep in range(a.epochs):
    perm = torch.randperm(ntr, device=dev); tot = 0
    for i in range(0, ntr, 128):
        rz = Rt[perm[i:i + 128]]; B = rz.shape[0]
        frac = torch.rand(B, 1, device=dev) * 0.5 + 0.05; mask = torch.rand(B, Ft, device=dev) < frac
        mask[torch.arange(B), torch.randint(0, Ft, (B,))] = True
        f = m.f(drop=a.g_drop); f_tr = f[ftr_t]
        z = m.enc(f_tr, rz * mask, mask) if a.kind != "D0" else torch.zeros(B, a.M, device=dev)
        pred = m.dec(f, z)[:, ftr_t]; tgt = ~mask
        loss = (((pred - rz) ** 2) * tgt).sum() / tgt.sum() + a.z_l2 * (z ** 2).sum(1).mean() + a.p_l2 * (m.P.weight ** 2).sum() + a.g_l2 * (m.g ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * B
    if ep % 100 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {tot / ntr:.4f}", flush=True)

def fit_direct(f, rz, mask, steps=200, lr=0.05, l2=1e-2):
    z = torch.zeros(1, a.M, device=dev, requires_grad=True); o = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        pred = m.dec(f, z)[:, ftr_t]; loss = (((pred - rz) ** 2) * mask).sum() / mask.sum() + l2 * (z ** 2).sum()
        o.zero_grad(); loss.backward(); o.step()
    return z.detach()
def corr(x, y): return np.corrcoef(x, y)[0, 1] if np.std(y) > 0 and np.std(x) > 0 else np.nan
m.eval(); rows = []
with torch.no_grad():
    m.g_mask[torch.tensor(fte, device=dev)] = 0; f = m.f(); f_tr = f[ftr_t]
    for n_est in (5, 10, 20, 40):
        g = np.random.default_rng(1000 + a.seed)
        for k in te:
            if sd[k] == 0: continue
            est = g.choice(Ft, n_est, replace=False); mask = np.zeros(Ft, bool); mask[est] = True
            y = R[k]; ytr = y[ftr]; mu, s_ = ytr[est].mean(), ytr[est].std() + 1e-6
            rz = torch.tensor(((ytr - mu) / s_)[None], device=dev, dtype=torch.float32); mk = torch.tensor(mask[None], device=dev)
            seen = ftr[~mask]; row = dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, rater=int(k), n_est=n_est,
                r_pop_seen=corr(pop[seen], y[seen]), r_pop_unseen=corr(pop[fte], y[fte]), r_popimg_unseen=corr(pop_img[fte], y[fte]))
            z = m.enc(f_tr, rz * mk, mk) if a.kind != "D0" else torch.zeros(1, a.M, device=dev)
            pr = m.dec(f, z)[0].cpu().numpy(); row["r_amort_seen"] = corr(pr[seen], y[seen]); row["r_amort_unseen"] = corr(pr[fte], y[fte])
            if a.kind != "D0":
                with torch.enable_grad(): zd = fit_direct(f, rz, mk.float())
                pd_ = m.dec(f, zd)[0].cpu().numpy(); row["r_direct_seen"] = corr(pd_[seen], y[seen]); row["r_direct_unseen"] = corr(pd_[fte], y[fte])
                row.update({f"z{j}": float(z[0, j]) for j in range(a.M)})
            rows.append(row)
df = pd.DataFrame(rows); tag = f"{a.kind}_D{a.D}_M{a.M}_{a.embed}_gd{a.g_drop}_f{a.fold}_s{a.seed}"
df.to_csv(f"{a.out}/eval_{tag}.csv", index=False); torch.save(m.state_dict(), f"{a.out}/model_{tag}.pt"); np.save(f"{a.out}/faces_{tag}.npy", f.cpu().numpy())
print(df.groupby("n_est")[[c for c in df.columns if c.startswith("r_")]].mean().round(3).to_string())
