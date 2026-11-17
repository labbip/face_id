"""Face-ID on the London Set (attractiveness ratings, 2,513 raters x 102 faces).

Two-factor model for scalar ratings: shared face embedding f_i (D) x perceiver latent z (M).
  D0: r = w0.f_i + b                      (population)
  D1: r = (w0 * softplus(s(z))).f_i + b(z) (same-sign reweighting of shared dims)
  D2: r = w(z).f_i + b(z)                 (free linear direction)
  D3: D2 + u_i(z)                         (face-specific offsets)
Amortised encoder (set of (f_i, r_i) -> z) trained with random enc/target face splits per rater.
Evaluation: held-out raters, n_est random faces -> z -> Pearson/Spearman on the remaining faces.
Baselines: population mean (train raters) with per-rater affine on est faces; direct z optimisation.
Run: uv run python -I scripts/23_london_faceid.py --kind D1 --fold 0
"""
import argparse, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.stats import spearmanr
p = argparse.ArgumentParser()
p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=300); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--out", default="results/london")
a = p.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
import os; os.makedirs(a.out, exist_ok=True)

r = pd.read_csv("data/raw/london/london_faces_ratings.csv"); R = r.iloc[:, 3:].to_numpy(np.float32)
N, F = R.shape
fid = np.random.default_rng(0).permutation(N) % a.folds            # fold assignment fixed across runs
tr, te = np.where(fid != a.fold)[0], np.where(fid == a.fold)[0]
sd = R.std(1); keep_tr = tr[sd[tr] > 0]                              # drop constant raters from training
Rz = (R - R.mean(1, keepdims=True)) / np.where(sd[:, None] > 0, sd[:, None], 1)   # per-rater z-score (train)
pop = R[keep_tr].mean(0)                                             # population mean rating per face

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * D + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, rz, mask):            # f (F,D); rz (B,F); mask (B,F) bool
        B = rz.shape[0]; fe = f[None].expand(B, -1, -1)
        x = torch.cat([fe, rz[..., None], rz[..., None] * fe], -1)
        h = s.net(x); al = s.att(h).squeeze(-1).masked_fill(~mask, -1e9); w = torch.softmax(al, 1)
        return s.head((h * w[..., None]).sum(1))
class Dec(nn.Module):
    def __init__(s, D, M, F, kind, h):
        super().__init__(); s.kind = kind; s.w0 = nn.Parameter(torch.randn(D) * 0.1); s.b0 = nn.Parameter(torch.zeros(1))
        if kind != "D0": s.bz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, 1))
        if kind == "D1": s.sz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind in ("D2", "D3"): s.wz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind == "D3": s.uz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, F))
    def forward(s, f, z):                   # -> (B,F)
        if s.kind == "D0": return (f @ s.w0)[None] + s.b0
        if s.kind == "D1": w = s.w0[None] * Fn.softplus(s.sz(z) + 0.5413)   # softplus(0.5413)=1
        else: w = s.w0[None] + s.wz(z)
        out = w @ f.T + s.bz(z)
        if s.kind == "D3": out = out + s.uz(z)
        return out
class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Parameter(torch.randn(F, a.D) * 0.3); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, F, a.kind, a.hidden)
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
Rt = torch.tensor(Rz[keep_tr], device=dev); ntr = len(keep_tr)
for ep in range(a.epochs):
    perm = torch.randperm(ntr, device=dev); tot = 0
    for i in range(0, ntr, 128):
        rz = Rt[perm[i:i + 128]]; B = rz.shape[0]
        frac = torch.rand(B, 1, device=dev) * 0.5 + 0.05                   # 5-55% of faces for encoding
        mask = torch.rand(B, F, device=dev) < frac
        mask[torch.arange(B), torch.randint(0, F, (B,))] = True             # at least one
        z = m.enc(m.f, rz * mask, mask) if a.kind != "D0" else torch.zeros(B, a.M, device=dev)
        pred = m.dec(m.f, z); tgt = ~mask
        loss = (((pred - rz) ** 2) * tgt).sum() / tgt.sum() + a.z_l2 * (z ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * B
    if ep % 50 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {tot / ntr:.4f}", flush=True)

def fit_direct(rz, mask, steps=200, lr=0.05, l2=1e-2):
    z = torch.zeros(rz.shape[0], a.M, device=dev, requires_grad=True); o = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        pred = m.dec(m.f, z); loss = (((pred - rz) ** 2) * mask).sum(1) / mask.sum(1) + l2 * (z ** 2).sum(1)
        o.zero_grad(); loss.sum().backward(); o.step()
    return z.detach()

m.eval(); rows = []
with torch.no_grad():
    for n_est in (5, 10, 20, 40):
        g = np.random.default_rng(1000 + a.seed)
        for k in te:
            if sd[k] == 0: continue
            est = g.choice(F, n_est, replace=False); mask = np.zeros(F, bool); mask[est] = True
            y = R[k]; mu, s_ = y[est].mean(), y[est].std() + 1e-6
            rz = torch.tensor(((y - mu) / s_)[None], device=dev, dtype=torch.float32); mk = torch.tensor(mask[None], device=dev)
            row = dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, rater=int(k), n_est=n_est)
            # population baseline (affine on est faces)
            p0 = pop                                              # correlation metrics are affine-invariant
            row["r_pop"] = np.corrcoef(p0[~mask], y[~mask])[0, 1]; row["rho_pop"] = spearmanr(p0[~mask], y[~mask])[0]
            z = m.enc(m.f, rz * mk, mk) if a.kind != "D0" else torch.zeros(1, a.M, device=dev)
            pr = m.dec(m.f, z)[0].cpu().numpy()
            row["r_amort"] = np.corrcoef(pr[~mask], y[~mask])[0, 1]; row["rho_amort"] = spearmanr(pr[~mask], y[~mask])[0]
            if a.kind != "D0":
                with torch.enable_grad(): zd = fit_direct(rz, mk.float())
                pd_ = m.dec(m.f, zd)[0].cpu().numpy()
                row["r_direct"] = np.corrcoef(pd_[~mask], y[~mask])[0, 1]; row["rho_direct"] = spearmanr(pd_[~mask], y[~mask])[0]
                row.update({f"z{j}": float(z[0, j]) for j in range(a.M)})
            rows.append(row)
df = pd.DataFrame(rows); tag = f"{a.kind}_D{a.D}_M{a.M}_f{a.fold}_s{a.seed}"
df.to_csv(f"{a.out}/eval_{tag}.csv", index=False); torch.save(m.state_dict(), f"{a.out}/model_{tag}.pt")
np.save(f"{a.out}/faces_{tag}.npy", m.f.detach().cpu().numpy())
print(df.groupby("n_est")[[c for c in df.columns if c.startswith("r_") or c.startswith("rho_")]].mean().round(3).to_string())
