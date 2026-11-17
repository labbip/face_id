"""Face-ID on AVFS odd-one-out triplets (novel-stimuli set: 56 faces, 632 annotators, ~127 triplets each).
Shared face embedding f (D) x annotator latent z (M); similarity s_ij and odd-one-out = softmax over the 3 pair similarities.
  D0: s = f_i.f_j                     D1: s = sum_d w_d(z) f_id f_jd (softplus, init 1)
  D2: s = (A(z) f_i).(A(z) f_j)       D3: D2 with f_i + u_i(z)
Amortised encoder over the annotator's estimation triplets; annotator 10-fold CV; evaluation on each held-out annotator's
test triplets with z estimated from n_est estimation triplets. Baselines: D0 (population), direct z optimisation.
Run: uv run python -I scripts/28_avfs_faceid.py --kind D1 --fold 0
"""
import argparse, json, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
p = argparse.ArgumentParser()
p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=400); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--min_trip", type=int, default=60); p.add_argument("--out", default="results/avfs"); p.add_argument("--path", default=None)
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
D_ = "data/raw/avfs/unzip/avfs-dataset-v1/"; tv = json.load(open(a.path if a.path else D_ + "ooo_test_novel_stimuli.json"))
rows = [(t["annotator_id"], *q, pos) for t in tv.values() for q, pos in zip(t["triplet_questions"], t["odd_one_out_positions"])]
df = pd.DataFrame(rows, columns=["ann", "a", "b", "c", "odd"])
faces = {f: i for i, f in enumerate(sorted(set(df.a) | set(df.b) | set(df.c)))}; F = len(faces)
for c in "abc": df[c] = df[c].map(faces)
cnt = df.ann.value_counts(); df = df[df.ann.isin(cnt[cnt >= a.min_trip].index)].reset_index(drop=True)
anns = sorted(df.ann.unique()); A = len(anns); df["k"] = df.ann.map({k: i for i, k in enumerate(anns)})
rng0 = np.random.default_rng(0); fid = rng0.permutation(A) % a.folds; df["est"] = rng0.random(len(df)) < 0.5   # fixed across runs
df["fold"] = fid[df.k]
# reorder each triplet as (odd, p, q): the judged-similar pair is (p, q)
T = df[["a", "b", "c"]].to_numpy(); o = df.odd.to_numpy()
odd = T[np.arange(len(T)), o]; rest = np.array([[t[j] for j in range(3) if j != oo] for t, oo in zip(T, o)])
df["o"], df["p"], df["q"] = odd, rest[:, 0], rest[:, 1]
tr = df[df.fold != a.fold]; te = df[df.fold == a.fold]
print(f"faces {F} annotators {A} (train {tr.k.nunique()}, test {te.k.nunique()}), triplets {len(df)}")

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(3 * D + 3, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, o, pq, kidx, n):     # o (T,), pq (T,2), kidx (T,) annotator slot -> z (n, M)
        fo, fp, fq = f[o], f[pq[:, 0]], f[pq[:, 1]]
        x = torch.cat([fo, fp + fq, (fp - fq).abs(), (fp * fq).sum(-1, keepdim=True), (fo * fp).sum(-1, keepdim=True), (fo * fq).sum(-1, keepdim=True)], -1)
        h = s.net(x); al = s.att(h).squeeze(-1); al = al - al.max(); w = torch.exp(al)
        den = torch.zeros(n, device=f.device).index_add_(0, kidx, w) + 1e-12; w = w / den[kidx]
        return s.head(torch.zeros(n, h.shape[1], device=f.device).index_add_(0, kidx, h * w[:, None]))
class Dec(nn.Module):
    def __init__(s, D, M, F, kind, h):
        super().__init__(); s.kind, s.D = kind, D
        if kind == "D1": s.sz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D))
        if kind in ("D2", "D3"):
            s.az = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, D * D))
            with torch.no_grad(): s.az[-1].weight.mul_(0.1); s.az[-1].bias.copy_(torch.eye(D).flatten())
        if kind == "D3": s.uz = nn.Sequential(nn.Linear(M, h), nn.GELU(), nn.Linear(h, F * D)); nn.init.zeros_(s.uz[-1].bias); s.uz[-1].weight.data.mul_(0.1)
    def logits(s, f, z, o, pq, kidx):      # -> (T, 3): [s_pq, s_oq, s_op]; target index 0 (odd = o)
        fo, fp, fq = f[o], f[pq[:, 0]], f[pq[:, 1]]
        if s.kind == "D0": sim = lambda x, y: (x * y).sum(-1)
        elif s.kind == "D1": w = Fn.softplus(s.sz(z) + 0.5413)[kidx]; sim = lambda x, y: (w * x * y).sum(-1)
        else:
            Am = s.az(z).view(-1, s.D, s.D)[kidx]
            if s.kind == "D3":
                U = s.uz(z).view(-1, F, s.D); fo, fp, fq = fo + U[kidx, o], fp + U[kidx, pq[:, 0]], fq + U[kidx, pq[:, 1]]
            fo, fp, fq = (torch.einsum("tij,tj->ti", Am, v) for v in (fo, fp, fq)); sim = lambda x, y: (x * y).sum(-1)
        return torch.stack([sim(fp, fq), sim(fo, fq), sim(fo, fp)], 1)
class Model(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Parameter(torch.randn(F, a.D) * 0.3); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, F, a.kind, a.hidden)
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
def tens(d): return (torch.tensor(d.o.to_numpy(), device=dev), torch.tensor(d[["p", "q"]].to_numpy(), device=dev), torch.tensor(d.k.to_numpy(), device=dev))
tr_k = np.sort(tr.k.unique()); slot = {k: i for i, k in enumerate(tr_k)}; trd = tr.assign(k=tr.k.map(slot)); n_tr = len(tr_k)
o_t, pq_t, k_t = tens(trd); gen = torch.Generator(device=dev).manual_seed(a.seed)
for ep in range(a.epochs):
    # random enc/target split of each training annotator's triplets (per-triplet Bernoulli with annotator-specific rate)
    rate = (torch.rand(n_tr, device=dev, generator=gen) * 0.6 + 0.1)[k_t]; enc = torch.rand(len(k_t), device=dev, generator=gen) < rate
    if a.kind != "D0":
        z = m.enc(m.f, o_t[enc], pq_t[enc], k_t[enc], n_tr)
    else: z = torch.zeros(n_tr, a.M, device=dev)
    lg = m.dec.logits(m.f, z, o_t[~enc], pq_t[~enc], k_t[~enc])
    loss = Fn.cross_entropy(lg, torch.zeros(len(lg), dtype=torch.long, device=dev)) + a.z_l2 * (z ** 2).sum(1).mean() + 1e-4 * (m.f ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {loss.item():.4f} acc {(lg.argmax(1) == 0).float().mean():.4f}", flush=True)

def fit_direct(o, pq, kidx, n, steps=200, lr=0.05, l2=1e-2):
    z = torch.zeros(n, a.M, device=dev, requires_grad=True); op = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        lg = m.dec.logits(m.f, z, o, pq, kidx); ce = Fn.cross_entropy(lg, torch.zeros(len(lg), dtype=torch.long, device=dev), reduction="none")
        loss = torch.zeros(n, device=dev).index_add_(0, kidx, ce) / torch.bincount(kidx, minlength=n).clamp(min=1) + l2 * (z ** 2).sum(1)
        op.zero_grad(); loss.sum().backward(); op.step()
    return z.detach()
m.eval(); out = []
te_k = np.sort(te.k.unique()); slot = {k: i for i, k in enumerate(te_k)}; ted = te.assign(k=te.k.map(slot)); n_te = len(te_k)
tst = ted[~ted.est]; o_s, pq_s, k_s = tens(tst)
g = np.random.default_rng(1000 + a.seed)
with torch.no_grad():
    for n_est in (10, 20, 40, 60):
        sh = ted[ted.est].sample(frac=1, random_state=int(g.integers(1e9))); est = sh[sh.groupby("k").cumcount() < n_est]
        o_e, pq_e, k_e = tens(est)
        z = m.enc(m.f, o_e, pq_e, k_e, n_te) if a.kind != "D0" else torch.zeros(n_te, a.M, device=dev)
        acc = (m.dec.logits(m.f, z, o_s, pq_s, k_s).argmax(1) == 0).float()
        acc_k = torch.zeros(n_te, device=dev).index_add_(0, k_s, acc) / torch.bincount(k_s, minlength=n_te)
        row = pd.DataFrame(dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, ann=[anns[k] for k in te_k], n_est=n_est, acc_amort=acc_k.cpu().numpy()))
        if a.kind != "D0":
            with torch.enable_grad(): zd = fit_direct(o_e, pq_e, k_e, n_te)
            accd = (m.dec.logits(m.f, zd, o_s, pq_s, k_s).argmax(1) == 0).float()
            row["acc_direct"] = (torch.zeros(n_te, device=dev).index_add_(0, k_s, accd) / torch.bincount(k_s, minlength=n_te)).cpu().numpy()
            for j in range(a.M): row[f"z{j}"] = z[:, j].cpu().numpy()
        out.append(row)
df_out = pd.concat(out); tag = f"{a.kind}_D{a.D}_M{a.M}_f{a.fold}_s{a.seed}"
df_out.to_csv(f"{a.out}/eval_{tag}.csv", index=False); torch.save(m.state_dict(), f"{a.out}/model_{tag}.pt"); np.save(f"{a.out}/faces_{tag}.npy", m.f.detach().cpu().numpy())
print(df_out.groupby("n_est")[[c for c in df_out.columns if c.startswith("acc")]].mean().round(4).to_string())
