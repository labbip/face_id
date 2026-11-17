"""Face-ID on the AVFS main set (4,921 FFHQ faces, 1,283 annotators with >= 100 triplets) with CLIP image embeddings:
f_i = P e_i (CLIP ViT-B/32 -> PCA --pcs -> D). Decoders D0-D3 as in 28_avfs_faceid.py. Annotator 10-fold CV.
Evaluation on held-out annotators, z from n_est of their estimation-half triplets:
  main : accuracy on their test-half triplets (sparse, mostly new face pairs)
  novel: accuracy on their triplets of the 56 novel faces (never in training)       [z from all est triplets]
  same : accuracy on the 1,000 common triplets they judged + leave-one-out modal-vote agreement (ceiling) [z from all est]
Run: uv run python -I scripts/38_avfs_faceid_img.py --kind D1 --fold 0
"""
import argparse, json, os, collections, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.decomposition import PCA
p = argparse.ArgumentParser()
p.add_argument("--kind", default="D1"); p.add_argument("--D", type=int, default=8); p.add_argument("--M", type=int, default=4); p.add_argument("--pcs", type=int, default=32)
p.add_argument("--fold", type=int, default=0); p.add_argument("--folds", type=int, default=10); p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=400); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--z_l2", type=float, default=1e-3); p.add_argument("--p_l2", type=float, default=1e-4)
p.add_argument("--hidden", type=int, default=64); p.add_argument("--min_trip", type=int, default=100); p.add_argument("--out", default="results/avfs_img")
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
D_ = "data/raw/avfs/unzip/avfs-dataset-v1/"
ids = json.load(open("results/avfs_face_ids.json")); fidx = {f: i for i, f in enumerate(ids)}; F = len(ids)
def load(fn):
    tv = json.load(open(D_ + fn)); rows = [(t["annotator_id"], *q, pos) for t in tv.values() for q, pos in zip(t["triplet_questions"], t["odd_one_out_positions"])]
    d = pd.DataFrame(rows, columns=["ann", "a", "b", "c", "odd"])
    for c in "abc": d[c] = d[c].map(fidx)
    T = d[["a", "b", "c"]].to_numpy(); o = d.odd.to_numpy(); d["o"] = T[np.arange(len(T)), o]
    rest = np.array([[t[j] for j in range(3) if j != oo] for t, oo in zip(T, o)]); d["p"], d["q"] = rest[:, 0], rest[:, 1]; return d
main, novel, same = load("ooo_train_val.json"), load("ooo_test_novel_stimuli.json"), load("ooo_test_same_stimuli.json")
cnt = main.ann.value_counts(); main = main[main.ann.isin(cnt[cnt >= a.min_trip].index)].reset_index(drop=True)
anns = sorted(main.ann.unique()); A = len(anns); amap = {k: i for i, k in enumerate(anns)}; main["k"] = main.ann.map(amap)
rng0 = np.random.default_rng(0); fid = rng0.permutation(A) % a.folds; main["est"] = rng0.random(len(main)) < 0.5; main["fold"] = fid[main.k]
train_faces = np.array(sorted(set(main.a) | set(main.b) | set(main.c)))
E = np.load("results/avfs_embed_clip.npy"); pca = PCA(a.pcs, random_state=0).fit(E[train_faces]); X = pca.transform(E); X = (X - X[train_faces].mean(0)) / X[train_faces].std(0)
Xt = torch.tensor(X, device=dev, dtype=torch.float32)
tr, te = main[main.fold != a.fold], main[main.fold == a.fold]
print(f"faces {F} (train {len(train_faces)}), annotators {A} (train {tr.ann.nunique()}, test {te.ann.nunique()}), triplets {len(main)}", flush=True)

class Enc(nn.Module):
    def __init__(s, D, M, h):
        super().__init__(); s.net = nn.Sequential(nn.Linear(3 * D + 3, h), nn.GELU(), nn.Linear(h, h), nn.GELU()); s.att = nn.Linear(h, 1); s.head = nn.Linear(h, M)
    def forward(s, f, o, pq, kidx, n):
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
    def logits(s, f, z, o, pq, kidx):
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
    def __init__(s): super().__init__(); s.P = nn.Linear(a.pcs, a.D); s.enc = Enc(a.D, a.M, a.hidden); s.dec = Dec(a.D, a.M, F, a.kind, a.hidden)
    def f(s): return s.P(Xt)
m = Model().to(dev); opt = torch.optim.Adam(m.parameters(), lr=a.lr)
def tens(d, col="k"): return (torch.tensor(d.o.to_numpy(), device=dev), torch.tensor(d[["p", "q"]].to_numpy(), device=dev), torch.tensor(d[col].to_numpy(), device=dev))
tr_k = np.sort(tr.k.unique()); trd = tr.assign(k=tr.k.map({k: i for i, k in enumerate(tr_k)})); n_tr = len(tr_k)
o_t, pq_t, k_t = tens(trd); gen = torch.Generator(device=dev).manual_seed(a.seed); zero = lambda n: torch.zeros(n, dtype=torch.long, device=dev)
for ep in range(a.epochs):
    rate = (torch.rand(n_tr, device=dev, generator=gen) * 0.6 + 0.1)[k_t]; enc = torch.rand(len(k_t), device=dev, generator=gen) < rate
    f = m.f(); z = m.enc(f, o_t[enc], pq_t[enc], k_t[enc], n_tr) if a.kind != "D0" else torch.zeros(n_tr, a.M, device=dev)
    lg = m.dec.logits(f, z, o_t[~enc], pq_t[~enc], k_t[~enc])
    loss = Fn.cross_entropy(lg, zero(len(lg))) + a.z_l2 * (z ** 2).sum(1).mean() + a.p_l2 * (m.P.weight ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0 or ep == a.epochs - 1: print(f"ep {ep} loss {loss.item():.4f} acc {(lg.argmax(1) == 0).float().mean():.4f}", flush=True)

def fit_direct(f, o, pq, kidx, n, steps=200, lr=0.05, l2=1e-2):
    z = torch.zeros(n, a.M, device=dev, requires_grad=True); op = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        lg = m.dec.logits(f, z, o, pq, kidx); ce = Fn.cross_entropy(lg, zero(len(lg)), reduction="none")
        loss = torch.zeros(n, device=dev).index_add_(0, kidx, ce) / torch.bincount(kidx, minlength=n).clamp(min=1) + l2 * (z ** 2).sum(1)
        op.zero_grad(); loss.sum().backward(); op.step()
    return z.detach()
def acc_by(f, z, d, n):     # per-annotator accuracy on triplets d (with slot col k)
    o, pq, k = tens(d); acc = (m.dec.logits(f, z, o, pq, k).argmax(1) == 0).float()
    return (torch.zeros(n, device=dev).index_add_(0, k, acc) / torch.bincount(k, minlength=n).clamp(min=1)).cpu().numpy()
m.eval(); out = []
te_k = np.sort(te.k.unique()); slot = {k: i for i, k in enumerate(te_k)}; ted = te.assign(k=te.k.map(slot)); n_te = len(te_k); te_names = [anns[k] for k in te_k]
tst = ted[~ted.est]; g = np.random.default_rng(1000 + a.seed)
nov = novel[novel.ann.isin(te_names)].assign(k=lambda d: d.ann.map({n: slot[amap[n]] for n in te_names}))
sm = same[same.ann.isin(te_names)].assign(k=lambda d: d.ann.map({n: slot[amap[n]] for n in te_names}))
# leave-one-out modal-vote agreement on the common triplets (ceiling for individual prediction)
votes = collections.defaultdict(list)
for o_, p_, q_, an in zip(same.o, same.p, same.q, same.ann): votes[(o_, p_, q_) if False else tuple(sorted((o_, p_, q_)))].append((an, o_))
def loo_agree(d):
    res = np.full(n_te, np.nan); hits = collections.defaultdict(list)
    for o_, p_, q_, an, k in zip(d.o, d.p, d.q, d.ann, d.k):
        others = [oo for aa, oo in votes[tuple(sorted((o_, p_, q_)))] if aa != an]
        if others: modal = collections.Counter(others).most_common(1)[0][0]; hits[k].append(float(modal == o_))
    for k, v in hits.items(): res[k] = np.mean(v)
    return res
with torch.no_grad():
    f = m.f()
    for n_est in (20, 50, 100, 10 ** 6):
        sh = ted[ted.est].sample(frac=1, random_state=int(g.integers(1e9))); est = sh[sh.groupby("k").cumcount() < n_est]
        o_e, pq_e, k_e = tens(est); n_used = est.groupby("k").size().reindex(range(n_te)).fillna(0).to_numpy()
        z = m.enc(f, o_e, pq_e, k_e, n_te) if a.kind != "D0" else torch.zeros(n_te, a.M, device=dev)
        row = pd.DataFrame(dict(kind=a.kind, D=a.D, M=a.M, fold=a.fold, seed=a.seed, ann=te_names, n_est=n_est, n_est_used=n_used, acc_main=acc_by(f, z, tst, n_te)))
        if a.kind != "D0":
            with torch.enable_grad(): zd = fit_direct(f, o_e, pq_e, k_e, n_te)
            row["acc_main_direct"] = acc_by(f, zd, tst, n_te)
            for j in range(a.M): row[f"z{j}"] = z[:, j].cpu().numpy()
        if n_est == 10 ** 6:
            an = acc_by(f, z, nov, n_te); an[~np.isin(np.arange(n_te), nov.k.unique())] = np.nan; row["acc_novel"] = an
            asm = acc_by(f, z, sm, n_te); asm[~np.isin(np.arange(n_te), sm.k.unique())] = np.nan; row["acc_same"] = asm; row["agree_modal_loo"] = loo_agree(sm)
            if a.kind != "D0":
                an = acc_by(f, zd, nov, n_te); an[~np.isin(np.arange(n_te), nov.k.unique())] = np.nan; row["acc_novel_direct"] = an
        out.append(row)
df_out = pd.concat(out); tag = f"{a.kind}_D{a.D}_M{a.M}_f{a.fold}_s{a.seed}"
df_out.to_csv(f"{a.out}/eval_{tag}.csv", index=False); torch.save(m.state_dict(), f"{a.out}/model_{tag}.pt")
print(df_out.groupby("n_est")[[c for c in df_out.columns if c.startswith("acc") or c.startswith("agree")]].mean().round(4).to_string())
