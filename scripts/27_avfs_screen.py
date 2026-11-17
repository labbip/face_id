"""AVFS screening: is there systematic idiosyncratic variance in face-similarity odd-one-out judgments?
Population embedding (dot-product SPoSE-like, D dims) trained on each annotator's estimation half;
per-annotator D1 (nonneg dimension weights) fitted on the same half; both evaluated on the held-out half.
Run: uv run python -I scripts/27_avfs_screen.py [--D 16]
"""
import json, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as Fn, collections
p = argparse.ArgumentParser(); p.add_argument("--D", type=int, default=16); p.add_argument("--epochs", type=int, default=40)
p.add_argument("--l2", type=float, default=1e-4); p.add_argument("--w_l2", type=float, default=1e-2); p.add_argument("--min_trip", type=int, default=100); p.add_argument("--set", default="train_val"); p.add_argument("--tag", default=""); p.add_argument("--path", default=None)
a = p.parse_args(); dev = "cuda"; torch.manual_seed(0); rng = np.random.default_rng(0)
D_ = "data/raw/avfs/unzip/avfs-dataset-v1/"
tv = json.load(open(a.path if a.path else D_ + {"train_val": "ooo_train_val.json", "novel": "ooo_test_novel_stimuli.json", "same": "ooo_test_same_stimuli.json"}[a.set]))
rows = []
for t in tv.values():
    for q, pos in zip(t["triplet_questions"], t["odd_one_out_positions"]): rows.append((t["annotator_id"], *q, pos))
df = pd.DataFrame(rows, columns=["ann", "a", "b", "c", "odd"])
faces = {f: i for i, f in enumerate(sorted(set(df.a) | set(df.b) | set(df.c)))}; F = len(faces)
for c in "abc": df[c] = df[c].map(faces)
cnt = df.ann.value_counts(); df = df[df.ann.isin(cnt[cnt >= a.min_trip].index)].reset_index(drop=True)
anns = {k: i for i, k in enumerate(sorted(df.ann.unique()))}; df["k"] = df.ann.map(anns); A = len(anns)
df["est"] = rng.random(len(df)) < 0.5
print(f"faces {F}, annotators (>= {a.min_trip} triplets) {A}, triplets {len(df)}, est {df.est.sum()}")
T = torch.tensor(df[["a", "b", "c"]].to_numpy(), device=dev); Y = torch.tensor(df.odd.to_numpy(), device=dev)
K = torch.tensor(df.k.to_numpy(), device=dev); E = torch.tensor(df.est.to_numpy(), device=dev)

def logits(f, w=None):            # odd=a <-> pair (b,c) most similar
    fa, fb, fc = f[T[:, 0]], f[T[:, 1]], f[T[:, 2]]
    if w is None: s = lambda x, y: (x * y).sum(-1)
    else: s = lambda x, y: (w[K] * x * y).sum(-1)
    return torch.stack([s(fb, fc), s(fa, fc), s(fa, fb)], 1)
# 1. population embedding on est triplets
f = torch.nn.Parameter(torch.randn(F, a.D, device=dev) * 0.1); opt = torch.optim.Adam([f], lr=1e-2)
idx = torch.where(E)[0]
for ep in range(a.epochs):
    perm = idx[torch.randperm(len(idx), device=dev)]
    for i in range(0, len(perm), 8192):
        b = perm[i:i + 8192]; lg = logits(f)[b]
        loss = Fn.cross_entropy(lg, Y[b]) + a.l2 * (f ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            acc = (logits(f).argmax(1) == Y).float(); print(f"ep {ep} acc est {acc[E].mean():.4f} test {acc[~E].mean():.4f}", flush=True)
f = f.detach()
# 2. per-annotator D1 weights (softplus, init 1) on est triplets, ridge toward 1
th = torch.nn.Parameter(torch.full((A, a.D), 0.5413, device=dev)); opt = torch.optim.Adam([th], lr=0.05)
cntk = torch.bincount(K[E], minlength=A).float()
for it in range(300):
    w = Fn.softplus(th); lg = logits(f, w)[E]
    ce = Fn.cross_entropy(lg, Y[E], reduction="none")
    loss = ce.mean() + a.w_l2 * ((th - 0.5413) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    w = Fn.softplus(th)
    acc_pop = (logits(f).argmax(1) == Y).float(); acc_ind = (logits(f, w).argmax(1) == Y).float()
    # per-annotator means on test half
    out = pd.DataFrame(dict(k=K.cpu().numpy(), est=E.cpu().numpy(), popacc=acc_pop.cpu().numpy(), ind=acc_ind.cpu().numpy()))
    g = out[~out.est].groupby("k")[["popacc", "ind"]].mean(); g["n_test"] = out[~out.est].groupby("k").size()
    d = g.ind - g.popacc
    from scipy.stats import ttest_rel, wilcoxon
    print(f"TEST half: population acc {g.popacc.mean():.4f}, D1 individual acc {g.ind.mean():.4f}, delta {d.mean():+.4f} (se {d.std()/np.sqrt(len(d)):.4f}, t p={ttest_rel(g.ind, g.popacc).pvalue:.1e}, wilcoxon p={wilcoxon(d).pvalue:.1e}), n={len(d)}")
    ge = out[out.est].groupby("k")[["popacc", "ind"]].mean(); print(f"EST half (fit): population {ge.popacc.mean():.4f}, individual {ge.ind.mean():.4f}")
    # by annotator volume
    g["bin"] = pd.cut(g.n_test, [0, 100, 200, 500, 5000]); print(g.groupby("bin", observed=True).apply(lambda x: pd.Series(dict(n=len(x), pop=x.popacc.mean(), ind=x.ind.mean(), delta=(x.ind - x.popacc).mean()))).round(4))
    # weight dispersion across annotators
    lw = torch.log(w).cpu().numpy(); print("log-weight SD per dim:", np.round(lw.std(0), 3), " mean |log w|:", np.abs(lw).mean().round(3))
    g.to_csv(f"results/avfs_screen{a.tag}_annotators.csv"); np.save(f"results/avfs_screen{a.tag}_faces.npy", f.cpu().numpy()); np.save(f"results/avfs_screen{a.tag}_weights.npy", w.cpu().numpy())
    json.dump({k: int(v) for k, v in anns.items()}, open(f"results/avfs_screen{a.tag}_annotators.json", "w"))
