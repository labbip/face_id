"""Control for the structure comparison: simulated participants with known D1-type (diagonal reweighting) or D2-type
(non-diagonal linear transform A_k = I + eps*sum_m z_km B_m from a rank-4 latent) individual differences on the population embedding of a D0 run; Face-ID
D1 and D2 are trained on them and evaluated on fold-out simulated participants. Shows whether D2 > D1 when D2-type
structure is present, and D2 ~= D1 when it is absent. eps is calibrated so that the D2 sims deviate from the population
RDM as much as the D1 sims (same mean Spearman between individual true RDM and population RDM).
usage: uv run python -I scripts/48_sim_recovery_d2.py <D0_run_dir> --sim D2 --kind D1 [--noise 0.45 --wsd 0.6 --M 4]
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
from face_id.paths import PROC, RESULTS
from face_id.data import make_folds, split_trials, trials_to_rdm, Trial, Participant
from face_id.train import train_model, infer_z, fit_z_direct, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--noise", type=float, default=0.45); ap.add_argument("--wsd", type=float, default=0.6)
ap.add_argument("--M", type=int, default=4); ap.add_argument("--sim", default="D2"); ap.add_argument("--kind", default="D1"); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--d2_rank", type=int, default=4)
args = ap.parse_args()
a = json.load(open(os.path.join(args.run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{a['exp']}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if a["exp"] == 1 else ("source", "version_race"))
f = np.load(os.path.join(args.run, "faces.npy")); D = f.shape[1]; N = len(parts)
rng = np.random.default_rng(args.seed); iu = np.triu_indices(nf, k=1)
def dist(delta, w=None, A=None):
    if A is not None: return np.sqrt(((delta @ A.T) ** 2).sum(1))
    return np.sqrt(((delta ** 2) * w).sum(1))
pop_rdm = dist(f[iu[0]] - f[iu[1]], w=np.ones(D))
W = np.exp(args.wsd * rng.standard_normal((N, D)))
target = np.mean([nan_spearman(dist(f[iu[0]] - f[iu[1]], w=W[i]), pop_rdm) for i in range(300)])
# D2-type individual differences generated from a low-dimensional latent (rank --d2_rank), the generative counterpart of the D2 decoder:
# A_k = I + eps * sum_m z_km B_m, z_k ~ N(0, I), B_m shared random matrices
Btrue = rng.standard_normal((args.d2_rank, D, D)) / np.sqrt(D); Ztrue = rng.standard_normal((N, args.d2_rank))
G = np.einsum("km,mij->kij", Ztrue, Btrue)
def mean_rho(eps): return np.mean([nan_spearman(dist(f[iu[0]] - f[iu[1]], A=np.eye(D) + eps * G[i]), pop_rdm) for i in range(300)])
lo, hi = 0.0, 2.0
for _ in range(25):
    mid = (lo + hi) / 2
    if mean_rho(mid) > target: lo = mid
    else: hi = mid
eps = (lo + hi) / 2
print(f"sim {args.sim}: target rho(true, pop) = {target:.3f} (D1 sims, wsd {args.wsd}); D2 eps = {eps:.3f} -> {mean_rho(eps):.3f}", flush=True)
A = np.eye(D)[None] + eps * G
def true_rdm(i): return dist(f[iu[0]] - f[iu[1]], w=W[i]) if args.sim == "D1" else dist(f[iu[0]] - f[iu[1]], A=A[i])
def simulate(i, p, sigma):
    trials = []
    for t in p.ma:
        ii, jj = np.triu_indices(len(t.faces), k=1); delta = f[t.faces[ii]] - f[t.faces[jj]]
        d = dist(delta, w=W[i]) if args.sim == "D1" else dist(delta, A=A[i]); d = d / np.sqrt((d ** 2).mean())
        d = np.abs(d + sigma * rng.standard_normal(len(d)))
        tt = Trial.__new__(Trial); tt.faces = t.faces; tt.xy = t.xy; tt.task = "MA"; tt.d = d / np.sqrt((d ** 2).mean() + 1e-12); trials.append(tt)
    return Participant(p.pid, trials, [], p.meta)
sims = [simulate(i, p, args.noise) for i, p in enumerate(parts)]
train = [s for s, fo in zip(sims, folds) if fo != a["fold"]]; test = [s for s, fo in zip(sims, folds) if fo == a["fold"]]; tidx = np.where(folds == a["fold"])[0]
sh = [nan_spearman(trials_to_rdm(e, nf), trials_to_rdm(h, nf)) for e, h in (split_trials(p, 0.7, 0) for p in test[:200])]
print("simulated est-vs-hold rho: %.3f" % np.nanmean(sh), flush=True)
model, hist = train_model(train, nf, args.kind, D, args.M, None, args.epochs, 32, 5e-3, 0.0, args.seed, "cuda", False, None, None, det=True, f_init=f)
est, hold = zip(*[split_trials(p, 0.7, 0) for p in test])
mu, _ = infer_z(model, list(est), "cuda"); zd = fit_z_direct(model, list(est), "cuda", steps=300, lr=0.05, l2=1e-3, init=mu)
P0 = predict_full_rdm(model, torch.zeros(1, args.M).cuda(), nf, "cuda")[0]; Pm = predict_full_rdm(model, mu, nf, "cuda"); Pd = predict_full_rdm(model, zd, nf, "cuda")
Ptrue = np.array([true_rdm(i) for i in tidx]); H = [trials_to_rdm(h, nf) for h in hold]
out = dict(exp=a["exp"], fold=a["fold"], sim=args.sim, kind=args.kind, M=args.M, seed=args.seed, d2_rank=args.d2_rank, eps=float(eps), target_rho=float(target), n_test=len(test), sim_splithalf=float(np.nanmean(sh)))
for nm, P in [("pop", [P0] * len(test)), ("amortized", Pm), ("direct", Pd), ("oracle", Ptrue)]:
    out[f"hold_{nm}"] = float(np.nanmean([nan_spearman(P[i], H[i]) for i in range(len(test))])); out[f"true_{nm}"] = float(np.nanmean([nan_spearman(P[i], Ptrue[i]) for i in range(len(test))]))
    print("%-10s hold rho %.3f | rho with true full RDM %.3f" % (nm, out[f"hold_{nm}"], out[f"true_{nm}"]), flush=True)
os.makedirs(os.path.join(RESULTS, "sim_recovery"), exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, f"sim_recovery{'_n03' if args.noise != 0.45 else ''}/exp{a['exp']}_f{a['fold']}_sim{args.sim}_kind{args.kind}_M{args.M}_s{args.seed}.json"), "w"), indent=1)
