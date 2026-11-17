"""Control (plan §6.8, proper version): train Face-ID on simulated participants with known low-dimensional
individual differences (D1 reweighting of the population embedding), then test whether the amortised
encoder (and direct inference) recovers the generating weights on new simulated participants.
Trial structure (faces per trial) is taken from the real participants; noise matched so that the
simulated split reliability is close to the real one.
usage: uv run python scripts/09b_sim_train_recovery.py <D0_run_dir> --noise 0.45 --wsd 0.6 --M 8
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC, RESULTS
from face_id.data import make_folds, split_trials, trials_to_rdm, Trial, Participant
from face_id.train import train_model, infer_z, fit_z_direct, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--noise", type=float, default=0.45); ap.add_argument("--wsd", type=float, default=0.6)
ap.add_argument("--M", type=int, default=8); ap.add_argument("--kind", default="D1"); ap.add_argument("--epochs", type=int, default=100)
args = ap.parse_args()
a = json.load(open(os.path.join(args.run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{a['exp']}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if a["exp"] == 1 else ("source", "version_race"))
f = np.load(os.path.join(args.run, "faces.npy")); D = f.shape[1]
rng = np.random.default_rng(0)
def simulate(p, w, sigma):
    trials = []
    for t in p.ma:
        ii, jj = np.triu_indices(len(t.faces), k=1)
        d = np.sqrt((((f[t.faces[ii]] - f[t.faces[jj]]) ** 2) * w).sum(1)); d = d / np.sqrt((d ** 2).mean())
        d = np.abs(d + sigma * rng.standard_normal(len(d)))
        tt = Trial.__new__(Trial); tt.faces = t.faces; tt.xy = t.xy; tt.task = "MA"; tt.d = d / np.sqrt((d ** 2).mean() + 1e-12); trials.append(tt)
    return Participant(p.pid, trials, [], p.meta)
W = np.exp(args.wsd * rng.standard_normal((len(parts), D)))
sims = [simulate(p, W[i], args.noise) for i, p in enumerate(parts)]
train = [s for s, fo in zip(sims, folds) if fo != a["fold"]]; test = [s for s, fo in zip(sims, folds) if fo == a["fold"]]
Wtest = W[folds == a["fold"]]
sh = [nan_spearman(trials_to_rdm(e, nf), trials_to_rdm(h, nf)) for e, h in (split_trials(p, 0.7, 0) for p in test[:200])]
print("simulated est-vs-hold rho (real exp1 ~0.23): %.3f" % np.nanmean(sh), flush=True)
model, hist = train_model(train, nf, args.kind, D, args.M, None, args.epochs, 32, 5e-3, 0.0, 0, "cuda", False, None, None, det=True, f_init=f)
print("final train loss %.4f" % hist[-1]["loss"])
est, hold = zip(*[split_trials(p, 0.7, 0) for p in test])
mu, _ = infer_z(model, list(est), "cuda"); zd = fit_z_direct(model, list(est), "cuda", steps=300, lr=0.05, l2=1e-3, init=mu)
fe = model.emb().detach().cpu().numpy(); iu = np.triu_indices(nf, k=1)
P0 = predict_full_rdm(model, torch.zeros(1, args.M).cuda(), nf, "cuda")[0]; Pm = predict_full_rdm(model, mu, nf, "cuda"); Pd = predict_full_rdm(model, zd, nf, "cuda")
Ptrue = np.array([np.sqrt((((f[iu[0]] - f[iu[1]]) ** 2) * Wtest[i]).sum(1)) for i in range(len(test))])
H = [trials_to_rdm(h, nf) for h in hold]
for nm, P in [("pop(z=0)", [P0] * len(test)), ("amortized", Pm), ("direct", Pd), ("oracle", Ptrue)]:
    print("%-10s hold rho %.3f | rho with true full RDM %.3f" % (nm, np.nanmean([nan_spearman(P[i], H[i]) for i in range(len(test))]), np.nanmean([nan_spearman(P[i], Ptrue[i]) for i in range(len(test))])))
print("spearman(learned emb RDM, generating emb RDM) = %.3f" % nan_spearman(np.sqrt(((fe[iu[0]] - fe[iu[1]]) ** 2).sum(1)), np.sqrt(((f[iu[0]] - f[iu[1]]) ** 2).sum(1))))
if args.kind == "D1":
    # weights are only identifiable up to the learned embedding's axes; use CCA between true log w and decoded log w
    from sklearn.cross_decomposition import CCA
    for nm, z in [("amortized", mu), ("direct", zd)]:
        wd = np.log(model.dec.weights(z).detach().cpu().numpy()); cca = CCA(n_components=min(4, D)).fit(np.log(Wtest), wd); U, V = cca.transform(np.log(Wtest), wd)
        print(f"{nm}: CCA canonical corr (true log w vs decoded log w):", np.round([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])], 2),
              "| per-dim corr:", np.round([np.corrcoef(np.log(Wtest[:, k]), wd[:, k])[0, 1] for k in range(D)], 2))
