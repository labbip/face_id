"""Control (plan §6.8): simulated participants with known decoder parameters.
Using a trained model's face embedding (train fold), generate artificial participants whose
dissimilarities follow D1 (known weights w) or D2 (known A), with on-screen noise matched to the
real split-half reliability, using the real participants' trial structure (faces per trial).
Then (a) train the model on real-participant trials replaced by simulated ones? -- no: here we test
inference only: given the trained encoder/decoder, does z (amortised and direct) recover the
generating w (correlation between true and decoded weights), and does rho_hold exceed the
population baseline when individual differences are known to be low-dimensional?
usage: uv run python scripts/09_simulated_participants.py <run_dir> [--noise 1.0]
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC, RESULTS
from face_id.models import FaceID
from face_id.data import make_folds, split_trials, trials_to_rdm, Trial, Participant
from face_id.train import infer_z, fit_z_direct, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--noise", type=float, default=1.0); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--wsd", type=float, default=0.6, help="sd of log-weights across simulated participants")
args = ap.parse_args()
a = json.load(open(os.path.join(args.run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{a['exp']}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if a["exp"] == 1 else ("source", "version_race"))
test = [p for p, f in zip(parts, folds) if f == a["fold"]]
m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(args.run, "model.pt"))); m.eval()
f = m.emb().detach().cpu().numpy()
rng = np.random.default_rng(0)
# noise level: choose sigma so that simulated split-half reliability ~ real (calibrated by trial)
def simulate(p, w, sigma):
    """replace p's MA trials' distances by ||w*(f_i-f_j)|| + noise, embedded in 2D via MDS-like noise:
    we add Gaussian noise to the dissimilarity and renormalise (no 2-D projection)."""
    trials = []
    for t in p.ma:
        ii, jj = np.triu_indices(len(t.faces), k=1)
        d = np.sqrt((((f[t.faces[ii]] - f[t.faces[jj]]) ** 2) * w).sum(1))
        d = d / np.sqrt((d ** 2).mean()); d = np.abs(d + sigma * rng.standard_normal(len(d)))
        tt = Trial.__new__(Trial); tt.faces = t.faces; tt.xy = t.xy; tt.task = "MA"; tt.d = d / np.sqrt((d ** 2).mean() + 1e-12)
        trials.append(tt)
    return Participant(p.pid, trials, [], p.meta)
sel = [test[i] for i in rng.choice(len(test), min(args.n, len(test)), replace=False)]
W = np.exp(args.wsd * rng.standard_normal((len(sel), f.shape[1])))
sims = [simulate(p, W[i], args.noise) for i, p in enumerate(sel)]
# reliability check
sh = []
for p in sims:
    est, hold = split_trials(p, 0.7, 0); sh.append(nan_spearman(trials_to_rdm(est, nf), trials_to_rdm(hold, nf)))
print("simulated est-vs-hold rho (cf. real ~0.23): %.3f" % np.nanmean(sh))
est, hold = zip(*[split_trials(p, 0.7, 0) for p in sims])
mu, _ = infer_z(m, list(est), "cuda"); zd = fit_z_direct(m, list(est), "cuda", steps=300, lr=0.05, l2=1e-3, init=mu)
z0 = torch.zeros(1, a["M"]).cuda()
P0 = predict_full_rdm(m, z0, nf, "cuda")[0]; Pm = predict_full_rdm(m, mu, nf, "cuda"); Pd = predict_full_rdm(m, zd, nf, "cuda")
# true RDM
iu = np.triu_indices(nf, k=1)
Ptrue = np.array([np.sqrt((((f[iu[0]] - f[iu[1]]) ** 2) * W[i]).sum(1)) for i in range(len(sims))])
H = [trials_to_rdm(h, nf) for h in hold]
for nm, P in [("pop(z=0)", [P0] * len(sims)), ("amortized", Pm), ("direct", Pd), ("oracle(true w)", Ptrue)]:
    print("%-14s hold rho %.3f | rho with true full RDM %.3f" % (nm, np.nanmean([nan_spearman(P[i], H[i]) for i in range(len(sims))]), np.nanmean([nan_spearman(P[i], Ptrue[i]) for i in range(len(sims))])))
if a["kind"] == "D1":
    wa = m.dec.weights(mu).detach().cpu().numpy(); wd = m.dec.weights(zd).detach().cpu().numpy()
    print("weight recovery: corr(true logw, decoded logw) per dim | amortized:", np.round([np.corrcoef(np.log(W[:, k]), np.log(wa[:, k]))[0, 1] for k in range(f.shape[1])], 2))
    print("                                                    | direct   :", np.round([np.corrcoef(np.log(W[:, k]), np.log(wd[:, k]))[0, 1] for k in range(f.shape[1])], 2))
