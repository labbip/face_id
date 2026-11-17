"""Simulated-participant control at the MEASURED reliability of the real data (2026-11-17).
Same generative models, trial structure, fold and training settings as scripts/48_sim_recovery_d2.py (D1-type: log-normal dimension weights;
D2-type: A_k = I + eps * sum_m z_km B_m, rank 4, eps calibrated to the D1-type rho(true RDM, population RDM)), but
  (1) the reliability of the REAL participants is first computed with the definition 48 uses for `sim_splithalf`
      (Spearman between the inverse-MDS RDMs of the 70 % estimation trials and the 30 % held-out trials, split_trials(p, 0.7, 0)),
  (2) the noise SD is calibrated by bisection (common random numbers) so that the simulated value equals the real one (all participants),
  (3) with --match_pop the size of the individual differences (wsd; eps follows) is calibrated as well, so that the held-out rho of the
      population RDM also equals the real value (outer bisection on wsd, inner on the noise),
  (4) D1 and D2 models are trained in one process on the SAME simulated participants, and per-participant values are written.
With --noise X the calibration is skipped (--noise 0.3 reproduces the simulated data of 48 for the same seed).
Outputs: results/sim_recovery_<tag>/exp<e>_s<seed>_sim<D1|D2>.json (means, calibration) and ..._raters.csv (one row per fold-0 simulated participant).
The 70/30 split is the one of face_id.data.split_trials, but seeded with crc32 of the participant id: the library function uses Python's hash(), which
differs between processes (`python -I` ignores PYTHONHASHSEED), so the split of 48 is not reproducible; the definition (random 70 %) is the same.
Run: uv run python -I scripts/88_sim_recovery_matched.py --exp 2 --seed 0 --tag matched
     uv run python -I scripts/88_sim_recovery_matched.py --exp 2 --seed 0 --tag matched_id --match_pop
"""
import os, json, pickle, argparse, time, zlib
import numpy as np, pandas as pd, torch
from face_id.paths import PROC, RESULTS
from face_id.data import make_folds, trials_to_rdm, Trial, Participant
from face_id.train import train_model, infer_z, fit_z_direct, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="matched")
ap.add_argument("--noise", type=float, default=None); ap.add_argument("--wsd", type=float, default=0.6); ap.add_argument("--match_pop", action="store_true")
ap.add_argument("--M", type=int, default=8); ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--d2_rank", type=int, default=4); ap.add_argument("--sims", default="D1,D2"); ap.add_argument("--no_train", action="store_true")
args = ap.parse_args()
run = os.path.join(RESULTS, f"runs/exp{args.exp}_D0_D8_M4_f0_s0_cv"); a = json.load(open(os.path.join(run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if args.exp == 1 else ("source", "version_race"))
f = np.load(os.path.join(run, "faces.npy")); D = f.shape[1]; N = len(parts); iu = np.triu_indices(nf, k=1); dF = f[iu[0]] - f[iu[1]]
tidx = np.where(folds == a["fold"])[0]; out_dir = os.path.join(RESULTS, f"sim_recovery_{args.tag}"); os.makedirs(out_dir, exist_ok=True)
def split_trials(p, frac=0.7, seed=0):      # as face_id.data.split_trials (random 70 % / 30 % of the arrangement trials), but seeded with a stable hash of the participant id
    idx = np.random.default_rng(zlib.crc32(f"{p.pid}|{seed}".encode())).permutation(len(p.ma)); k = max(1, min(int(round(frac * len(p.ma))), len(p.ma) - 1))
    return [p.ma[i] for i in idx[:k]], [p.ma[i] for i in idx[k:]]
def dist(delta, w=None, A=None): return np.sqrt(((delta @ A.T) ** 2).sum(1)) if A is not None else np.sqrt(((delta ** 2) * w).sum(1))
pop_rdm = dist(dF, w=np.ones(D))
def rel_of(ps):      # per participant: est-vs-hold rho (the 48 definition) and held-out rho of the population RDM
    sh, hp = [], []
    for p in ps:
        e, h = split_trials(p, 0.7, 0); H = trials_to_rdm(h, nf); sh.append(nan_spearman(trials_to_rdm(e, nf), H)); hp.append(nan_spearman(pop_rdm, H))
    return np.array(sh), np.array(hp)
t0 = time.time(); rsh, rhp = rel_of(parts)
real = dict(splithalf_all=float(np.nanmean(rsh)), splithalf_fold0=float(np.nanmean(rsh[tidx])), splithalf_fold0_first200=float(np.nanmean(rsh[tidx][:200])),
            hold_pop_all=float(np.nanmean(rhp)), hold_pop_fold0=float(np.nanmean(rhp[tidx])), n_all=N, n_fold0=len(tidx))
print("real exp%d:" % args.exp, {k: round(v, 4) for k, v in real.items()}, "(%.0f s)" % (time.time() - t0), flush=True)
# random draws in the order of 48: weights, B, Z, then the trial noise (participant by participant, trial by trial)
rng = np.random.default_rng(args.seed); Wn = rng.standard_normal((N, D)); Btrue = rng.standard_normal((args.d2_rank, D, D)) / np.sqrt(D); Ztrue = rng.standard_normal((N, args.d2_rank))
G = np.einsum("km,mij->kij", Ztrue, Btrue); pairs = [[np.triu_indices(len(t.faces), k=1) for t in p.ma] for p in parts]
E = [[rng.standard_normal(len(ii)) for ii, jj in pp] for pp in pairs]          # common random numbers for every noise level and both sim types
def structure(wsd):
    W = np.exp(wsd * Wn); target = np.mean([nan_spearman(dist(dF, w=W[i]), pop_rdm) for i in range(300)])
    rho = lambda eps: np.mean([nan_spearman(dist(dF, A=np.eye(D) + eps * G[i]), pop_rdm) for i in range(300)])
    lo, hi = 0.0, 8.0
    for _ in range(25):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if rho(mid) > target else (lo, mid)
    eps = (lo + hi) / 2; return W, np.eye(D)[None] + eps * G, float(eps), float(target), float(rho(eps))
def clean(sim, W, A):       # noiseless within-trial distances, RMS-normalised as in 48
    out = []
    for i, p in enumerate(parts):
        row = []
        for t, (ii, jj) in zip(p.ma, pairs[i]):
            delta = f[t.faces[ii]] - f[t.faces[jj]]; d = dist(delta, w=W[i]) if sim == "D1" else dist(delta, A=A[i]); row.append(d / np.sqrt((d ** 2).mean()))
        out.append(row)
    return out
def simulate(C, sigma, idx=None):
    sims = []
    for i in (range(N) if idx is None else idx):
        p = parts[i]; trials = []
        for t, d0, e in zip(p.ma, C[i], E[i]):
            d = np.abs(d0 + sigma * e); tt = Trial.__new__(Trial); tt.faces = t.faces; tt.xy = t.xy; tt.task = "MA"; tt.d = d / np.sqrt((d ** 2).mean() + 1e-12); trials.append(tt)
        sims.append(Participant(p.pid, trials, [], p.meta))
    return sims
def calib_noise(C, target, log=None):
    lo, hi = 0.05, 3.0
    for _ in range(14):
        mid = (lo + hi) / 2; sh, hp = rel_of(simulate(C, mid)); v = np.nanmean(sh); lo, hi = (mid, hi) if v > target else (lo, mid)
        if log is not None: log.append((mid, float(v), float(np.nanmean(hp))))
    return (lo + hi) / 2
for sim in args.sims.split(","):
    t0 = time.time(); wsd = args.wsd; trace = []
    if args.match_pop:      # outer bisection on the size of the individual differences: larger wsd -> lower held-out rho of the population RDM at fixed reliability
        lo, hi = 0.3, 4.0
        for _ in range(9):
            wsd = (lo + hi) / 2; W, A, eps, target, rho_eps = structure(wsd); C = clean(sim, W, A); s = calib_noise(C, real["splithalf_all"]); sh, hp = rel_of(simulate(C, s))
            trace.append((wsd, s, float(np.nanmean(sh)), float(np.nanmean(hp)), target)); print("  wsd %.3f noise %.3f splithalf %.4f hold_pop %.4f rho(true,pop) %.3f" % trace[-1], flush=True)
            lo, hi = (wsd, hi) if np.nanmean(hp) > real["hold_pop_all"] else (lo, wsd)
        wsd = (lo + hi) / 2
    W, A, eps, target, rho_eps = structure(wsd); C = clean(sim, W, A)
    sigma = args.noise if args.noise is not None else calib_noise(C, real["splithalf_all"])
    sims = simulate(C, sigma); ssh, shp = rel_of(sims)
    cal = dict(exp=args.exp, seed=args.seed, sim=sim, tag=args.tag, wsd=float(wsd), noise=float(sigma), eps=eps, target_rho=target, rho_d2=rho_eps,
               sim_splithalf_all=float(np.nanmean(ssh)), sim_splithalf_fold0=float(np.nanmean(ssh[tidx])), sim_splithalf_fold0_first200=float(np.nanmean(ssh[tidx][:200])),
               sim_hold_pop_all=float(np.nanmean(shp)), sim_hold_pop_fold0=float(np.nanmean(shp[tidx])), **{"real_" + k: v for k, v in real.items()}, wsd_trace=trace)
    print("sim %s: wsd %.3f eps %.3f rho(true,pop) %.3f | noise %.4f -> splithalf all %.4f fold0 %.4f | hold_pop(true population RDM) all %.4f fold0 %.4f (%.0f s)" % (
        sim, wsd, eps, target, sigma, cal["sim_splithalf_all"], cal["sim_splithalf_fold0"], cal["sim_hold_pop_all"], cal["sim_hold_pop_fold0"], time.time() - t0), flush=True)
    if args.no_train: json.dump(cal, open(os.path.join(out_dir, f"exp{args.exp}_s{args.seed}_sim{sim}.json"), "w"), indent=1); continue
    train = [s for s, fo in zip(sims, folds) if fo != a["fold"]]; test = [sims[i] for i in tidx]
    est, hold = zip(*[split_trials(p, 0.7, 0) for p in test]); H = [trials_to_rdm(h, nf) for h in hold]
    Ptrue = np.array([dist(dF, w=W[i]) if sim == "D1" else dist(dF, A=A[i]) for i in tidx]); sp = lambda P, Q: np.array([nan_spearman(P[i], Q[i]) for i in range(len(test))])
    R = pd.DataFrame(dict(exp=args.exp, seed=args.seed, sim=sim, slot=tidx, n_trials=[len(p.ma) for p in test], splithalf=ssh[tidx], hold_oracle=sp(Ptrue, H)))
    for kind in ("D1", "D2"):
        model, hist = train_model(train, nf, kind, D, args.M, None, args.epochs, 32, 5e-3, 0.0, args.seed, "cuda", False, None, None, det=True, f_init=f)
        mu, _ = infer_z(model, list(est), "cuda"); zd = fit_z_direct(model, list(est), "cuda", steps=300, lr=0.05, l2=1e-3, init=mu)
        P0 = predict_full_rdm(model, torch.zeros(1, args.M).cuda(), nf, "cuda")[0]; Pm = predict_full_rdm(model, mu, nf, "cuda"); Pd = predict_full_rdm(model, zd, nf, "cuda")
        for nm, P in (("pop", [P0] * len(test)), ("amortized", Pm), ("direct", Pd)): R[f"hold_{nm}_{kind}"] = sp(P, H); R[f"true_{nm}_{kind}"] = sp(P, Ptrue)
        print("  model %s: hold pop %.4f amortized %.4f direct %.4f oracle %.4f | true pop %.3f amortized %.3f direct %.3f (%.0f s)" % (
            kind, *[R[f"hold_{n}_{kind}"].mean() for n in ("pop", "amortized", "direct")], R.hold_oracle.mean(), *[R[f"true_{n}_{kind}"].mean() for n in ("pop", "amortized", "direct")], time.time() - t0), flush=True)
    R.to_csv(os.path.join(out_dir, f"exp{args.exp}_s{args.seed}_sim{sim}_raters.csv"), index=False)
    json.dump(dict(cal, n_test=len(test), M=args.M, epochs=args.epochs, **{c: float(R[c].mean()) for c in R.columns if c.startswith(("hold_", "true_"))}), open(os.path.join(out_dir, f"exp{args.exp}_s{args.seed}_sim{sim}.json"), "w"), indent=1)
