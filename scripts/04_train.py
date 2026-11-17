"""Train Face-ID (D0..D3) with participant k-fold CV and evaluate on fold-out participants.

usage: uv run python scripts/04_train.py --exp 1 --kind D1 --fold 0 --seed 0 [--D 8 --M 4 --epochs 60]
Outputs: results/runs/exp{exp}_{kind}_D{D}_M{M}_f{fold}_s{seed}/{model.pt,hist.jsonl,eval.csv,faces.npy}
"""
import argparse, os, sys, json, time, pickle
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC, RESULTS
from face_id.data import load_experiment, make_folds
from face_id.train import train_model, evaluate, predict_full_rdm

ap = argparse.ArgumentParser()
ap.add_argument("--exp", type=int, default=1)
ap.add_argument("--kind", default="D1")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n_folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--D", type=int, default=8)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--Dp", type=int, default=None)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--beta", type=float, default=1e-3)
ap.add_argument("--nonneg", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--init_mds", action="store_true", help="initialise face embedding from classical MDS of the training-fold aggregate RDM")
ap.add_argument("--direct", action="store_true", help="also evaluate with per-participant direct z optimisation")
ap.add_argument("--det", action="store_true", help="deterministic encoder (no sampling, no KL; L2 on z instead)")
ap.add_argument("--z_l2", type=float, default=1e-3)
args = ap.parse_args()

cache = os.path.join(PROC, f"exp{args.exp}_participants.pkl")
if os.path.exists(cache):
    parts, n_faces = pickle.load(open(cache, "rb"))
else:
    parts, n_faces = load_experiment(args.exp)
    pickle.dump((parts, n_faces), open(cache, "wb"))
strat = ("Gender", "Race") if args.exp == 1 else ("source", "version_race")
folds = make_folds(parts, args.n_folds, seed=0, strat_keys=strat)
train = [p for p, f in zip(parts, folds) if f != args.fold]
test = [p for p, f in zip(parts, folds) if f == args.fold]
name = f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f{args.fold}_s{args.seed}{args.tag}"
# training-fold aggregate RDM (no fold-out data) from saved individual RDMs
ind = np.load(os.path.join(RESULTS, f"repro_exp{args.exp}_individual_RDMs.npy"))
ind_pids = pd.read_csv(os.path.join(RESULTS, f"repro_exp{args.exp}_individual_RDMs_pids.csv")).pid.tolist()
pid2row = {p: i for i, p in enumerate(ind_pids)}
from face_id.invmds import inverse_mds
from face_id.rdm import vec_to_mat
agg_train = inverse_mds(ind[[pid2row[p.pid] for p in train if p.pid in pid2row]])[0]
f_init = None
if args.init_mds:
    Mx = vec_to_mat(np.nan_to_num(agg_train, nan=np.nanmean(agg_train)), n_faces); J = np.eye(n_faces) - 1.0 / n_faces
    B = -0.5 * J @ (Mx ** 2) @ J; w, V = np.linalg.eigh(B); idx = np.argsort(w)[::-1][: args.D]
    f_init = V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
out = os.path.join(RESULTS, "runs", name); os.makedirs(out, exist_ok=True)
json.dump(vars(args), open(os.path.join(out, "args.json"), "w"))
log = open(os.path.join(out, "hist.jsonl"), "w")
t0 = time.time()
model, hist = train_model(train, n_faces, args.kind, args.D, args.M, args.Dp, args.epochs, args.batch,
                          args.lr, args.beta, args.seed, args.device, args.nonneg, log, val_parts=test[:100], f_init=f_init, det=args.det, z_l2=args.z_l2)
print(f"trained {name} in {time.time()-t0:.0f}s; final loss {hist[-1]['loss']:.4f}", flush=True)
torch.save(model.state_dict(), os.path.join(out, "model.pt"))
np.save(os.path.join(out, "faces.npy"), model.emb().detach().cpu().numpy())
# population RDM = D0 prediction of this model's embedding (z = 0) as reference
with torch.no_grad():
    z0 = torch.zeros(1, args.M, device=args.device)
    pop = predict_full_rdm(model, z0, n_faces, args.device)[0]
np.save(os.path.join(out, "pop_rdm.npy"), pop)
# aggregate RDM from training participants' trials (data-based population baseline)
agg = agg_train
np.save(os.path.join(out, "agg_train.npy"), agg_train)
rows = []
for frac, n_est in [(0.7, None)]:
    ev = evaluate(model, test, n_faces, frac_est=frac, n_est=n_est, seed=0, device=args.device, pop_rdm=agg)
    for r in ev:
        r.update({"frac_est": frac, "fold": args.fold, "seed": args.seed, "kind": args.kind, "infer": "amortized"})
    rows += ev
    if args.direct and args.kind != "D0":
        ev = evaluate(model, test, n_faces, frac_est=frac, n_est=n_est, seed=0, device=args.device, pop_rdm=agg, direct=True)
        for r in ev:
            r.update({"frac_est": frac, "fold": args.fold, "seed": args.seed, "kind": args.kind, "infer": "direct"})
        rows += ev
# learning curve on number of estimation trials
for n_est in [5, 10, 20, 40]:
    ev = evaluate(model, test, n_faces, frac_est=None, n_est=n_est, seed=0, device=args.device, quick=True)
    for r in ev:
        rows.append({"pid": r["pid"], "n_est_fixed": n_est, "rho_hold": r["rho_hold"], "fold": args.fold, "seed": args.seed, "kind": args.kind})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "eval.csv"), index=False)
for infer, m in df[df.frac_est == 0.7].groupby("infer"):
  print(infer)
  print("hold rho: model %.3f | est-RDM %.3f | agg %.3f | split-half(est) %.3f" % (
    m.rho_hold.mean(), m.rho_hold_estrdm.mean(), m.rho_hold_pop.mean(), m.rho_splithalf_est.mean()))
  if "rho_tr_all" in m:
    print("TR rho: model all %.3f unseen %.3f seen %.3f | est-RDM all %.3f | agg all %.3f unseen %.3f" % (
        m.rho_tr_all.mean(), m.rho_tr_unseen.mean(), m.rho_tr_seen.mean(), m.rho_tr_all_estrdm.mean(),
        m.rho_tr_all_pop.mean(), m.rho_tr_unseen_pop.mean()))
