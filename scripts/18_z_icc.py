"""H4: stability of z. For each fold model, encode fold-out participants from (a) the 70% estimation MA trials,
(b) the held-out 30% MA trials, (c) the test-retest trials (Exp1: mostly unseen faces; Exp2: seen faces).
Report ICC(2,1)-style agreement = correlation of each z dimension across the two encodings (participants as units)
and the predictive value of z_TR for held-out MA trials. usage: --exp 1 --kind D2 --M 4 --tag _cv"""
import argparse, os, sys, json, pickle
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, PROC
from face_id.models import FaceID
from face_id.data import make_folds, split_trials, trials_to_rdm
from face_id.train import infer_z, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--kind", default="D2"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--tag", default="_cv"); ap.add_argument("--D", type=int, default=8)
args = ap.parse_args()
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
folds = make_folds(parts, 10, 0, ("Gender", "Race") if args.exp == 1 else ("source", "version_race"))
def icc21(x, y):
    """ICC(2,1) for two raters (encodings) x, y over participants."""
    d = np.stack([x, y], 1); n, k = d.shape; gm = d.mean(); ms_r = k * ((d.mean(1) - gm) ** 2).sum() / (n - 1); ms_c = n * ((d.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
rows = []
for f in range(10):
    run = os.path.join(RESULTS, "runs", f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f{f}_s0{args.tag}"); a = json.load(open(os.path.join(run, "args.json")))
    m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(run, "model.pt"))); m.eval()
    test = [p for p, fo in zip(parts, folds) if fo == f and len(p.tr) >= 3]
    est, hold = zip(*[split_trials(p, 0.7, 0) for p in test]); tr = [p.tr for p in test]
    Ze = infer_z(m, list(est), "cuda")[0].cpu().numpy(); Zh = infer_z(m, list(hold), "cuda")[0].cpu().numpy(); Zt = infer_z(m, tr, "cuda")[0].cpu().numpy()
    H = [trials_to_rdm(h, nf) for h in hold]; P0 = predict_full_rdm(m, torch.zeros(1, a["M"]).cuda(), nf, "cuda")[0]
    Pe = predict_full_rdm(m, torch.as_tensor(Ze).cuda(), nf, "cuda"); Pt = predict_full_rdm(m, torch.as_tensor(Zt).cuda(), nf, "cuda")
    for d in range(a["M"]):
        rows.append({"fold": f, "dim": d, "icc_est_hold": icc21(Ze[:, d], Zh[:, d]), "icc_est_tr": icc21(Ze[:, d], Zt[:, d]), "r_est_hold": np.corrcoef(Ze[:, d], Zh[:, d])[0, 1], "r_est_tr": np.corrcoef(Ze[:, d], Zt[:, d])[0, 1],
                     "rho_hold_from_est": np.nanmean([nan_spearman(Pe[i], H[i]) for i in range(len(test))]), "rho_hold_from_tr": np.nanmean([nan_spearman(Pt[i], H[i]) for i in range(len(test))]),
                     "rho_hold_pop": np.nanmean([nan_spearman(P0, H[i]) for i in range(len(test))]), "n": len(test)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"z_icc_exp{args.exp}_{args.kind}_M{args.M}{args.tag}.csv"), index=False)
g = df.groupby("dim")[["icc_est_hold", "icc_est_tr", "r_est_hold", "r_est_tr"]].mean(); print(g.round(3))
print("ICC mean over dims: est vs hold %.3f | est vs test-retest %.3f" % (df.icc_est_hold.mean(), df.icc_est_tr.mean()))
print("held-out MA rho: z from est %.3f | z from test-retest trials %.3f | population %.3f (n per fold ~%d)" % (df.rho_hold_from_est.mean(), df.rho_hold_from_tr.mean(), df.rho_hold_pop.mean(), df.n.mean()))
