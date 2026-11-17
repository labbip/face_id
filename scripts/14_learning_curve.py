"""Learning curve with a fixed held-out set: re-evaluate saved CV models with n_est in {3,5,10,20,40,all}
estimation trials (subset of the 70% estimation split; held-out 30% unchanged), amortised and direct inference.
usage: uv run python scripts/14_learning_curve.py --exp 1 --kind D3 --M 16 --tag _cv [--seed 0]
"""
import argparse, glob, os, sys, json, pickle
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, PROC
from face_id.models import FaceID
from face_id.data import make_folds
from face_id.train import evaluate
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--kind", default="D3"); ap.add_argument("--M", type=int, default=16)
ap.add_argument("--tag", default="_cv"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--D", type=int, default=8)
args = ap.parse_args()
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
rows = []
for d in sorted(glob.glob(os.path.join(RESULTS, "runs", f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f*_s{args.seed}{args.tag}"))):
    a = json.load(open(os.path.join(d, "args.json")))
    folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if args.exp == 1 else ("source", "version_race"))
    test = [p for p, f in zip(parts, folds) if f == a["fold"]]
    m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(d, "model.pt"))); m.eval()
    for n_est in [3, 5, 10, 20, 40, None]:
        for direct in ([False, True] if args.kind != "D0" else [False]):
            ev = evaluate(m, test, nf, frac_est=0.7, n_est=n_est, seed=0, device="cuda", quick=True, direct=direct)
            for r in ev:
                rows.append({"pid": r["pid"], "fold": a["fold"], "n_est": r["n_est"], "n_est_req": -1 if n_est is None else n_est, "rho_hold": r["rho_hold"], "infer": "direct" if direct else "amortized"})
    print(d, "done", flush=True)
df = pd.DataFrame(rows); out = os.path.join(RESULTS, f"learning_curve_exp{args.exp}_{args.kind}_M{args.M}{args.tag}_s{args.seed}.csv"); df.to_csv(out, index=False)
print(df.groupby(["infer", "n_est_req"]).rho_hold.agg(["mean", "count"]).round(3))
