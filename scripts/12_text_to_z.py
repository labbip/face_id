"""H6: free-text strategy descriptions (Exp1 questionnaire) -> z_text -> arrangements of unseen participants.
Sentence embeddings (all-MiniLM-L6-v2 via sentence-transformers, run with `uv run --with sentence-transformers`),
ridge regression text-embedding -> z fitted on training-fold participants (z from the fold's eval.csv of OTHER folds
is not available for training participants, so we use the fold model's encoder on training participants' trials),
then evaluate on fold-out participants: decode z_text and compute rho_hold vs population and vs z from arrangements.
usage: uv run --with sentence-transformers python scripts/12_text_to_z.py --exp 1 --kind D3 --M 16 --tag _cv --fold 0
"""
import argparse, os, sys, json, pickle
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, PROC
from face_id.models import FaceID
from face_id.data import make_folds, split_trials, trials_to_rdm
from face_id.train import infer_z, predict_full_rdm
from face_id.rdm import nan_spearman
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=1); ap.add_argument("--kind", default="D3"); ap.add_argument("--M", type=int, default=16)
ap.add_argument("--tag", default="_cv"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--D", type=int, default=8)
args = ap.parse_args()
run = os.path.join(RESULTS, "runs", f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f{args.fold}_s{args.seed}{args.tag}")
a = json.load(open(os.path.join(run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
folds = make_folds(parts, a["n_folds"], 0, ("Gender", "Race") if args.exp == 1 else ("source", "version_race"))
part = pd.read_csv(os.path.join(PROC, f"exp{args.exp}_participants.csv"), low_memory=False).set_index("pid")
tcol = ["Your Strategy", "Traits Used"] if args.exp == 1 else ["Strategy", "TraitsUsed"]
texts = {p.pid: " ".join(str(part.loc[p.pid, c]) for c in tcol if c in part.columns) for p in parts}
cache = os.path.join(PROC, f"exp{args.exp}_text_emb.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); emb = dict(zip(z["pids"], z["emb"]))
else:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2"); pids = list(texts); E = st.encode([texts[p] for p in pids], batch_size=64, show_progress_bar=False)
    np.savez(cache, pids=np.array(pids), emb=E); emb = dict(zip(pids, E))
m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"]).cuda(); m.load_state_dict(torch.load(os.path.join(run, "model.pt"))); m.eval()
train = [p for p, f in zip(parts, folds) if f != args.fold]; test = [p for p, f in zip(parts, folds) if f == args.fold]
# z from arrangements (encoder on estimation trials; same split as evaluation)
def zs(ps):
    est = [split_trials(p, 0.7, 0)[0] for p in ps]; mu, _ = infer_z(m, est, "cuda"); return mu.cpu().numpy()
Ztr = zs(train); Zte = zs(test)
Xtr = np.array([emb[p.pid] for p in train]); Xte = np.array([emb[p.pid] for p in test])
ridge = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Xtr, Ztr); Zhat = ridge.predict(Xte)
r2 = 1 - ((Zte - Zhat) ** 2).sum(0) / ((Zte - Zte.mean(0)) ** 2).sum(0)
print("text -> z  out-of-fold R^2 per dim:", np.round(r2, 3), "| corr per dim:", np.round([np.corrcoef(Zte[:, k], Zhat[:, k])[0, 1] for k in range(Zte.shape[1])], 3))
# word-count control
wc = np.array([[len(texts[p.pid].split())] for p in train]); wct = np.array([[len(texts[p.pid].split())] for p in test])
Zwc = RidgeCV().fit(wc, Ztr).predict(wct)
hold = [split_trials(p, 0.7, 0)[1] for p in test]; H = [trials_to_rdm(h, nf) for h in hold]
P0 = predict_full_rdm(m, torch.zeros(1, a["M"]).cuda(), nf, "cuda")[0]
recs = {"pid": [p.pid for p in test], "fold": args.fold}
for nm, Z in [("z_arr", Zte), ("z_text", Zhat), ("z_wc", Zwc), ("pop", None)]:
    P = [P0] * len(test) if Z is None else predict_full_rdm(m, torch.as_tensor(Z, dtype=torch.float32).cuda(), nf, "cuda")
    recs[nm] = [nan_spearman(P[i], H[i]) for i in range(len(test))]
    print("%-22s hold rho %.3f" % (nm, np.nanmean(recs[nm])))
out = os.path.join(RESULTS, f"h6_text_exp{args.exp}_{args.kind}_M{args.M}{args.tag}.csv")
pd.DataFrame(recs).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
