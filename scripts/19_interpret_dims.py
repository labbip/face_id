"""Interpretation of embedding dimensions and of individual differences.
 - Exp1: correlate each embedding dimension with Lin et al. 100 trait ratings (top traits) [uses 08 logic].
 - Exp2: correlate dimensions with face race / gender (one-hot) and report which D1 weights vary most across
   participants (SD of log w per dim), and the correlation of the weights with perceiver race/gender.
usage: uv run python scripts/19_interpret_dims.py --exp 2 --kind D1 --tag _nonneg [--fold 0]"""
import argparse, os, sys, json, pickle
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr, f_oneway
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RESULTS, PROC, RAW
from face_id.models import FaceID
from face_id.data import split_trials
from face_id.train import infer_z
ap = argparse.ArgumentParser(); ap.add_argument("--exp", type=int, default=2); ap.add_argument("--kind", default="D1"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--D", type=int, default=8)
ap.add_argument("--tag", default="_nonneg"); ap.add_argument("--fold", type=int, default=0); args = ap.parse_args()
run = os.path.join(RESULTS, "runs", f"exp{args.exp}_{args.kind}_D{args.D}_M{args.M}_f{args.fold}_s0{args.tag}"); a = json.load(open(os.path.join(run, "args.json")))
parts, nf = pickle.load(open(os.path.join(PROC, f"exp{args.exp}_participants.pkl"), "rb"))
faces = pd.read_csv(os.path.join(PROC, f"exp{args.exp}_faces.csv")); part = pd.read_csv(os.path.join(PROC, f"exp{args.exp}_participants.csv"), low_memory=False).set_index("pid")
m = FaceID(nf, a["D"], a["M"], a["kind"], a["Dp"], nonneg=a.get("nonneg", False)).cuda(); m.load_state_dict(torch.load(os.path.join(run, "model.pt"))); m.eval()
f = m.emb().detach().cpu().numpy(); print("embedding: nonneg=%s, sparsity (frac < 0.05 of max) = %.2f, dim norms:" % (a.get("nonneg"), (f < 0.05 * f.max()).mean()), np.round(np.linalg.norm(f, axis=0), 2))
if args.exp == 1:
    L = os.path.join(RAW, "analysis_code_and_data/prior_study/Lin_2021"); rat = pd.read_csv(os.path.join(L, "AverageRating_RT100.csv"))
    rat["imageID"] = rat.imageID.str.replace("image ", "").astype(int); rat = rat.sort_values("imageID").reset_index(drop=True); traits = [c for c in rat.columns if c != "imageID"]; X = rat[traits].to_numpy(float)
    for d in range(f.shape[1]):
        rs = sorted([(spearmanr(f[:, d], X[:, k]).statistic, t) for k, t in enumerate(traits)], key=lambda x: -abs(x[0]))[:5]
        print(f"dim {d}: " + ", ".join(f"{t} {r:+.2f}" for r, t in rs))
else:
    fr = faces.Race.astype(str).str.lower(); fg = faces.Gender.astype(str).str.lower()
    for d in range(f.shape[1]):
        gr = {r: f[fr.to_numpy() == r, d].mean() for r in ("white", "black", "asian")}; gg = {g: f[fg.to_numpy() == g, d].mean() for g in fg.unique()}
        pr = f_oneway(*[f[fr.to_numpy() == r, d] for r in ("white", "black", "asian")]).pvalue
        print(f"dim {d}: race means " + ", ".join(f"{k} {v:.2f}" for k, v in gr.items()) + f" (ANOVA p={pr:.1e}); gender means " + ", ".join(f"{k} {v:.2f}" for k, v in gg.items()))
# individual differences: decoded weights
est = [split_trials(p, 0.7, 0)[0] for p in parts]; Z = np.concatenate([infer_z(m, est[b:b + 256], "cuda")[0].cpu().numpy() for b in range(0, len(parts), 256)])
W = m.dec.weights(torch.as_tensor(Z).cuda()).detach().cpu().numpy()
if args.kind == "D1":
    lw = np.log(W); print("SD of log-weight across participants per dim:", np.round(lw.std(0), 3), "| mean weight:", np.round(W.mean(0), 2))
    meta = pd.DataFrame({"pid": [p.pid for p in parts]}).join(part, on="pid")
    for d in range(lw.shape[1]):
        out = []
        for col in ["race_norm", "gender_norm"]:
            grp = [lw[(meta[col] == v).to_numpy(), d] for v in meta[col].dropna().unique() if (meta[col] == v).sum() >= 30]
            if len(grp) >= 2: out.append(f"{col} ANOVA p={f_oneway(*grp).pvalue:.1e}")
        print(f"dim {d}: " + "; ".join(out))
    np.save(os.path.join(RESULTS, f"weights_exp{args.exp}_{args.kind}{args.tag}_f{args.fold}.npy"), W)
else:
    A = W.reshape(len(W), m.dec.Dp, m.dec.D); print("A(z): mean |off-diagonal| / mean |diagonal| = %.3f; SD across participants of diag %.3f, off-diag %.3f" % (
        np.abs(A - np.eye(A.shape[1])[None] * A).mean() / np.abs(np.einsum("pii->pi", A)).mean(), np.einsum("pii->pi", A).std(0).mean(), (A - np.eye(A.shape[1])[None] * A).std(0).mean()))
