"""Derivative 4: correspondence between the arrangement-based face space and Lin et al. (2021) trait ratings.
 - agg-RDM vs Euclidean RDM of 100-trait mean ratings (z-scored) and of the 4 factor scores
 - per-trait Spearman between agg-RDM and single-trait RDMs (top traits)
 - (if a model run is given) the learned D0 embedding RDM vs the same, and CCA/Procrustes between embedding and factor scores
usage: uv run python scripts/08_lin_correspondence.py [run_dir]
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RAW, RESULTS, PROC
from face_id.rdm import nan_spearman
L = os.path.join(RAW, "analysis_code_and_data/prior_study/Lin_2021")
faces = pd.read_csv(os.path.join(PROC, "exp1_faces.csv"))
rat = pd.read_csv(os.path.join(L, "AverageRating_RT100.csv"))
rat["imageID"] = rat.imageID.str.replace("image ", "").astype(int)
rat = rat.sort_values("imageID").reset_index(drop=True)
assert (rat.imageID.to_numpy() == faces.imageID.to_numpy()).all()
traits = [c for c in rat.columns if c != "imageID"]
X = rat[traits].to_numpy(float); Xz = (X - X.mean(0)) / X.std(0)
iu = np.triu_indices(100, k=1)
def erdm(A): return np.sqrt(((A[iu[0]] - A[iu[1]]) ** 2).sum(1))
agg = np.load(os.path.join(RESULTS, "repro_exp1_aggRDM_v.npy"))
print("agg vs Lin 100-trait Euclidean RDM: rho=%.3f" % nan_spearman(agg, erdm(Xz)))
fs = pd.read_csv(os.path.join(L, "17FactorScore_Faces_EFA.csv"))
fs["imageID"] = fs.image.astype(str).str.extract(r"(\d+)")[0].astype(int); fs = fs.sort_values("imageID").reset_index(drop=True)
assert (fs.imageID.to_numpy() == faces.imageID.to_numpy()).all()
fnames = ["Warmth", "Competence", "Femininity", "Youth"]; F = fs[fnames].to_numpy(float)
print("agg vs Lin 4-factor Euclidean RDM: rho=%.3f" % nan_spearman(agg, erdm((F - F.mean(0)) / F.std(0))))
tr = sorted([(nan_spearman(agg, np.abs(Xz[iu[0], k] - Xz[iu[1], k])), t) for k, t in enumerate(traits)], reverse=True)
print("top single traits:", [(t, round(r, 3)) for r, t in tr[:10]]); print("bottom:", [(t, round(r, 3)) for r, t in tr[-5:]])
if len(sys.argv) > 1:
    f = np.load(os.path.join(sys.argv[1], "faces.npy"))
    print("model embedding RDM vs agg: %.3f | vs Lin100: %.3f | vs Lin4: %.3f" % (nan_spearman(erdm(f), agg), nan_spearman(erdm(f), erdm(Xz)), nan_spearman(erdm(f), erdm(F))))
    cca = CCA(n_components=min(4, f.shape[1])).fit(f, F); U, V = cca.transform(f, F)
    print("CCA canonical correlations (embedding vs 4 factors):", [round(np.corrcoef(U[:, i], V[:, i])[0, 1], 3) for i in range(U.shape[1])])
    # linear predictability of each Lin factor from the embedding (LOO-CV R^2)
    from sklearn.linear_model import RidgeCV; from sklearn.model_selection import cross_val_predict
    for i, nm in enumerate(fnames):
        yhat = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), f, F[:, i], cv=10)
        print(f"  {nm}: LOO-R2 = {1 - ((F[:, i] - yhat) ** 2).sum() / ((F[:, i] - F[:, i].mean()) ** 2).sum():.3f}")
    # which traits align with each embedding dimension (|r| top 3)
    for d in range(f.shape[1]):
        rs = sorted([(abs(spearmanr(f[:, d], X[:, k]).statistic), spearmanr(f[:, d], X[:, k]).statistic, t) for k, t in enumerate(traits)], reverse=True)[:4]
        print(f"dim {d}:", [(t, round(r, 2)) for _, r, t in rs])
