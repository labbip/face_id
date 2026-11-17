"""Build Experiment-1 processed tables from raw Meadows tree.json files.

Outputs (data/processed/):
  exp1_placements.parquet : pid, task, trial, face, x, y, excluded   (MA + TR trials)
  exp1_participants.csv   : one row per stage-2 participant (kept after the paper's
                            50%-valid-trials rule) with meta, questionnaire, TR info
  exp1_faces.csv          : face name -> index (0..99) in Lin et al. imageID order
Trial exclusion replicates utility_function rules (0.9 s placement interval;
multi-select readjustment).
"""
import json, os, sys, glob
import numpy as np, pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RAW, PROC, ROOT
from face_id.meadows import parse_exp1_tree, parse_exp1_questionnaire, trial_exclusion_flag

UNZ = os.path.join(ROOT, "data/raw/unzip/exp1")
S2 = os.path.join(UNZ, "stage2_data_individual_hashed")
TR = os.path.join(UNZ, "test_retest")
os.makedirs(PROC, exist_ok=True)

# ---- faces: Lin et al. order (imageID 1..100 sorted) -------------------------
im = pd.read_csv(os.path.join(RAW, "analysis_code_and_data/prior_study/Lin_2021/imageOrder_MTurk.txt"), sep=r"\s+")
im = im.rename(columns={im.columns[0]: "imageID", im.columns[1]: "FileName"})[["imageID", "FileName"]]
im = im.sort_values("imageID").reset_index(drop=True)
im["face"] = im.FileName.str.replace(".png", "", regex=False).str.replace(".jpg", "", regex=False)
im["face_idx"] = np.arange(len(im))
im.to_csv(os.path.join(PROC, "exp1_faces.csv"), index=False)
face2idx = dict(zip(im.face, im.face_idx))

# ---- stage-2 participants ------------------------------------------------------
meta = pd.read_csv(os.path.join(RAW, "analysis_code_and_data/data_Experiment1/stage2_alldata_meta_df_20230710_hashed.csv"))
print("stage2 meta rows", len(meta))
rows, pl, qs = [], [], []
for _, r in tqdm(meta.iterrows(), total=len(meta)):
    f = os.path.join(S2, f"Meadows_Stage2_noTR_D1_v_{r.version}_{r.MeadowsID}_tree.json")
    if not os.path.exists(f):
        print("missing", f); continue
    df, m, d = parse_exp1_tree(f, r.MeadowsID)
    df["task"] = "MA"
    pl.append(df)
    q = parse_exp1_questionnaire(d); q["pid"] = r.MeadowsID
    qs.append(q)
    m.update({"MTurkID": r.MTurkID, "version": r.version, "batch": r.batch,
              "paper_num_trials": r.num_trials, "paper_excluded": r.excluded_num_trials_overall,
              "paper_min_evidence": r.min_evidence})
    rows.append(m)
part = pd.DataFrame(rows)
qdf = pd.DataFrame(qs)
part = part.merge(qdf, on="pid", how="left")
pl = pd.concat(pl, ignore_index=True)

# compare with the paper's exclusion counts
agree = (part.n_excluded == part.paper_excluded).mean()
print(f"trial-exclusion count agreement with paper meta: {agree:.4f}; n_trials agree {(part.n_trials==part.paper_num_trials).mean():.4f}")
part["frac_excluded"] = part.n_excluded / part.n_trials
part["keep"] = part.frac_excluded <= 0.5
print("kept participants (<=50% excluded):", part.keep.sum(), "(paper: 1751)")

# ---- test-retest ---------------------------------------------------------------
trm = pd.read_csv(os.path.join(TR, "Meadows_current_metadata_V4_20240729_hashed.csv"))
trm = trm[trm.tasks_finished >= 22]
official = pd.read_csv(os.path.join(TR, "test_retest_corr_spearman.csv"), index_col=0)
print("official TR participants", official.shape[1])
tr_rows = []
cand = {}
for _, r in tqdm(trm.iterrows(), total=len(trm)):
    f = os.path.join(TR, "meadows_data_hashed", f"Meadows_Face_MATR_v_v4_{r['name']}_tree.json")
    if not os.path.exists(f): continue
    d = json.load(open(f))
    out = []
    ok = True
    for t in d["tasks"]:
        nm = eval(t["task"])["name"] if isinstance(t["task"], str) else t["task"]["name"]
        if not nm.startswith("TR_"): continue
        if len(t["trials"]) == 0: ok = False; break
        trial = t["trials"][0]
        ex, info = trial_exclusion_flag(trial["log"])
        for p in trial["positions"]:
            out.append((r["name"], nm, 0, p["name"], float(p["x"]), float(p["y"]), ex))
    if not ok or len(out) == 0: continue
    cand.setdefault(r.workerId, []).append((r["name"], pd.DataFrame(out, columns=["tr_name", "task", "trial", "face", "x", "y", "excluded"])))
# resolve duplicates: keep the Meadows entry with the higher mean within-set spearman (as in the paper)
from face_id.invmds import trial_distances
from scipy.stats import spearmanr
def mean_tr_corr(df):
    cs = []
    for s in range(1, 6):
        reps = []
        for rep in range(1, 4):
            g = df[df.task == f"TR_set{s}_rep{rep}"].sort_values("face")
            reps.append(trial_distances(g[["x", "y"]].to_numpy()))
        for a in range(3):
            for b in range(a + 1, 3):
                cs.append(spearmanr(reps[a], reps[b]).statistic)
    return np.nanmean(cs)
tr_pl = []
mt2pid = dict(zip(part.MTurkID, part.pid))
for wid, lst in cand.items():
    if wid not in mt2pid: continue
    if len(lst) > 1:
        lst = sorted(lst, key=lambda x: -mean_tr_corr(x[1]))
    name, df = lst[0]
    df = df.copy(); df["pid"] = mt2pid[wid]
    tr_pl.append(df)
    tr_rows.append({"pid": mt2pid[wid], "tr_name": name, "tr_mean_rho": mean_tr_corr(df),
                    "tr_n_excluded": int(df.groupby("task").excluded.first().sum())})
tr_pl = pd.concat(tr_pl, ignore_index=True)
trdf = pd.DataFrame(tr_rows)
part = part.merge(trdf, on="pid", how="left")
part["has_tr"] = part.tr_name.notna()
print("TR matched to stage-2 participants:", part.has_tr.sum())

# overlap between TR faces and the participant's 20 MA faces
ma_faces = pl.groupby("pid").face.apply(set)
tr_faces = tr_pl.groupby("pid").face.apply(set)
ov = pd.Series({p: len(tr_faces[p] & ma_faces[p]) for p in tr_faces.index})
part["tr_n_faces"] = part.pid.map(tr_faces.apply(len))
part["tr_overlap_ma"] = part.pid.map(ov)
print("TR faces per participant:", part.tr_n_faces.describe().to_dict())
print("TR faces overlapping with own MA faces:", part.tr_overlap_ma.describe().to_dict())

all_pl = pd.concat([pl.assign(tr_name=np.nan), tr_pl.drop(columns=["tr_name"]).assign(tr_name=np.nan)], ignore_index=True)
all_pl["face_idx"] = all_pl.face.map(face2idx)
assert all_pl.face_idx.notna().all(), all_pl[all_pl.face_idx.isna()].face.unique()[:10]
all_pl["face_idx"] = all_pl.face_idx.astype(int)
all_pl = all_pl.drop(columns=["tr_name"])
all_pl.to_parquet(os.path.join(PROC, "exp1_placements.parquet"), index=False)
part.to_csv(os.path.join(PROC, "exp1_participants.csv"), index=False)
print(all_pl.groupby("task").size().head(3)); print(all_pl.shape, part.shape)
