"""Build Experiment-2 processed tables.

Positions come from Meadows_Face_Arrangement_Task_v_v{3,6,7,8}_annotations.csv
(long format); trial-exclusion flags are recomputed from the MA task log in the
matching *_tree.json (same rules as Exp1). Test-retest trials (TR_set{k}_rep{r})
are appended to the main task.
Outputs: exp2_placements.parquet, exp2_participants.csv, exp2_faces.csv
Demographics (Connect questionnaires) are merged where available; SONA
demographics are added later from the authors' participant meta tables.
"""
import json, os, sys, glob, ast
import numpy as np, pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RAW, PROC, ROOT
from face_id.meadows import trial_exclusion_flag, separate_log_into_trials

D2 = os.path.join(RAW, "analysis_code_and_data/data_Experiment2")
faces = pd.read_csv(os.path.join(RAW, "analysis_code_and_data/stimuli_Experiment2/Experiment_2_face_image_filenames_and_source.csv"))
print(faces.columns.tolist())
fn = np.sort(faces.Experiment_2_FileName.to_numpy())
fdf = pd.DataFrame({"face": [os.path.splitext(x)[0] for x in fn], "FileName": fn, "face_idx": np.arange(len(fn))})
fdf = fdf.merge(faces.rename(columns={"Experiment_2_FileName": "FileName"}), on="FileName", how="left")
fdf.to_csv(os.path.join(PROC, "exp2_faces.csv"), index=False)
face2idx = dict(zip(fdf.face, fdf.face_idx))

versions = {3: ("SONA", None), 6: ("Connect", "white"), 7: ("Connect", "asian"), 8: ("Connect", "black")}
MA_IDX = 4
pl, rows = [], []
for v, (src, vrace) in versions.items():
    ann = pd.read_csv(os.path.join(D2, f"Meadows_Face_Arrangement_Task_v_v{v}_annotations.csv"))
    meta = pd.read_csv(os.path.join(D2, f"Meadows_Face_Arrangement_Task_v_v{v}_metadata_hashed.csv"))
    tree = json.load(open(os.path.join(D2, f"Meadows_Face_Arrangement_Task_v_v{v}_tree.json")))
    fin = meta[meta.status == "finished"]
    print(f"v{v} {src}: annotations participants {ann.participation.nunique()}, meta finished {len(fin)}, tree {len(tree)}")
    for p in tqdm(ann.participation.unique()):
        if p not in tree: print("no tree", p); continue
        tasks = tree[p]["tasks"]
        ma = tasks[MA_IDX]
        a = ann[ann.participation == p]
        ma_a = a[a.task == "MA_face"]
        n_trials = int(ma_a.trial.max()) + 1
        logs = separate_log_into_trials(ma["log"])
        # the authors' splitter asserts len(logs)==n_trials; keep but note mismatches
        n_log = len(logs)
        excl = {}
        for t in range(n_trials):
            if t < n_log:
                ex, _ = trial_exclusion_flag(logs[t])
            else:
                ex = False
            excl[t] = ex
        for _, r in ma_a.iterrows():
            pl.append((p, "MA", int(r.trial), r.stim1_name, float(r.x), float(r.y), excl[int(r.trial)]))
        # test-retest
        for t in tasks[MA_IDX + 1:]:
            nm = t.get("task"); nm = ast.literal_eval(nm)["name"] if isinstance(nm, str) else nm.get("name")
            if not nm or not nm.startswith("TR_"): continue
            ta = a[a.task == nm]
            if len(ta) == 0: continue
            ex, _ = trial_exclusion_flag(t["log"]) if t.get("log") else (False, {})
            for _, r in ta.iterrows():
                pl.append((p, nm, 0, r.stim1_name, float(r.x), float(r.y), ex))
        rows.append({"pid": p, "source": src, "version": v, "version_race": vrace,
                     "n_trials": n_trials, "n_log_trials": n_log, "n_excluded": int(sum(excl.values())),
                     "n_stimuli": len(ma["stimuli"]), "min_evidence": ma.get("min_evidence", np.nan),
                     "meadows_status": meta.set_index("name").status.get(p, np.nan)})
pl = pd.DataFrame(pl, columns=["pid", "task", "trial", "face", "x", "y", "excluded"])
part = pd.DataFrame(rows)
part["frac_excluded"] = part.n_excluded / part.n_trials
part["keep"] = (part.frac_excluded <= 0.5) & (part.meadows_status == "finished")
print("log/trial count mismatches:", (part.n_log_trials != part.n_trials).sum())
print("kept:", part.keep.sum(), "by source", part[part.keep].source.value_counts().to_dict(), "(paper: 433 UCSD + 1250 online = 1683)")

# Connect questionnaires (demographics)
QD = os.path.join(ROOT, "data/raw/unzip/exp2/Connect_server_data_hashed")
qrows = {}
for f in glob.glob(os.path.join(QD, "*questionnaire*")):
    try:
        q = pd.read_csv(f)
    except Exception as e:
        print("bad", f, e); continue
    if "task" not in q.columns or "Meadows_id_from_URL" not in q.columns:
        continue
    mead = q.Meadows_id_from_URL.iloc[0]
    rec = {"pid": mead, "q_file": os.path.basename(f), "q_start": q.start_time.iloc[0] if "start_time" in q else None}
    for tn in ["demographic_survey_p1", "demographic_survey_p2"]:
        s = q[q.task == tn]
        if len(s):
            try:
                rec.update({f"Q_{k}": v for k, v in json.loads(s.response.iloc[0]).items()})
            except Exception:
                try: rec.update({f"Q_{k}": v for k, v in ast.literal_eval(s.response.iloc[0]).items()})
                except Exception: pass
    if mead is None: continue
    # keep latest questionnaire per Meadows id
    if mead not in qrows or str(rec["q_start"]) > str(qrows[mead]["q_start"]):
        qrows[mead] = rec
qdf = pd.DataFrame(list(qrows.values()))
print("Connect questionnaires matched:", qdf.pid.isin(part.pid).sum())
part = part.merge(qdf, on="pid", how="left")

pl["face_idx"] = pl.face.map(face2idx)
assert pl.face_idx.notna().all(), pl[pl.face_idx.isna()].face.unique()[:10]
pl["face_idx"] = pl.face_idx.astype(int)
pl.to_parquet(os.path.join(PROC, "exp2_placements.parquet"), index=False)
part.to_csv(os.path.join(PROC, "exp2_participants.csv"), index=False)
print(pl.shape, part.shape, pl.task.value_counts().head(3).to_dict())
