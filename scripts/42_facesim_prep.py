"""Convert FaceSim3D (Hofmann et al., OSF 4a2jf; 100 CFD faces, triplet odd-one-out, 2D and 3D viewing conditions)
trial data into the AVFS-style JSON consumed by 27/28 (one 'trial' per participant: triplet_questions, odd_one_out_positions, annotator_id).
QC: drop catch-failed trials ('caught'), trials without a response, participants whose caught rate > 0.2 or with < 20 trials.
Outputs data/processed/facesim3d_{2D,3D}.json and data/processed/facesim3d_participants.csv.
Run: uv run python -I scripts/42_facesim_prep.py
"""
import glob, json, numpy as np, pandas as pd
R = "data/raw/facesim3d/main/"
d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(R + "*TrialResults.csv"))], ignore_index=True)
pdet = pd.read_csv(R + "2023-05-10_UXFData.FaceSim.ParticipantDetails.csv")
cond = pdet.drop_duplicates("ppid").set_index("ppid").group_exp
d["cond"] = d.ppid.map(cond)
d = d[d.cond.isin(["2D", "3D"]) & d.head_odd.notna() & d.head1.notna()]
caught_rate = d.groupby("ppid").caught.apply(lambda x: (x == True).mean())
d = d[d.caught != True]
n = d.groupby("ppid").size(); keep = n[(n >= 20)].index.intersection(caught_rate[caught_rate <= 0.2].index)
d = d[d.ppid.isin(keep)]
for c in ("head1", "head2", "head3", "head_odd"): d[c] = d[c].astype(int)
ok = (d.head_odd == d.head1) | (d.head_odd == d.head2) | (d.head_odd == d.head3); d = d[ok]
print("after QC: trials", len(d), "participants", d.ppid.nunique(), d.groupby("cond").ppid.nunique().to_dict(), "faces", pd.unique(d[["head1", "head2", "head3"]].values.ravel()).size)
rows = []
for cnd in ("2D", "3D"):
    sub = d[d.cond == cnd]; out = {}
    for i, (pid, g) in enumerate(sub.groupby("ppid")):
        T = g[["head1", "head2", "head3"]].to_numpy(); pos = (T == g.head_odd.to_numpy()[:, None]).argmax(1)
        out[str(i)] = dict(triplet_questions=T.tolist(), odd_one_out_positions=pos.tolist(), annotator_id=pid)
        rows.append(dict(ppid=pid, cond=cnd, n_trials=len(g), caught_rate=float(caught_rate[pid])))
    json.dump(out, open(f"data/processed/facesim3d_{cnd}.json", "w"))
    nn = sub.groupby("ppid").size(); print(f"{cnd}: participants {len(out)}, trials {len(sub)}, trials/participant mean {nn.mean():.0f} median {nn.median():.0f}, unique triplets {sub.triplet_id.nunique()}, judgments/triplet {len(sub)/sub.triplet_id.nunique():.2f}")
P = pd.DataFrame(rows).merge(pdet.drop_duplicates("ppid")[["ppid", "age_years", "ethnic", "sex"]], on="ppid", how="left"); P.to_csv("data/processed/facesim3d_participants.csv", index=False)
print(P.sex.value_counts().to_dict(), P.ethnic.value_counts().head(5).to_dict())
