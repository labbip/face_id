"""Merge the authors' Exp2 participant meta tables (results_Experiment2/{SONA,Connect}_participant_meta_df.csv)
into data/processed/exp2_participants.csv: final inclusion (SONA pre-screening exclusions), demographics."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import RAW, PROC
R2 = os.path.join(RAW, "analysis_code_and_data/results_Experiment2")
part = pd.read_csv(os.path.join(PROC, "exp2_participants.csv"))
sona = pd.read_csv(os.path.join(R2, "SONA_participant_meta_df.csv"), low_memory=False)
conn = pd.read_csv(os.path.join(R2, "Connect_participant_meta_df.csv"), low_memory=False)
print("SONA meta", sona.shape, "Exclude_participant:", sona.Exclude_participant.value_counts(dropna=False).to_dict())
print("Connect meta", conn.shape)
s = sona.set_index("Meadows_name")
c = conn.set_index("Meadows_name")
part["authors_exclude"] = np.nan
m = part.pid.isin(s.index)
part.loc[m, "authors_exclude"] = part.loc[m, "pid"].map(s.Exclude_participant).astype(float)
# SONA demographics
for col, src in [("Gender", "MA_Q_Gender"), ("Race", "MA_Q_Race"), ("Ethnicity", "MA_Q_Ethnicity"), ("Age", "MA_Q_Age"),
                 ("Language", "MA_Q_Language"), ("Education", "MA_Q_Education"), ("Strategy", "MA_Q_Your_Strategy"), ("TraitsUsed", "MA_Q_Traits_Used"),
                 ("CFMT_correct", "CFMT_correct"), ("Quiz_correct", "Quiz_correct")]:
    if src in s:
        v = part.pid.map(s[src]).astype(object); part[col] = v.where(m, part[col].astype(object) if col in part else np.nan)
mc = part.pid.isin(c.index)
def race_from_connect(r):
    labs = [k.replace("Q_race_", "") for k in ["Q_race_white", "Q_race_black", "Q_race_asian", "Q_race_native", "Q_race_pacific", "Q_race_other"] if k in r and isinstance(r[k], str) and r[k] not in ("", "nan")]
    return "+".join(labs) if labs else np.nan
vr = part.pid.map(c.apply(race_from_connect, axis=1)).astype(object); part["Race"] = vr.where(mc, part["Race"].astype(object))
for col, src in [("Gender", "Q_gender"), ("Ethnicity", "Q_hispanic"), ("Age", "Q_age"), ("Language", "Q_language"), ("Education", "Q_education"),
                 ("Strategy", "Q_strategy_used"), ("TraitsUsed", "Q_trait_used"), ("CFMT_correct", "CFMT_correct"), ("Quiz_correct", "Quiz_correct")]:
    if src in c:
        v = part.pid.map(c[src]).astype(object); part[col] = v.where(mc, part[col].astype(object) if col in part else np.nan)
part["keep_final"] = part.keep & (part.authors_exclude.fillna(0) == 0)
print("keep_final:", part.keep_final.sum(), part[part.keep_final].source.value_counts().to_dict(), "(paper 433 + 1250)")
print("Race (final):", part[part.keep_final].Race.value_counts().head(8).to_dict())
print("Gender (final):", part[part.keep_final].Gender.value_counts().head(6).to_dict())
part["keep_prescreen_only"] = part.keep
part["keep"] = part.keep_final
part.to_csv(os.path.join(PROC, "exp2_participants.csv"), index=False)
