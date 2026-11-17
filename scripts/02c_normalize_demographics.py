"""Add harmonised demographic columns (gender_norm, race_norm, age_num, native_english) to both participant tables."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from face_id.paths import PROC

def norm_gender(x):
    x = str(x).strip().lower()
    if x in ("woman", "female", "f"): return "woman"
    if x in ("man", "male", "m"): return "man"
    if x in ("nan", "", "none"): return np.nan
    return "other"

def norm_race(x):
    x = str(x).strip().lower()
    if x in ("nan", "", "none"): return np.nan
    if "+" in x: return "multi"
    if x.startswith("white") or "caucasian" in x: return "white"
    if x.startswith("black") or "african" in x: return "black"
    if x.startswith("asian"): return "asian"
    return "other"

for exp in (1, 2):
    f = os.path.join(PROC, f"exp{exp}_participants.csv")
    d = pd.read_csv(f, low_memory=False)
    d["gender_norm"] = d.Gender.map(norm_gender)
    d["race_norm"] = d.Race.map(norm_race)
    d["age_num"] = pd.to_numeric(d.Age, errors="coerce")
    lang = d.Language.astype(str).str.strip().str.lower()
    d["native_english"] = lang.str.startswith("engl") | lang.isin(["american english", "english (us)"])
    d.to_csv(f, index=False)
    k = d[d.keep]
    print(f"exp{exp} kept {len(k)}: gender", k.gender_norm.value_counts(dropna=False).to_dict(), "race", k.race_norm.value_counts(dropna=False).to_dict(),
          "age mean %.1f" % k.age_num.mean(), "native english %.3f" % k.native_english.mean())
