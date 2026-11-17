"""Odorant embeddings for Keller & Vosshall (2016): PubChem CID -> SMILES (PUG REST, cached in data/raw/keller/pubchem_smiles.json)
-> RDKit 2D physico-chemical descriptors (results/keller_embed_desc.npy, standardised, constant/NaN columns dropped) and
Morgan count fingerprints r=2, 1024 bits (results/keller_embed_morgan.npy); CID order in results/keller_cids.npy.
Run: uv run --with openpyxl --with rdkit python -I scripts/58_keller_embed.py
"""
import json, os, urllib.request, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
w = pd.read_excel("data/raw/keller/keller_vosshall_2016_MOESM1.xlsx", header=2); raw = w["CID"].dropna().astype(str).str.strip().unique()
cids = sorted(int(float(c)) for c in raw if c.replace(".", "").isdigit()); odd = [c for c in raw if not c.replace(".", "").isdigit()]; print("CIDs", len(cids), "non-numeric CID entries (CAS?)", odd)
cache = "data/raw/keller/pubchem_smiles.json"; sm = json.load(open(cache)) if os.path.exists(cache) else {}
for c in odd:                                        # resolve CAS numbers typed into the CID column via PubChem name lookup
    try: r = json.load(urllib.request.urlopen(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{c}/cids/JSON", timeout=60))["IdentifierList"]["CID"][0]; sm[f"cas:{c}"] = r; cids.append(int(r)); print("CAS", c, "->", r)
    except Exception as e: print("CAS", c, "unresolved", e)
cids = sorted(set(cids))
todo = [c for c in cids if str(c) not in sm]
for i in range(0, len(todo), 100):
    b = todo[i:i + 100]; url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{','.join(map(str, b))}/property/SMILES,MolecularWeight/JSON"
    for p in json.load(urllib.request.urlopen(url, timeout=60))["PropertyTable"]["Properties"]: sm[str(p["CID"])] = {"smiles": p.get("SMILES") or p.get("IsomericSMILES") or p.get("ConnectivitySMILES"), "mw": p.get("MolecularWeight")}
    print(f"fetched {min(i + 100, len(todo))}/{len(todo)}", flush=True)
json.dump(sm, open(cache, "w"), indent=0)
mols = [Chem.MolFromSmiles(sm[str(c)]["smiles"]) for c in cids]; ok = np.array([m_ is not None for m_ in mols]); print("parsed", ok.sum(), "of", len(cids))
cids = [c for c, o in zip(cids, ok) if o]; mols = [m_ for m_ in mols if m_ is not None]
names = [n for n, _ in Descriptors.descList]
D = np.array([[f(m_) for _, f in Descriptors.descList] for m_ in mols], dtype=np.float64)
D[~np.isfinite(D)] = np.nan; keep = (np.nanstd(D, 0) > 0) & (np.isnan(D).mean(0) < 0.05); D = D[:, keep]; names = [n for n, k in zip(names, keep) if k]
D = np.where(np.isnan(D), np.nanmean(D, 0)[None], D); D = np.sign(D) * np.log1p(np.abs(D))      # tame heavy-tailed counts
D = (D - D.mean(0)) / D.std(0); print("descriptors kept", D.shape)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024); M = np.array([gen.GetCountFingerprintAsNumPy(m_) for m_ in mols], dtype=np.float32); print("morgan", M.shape, "active bits", (M.sum(0) > 0).sum())
os.makedirs("results", exist_ok=True); np.save("results/keller_cids.npy", np.array(cids)); np.save("results/keller_embed_desc.npy", D.astype(np.float32)); np.save("results/keller_embed_morgan.npy", M)
json.dump(names, open("results/keller_desc_names.json", "w"))
