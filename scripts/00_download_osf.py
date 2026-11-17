"""Download OSF nm7y8 (Xu et al. 2026) into data/raw/osf, skipping large derived
artefacts (VGG16 zips, bootstrapped RDM zips, permutation_test.zip, segmentation
.npy, plots, jspsych). Pass --all to fetch everything. File list was obtained
via the OSF v2 API and stored in data/raw/osf/_filelist.json."""
import json, os, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/raw/osf")
lst = json.load(open(os.path.join(OUT, "_filelist.json")))
allf = "--all" in sys.argv

def skip(p):
    if allf: return False
    if p.endswith(("VGG16_results.zip", "bootstrapped_participant_RDMs.zip", "permutation_test.zip",
                   "face_segmentation.npy", "multi_race_face_segmentation_v2.npy",
                   "shape_predictor_68_face_landmarks.dat", ".tar.gz")): return True
    if "/plots_Experiment" in p or "/jspsych-" in p: return True
    if p.endswith((".png", ".svg")): return True
    return False

def prio(p):
    if p.endswith((".txt", ".md", ".yml", ".py")): return 0
    if p.endswith(".ipynb") or "/stimuli_" in p: return 1
    if "/data_Experiment" in p and not p.endswith("processed.zip"): return 2
    if "/prior_study/" in p: return 3
    if "/results_Experiment" in p: return 4
    return 5

todo = sorted([(p, s, u) for p, s, u in lst if not skip(p)], key=lambda t: (prio(t[0]), t[1] or 0))
print(f"{len(todo)} files, {sum((s or 0) for _, s, _ in todo)/1e9:.2f} GB", flush=True)
log = open(os.path.join(ROOT, "logs/download.log"), "a")
for i, (p, s, u) in enumerate(todo):
    dst = os.path.join(OUT, p)
    if os.path.exists(dst) and (s is None or os.path.getsize(dst) == s):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ok = False
    for attempt in range(6):
        try:
            t0 = time.time()
            urllib.request.urlretrieve(u, dst + ".part")
            os.replace(dst + ".part", dst)
            msg = f"[{i+1}/{len(todo)}] {p} {os.path.getsize(dst)} B {time.time()-t0:.1f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            ok = True
            break
        except Exception as e:
            print("retry", p, e, flush=True)
            time.sleep(15 * (attempt + 1))
    if not ok:
        log.write(f"FAILED {p}\n"); log.flush()
    time.sleep(0.5)
print("DONE", flush=True); log.write("DONE\n")
