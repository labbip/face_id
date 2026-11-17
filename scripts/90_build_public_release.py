"""Assemble the public release (reports/github_release_spec.md) into release/ from an allow-list, and check it.
Nothing is published by this script: it only writes the directory release/ (not tracked by the analysis repository).
Nothing derived from XPASS-Vis is included (result files, rows of the statistics tables, logs; results/paper_numbers.md is regenerated inside release/).
Included: src/, scripts/ (without the internal HTML report builders), pyproject.toml, uv.lock, figures/, aggregate results
(results/*.csv without participant rows, results/figdata/, paper_numbers.md, paper_numbers_agg.json, results/sim_recovery*/ JSON,
results/sim_lapis_like/calibration.csv), logs of the jobs behind the reported results (absolute paths replaced), and the documents in release_docs/.
release/ is also the working tree of the public repository (its .git is kept across rebuilds; commit and push there by hand after a rebuild).
Checks (any hit aborts and empties release/): identifier columns in CSV files, model / array / pickle files, absolute home paths, e-mail
addresses, the host name, the user name, long hexadecimal strings (hashed identifiers), references to internal documents, files over 1 MB.  --final additionally refuses unfilled <PLACEHOLDERS> in the documents.
Run: uv run python -I scripts/90_build_public_release.py [--final]
"""
import argparse, csv, glob, os, re, shutil, socket, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--out", default="release"); a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); OUT = a.out
ID_COLS = {"pid", "rater", "ann", "participant_id", "subject", "slot", "user", "user_id", "annotator", "worker", "workerid", "rater_id", "subj", "id"}
ROWS_PER_PERSON = ("avfs_screen_", "cross_individual_", "repro_exp")                     # results/*.csv without an id column whose rows are still one participant / participant pair
SKIP_SCRIPTS = ("20_build_report.py", "47_build_summary_html.py")                       # internal HTML reports
SKIP_LOGS = re.compile(r"smoke|^04_b\d|download|unzip|^uv_|51_lapis_images|36_ffhq|_figures_|72_effect_req|88b_summary|^90_|^0[0-2]_")   # smoke tests, downloads (URLs with access tokens), figure-layout iterations, raw-data preparation (lists the hashed participant file names)
BAD_EXT = (".pt", ".npy", ".npz", ".pkl", ".parquet", ".DS_Store")
HOME = re.compile(r"/home/[A-Za-z0-9_.-]+(/Research/20[12]61020_face)?"); MAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"); INTERNAL = re.compile(r"NOTES\.md|HANDOFF|CLAUDE\.md|PAPER_CONTENT|reports/[A-Za-z0-9_]+\.(md|html)")      # files that do not exist in the release
XPASS = re.compile("xpass", re.I)      # decision of 2026-11-17: nothing derived from the password-protected XPASS-Vis data goes into the release (the paper reports those results)
HOST = socket.gethostname(); USER = os.path.expanduser("~").rstrip("/").split("/")[-1]; HEX = re.compile(r"\b[0-9a-f]{32,}\b")
def has_id(path):
    with open(path, newline="") as fh: hdr = next(csv.reader(fh), [])
    return [h for h in hdr if h.strip().lower() in ID_COLS]
def put(src, dst=None, scrub=False):
    dst = os.path.join(OUT, dst or src); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if scrub:
        t = open(src, errors="replace").read(); t = re.sub(r"/home/[A-Za-z0-9_.-]+/Research/20[12]61020_face", ".", t); t = HOME.sub("~", t); t = re.sub(r"/tmp/claude-[^\s\"']+", "<tmp>", t); t = re.sub(r"\b%s\b" % re.escape(USER), "user", t); open(dst, "w").write(t)
    else: shutil.copy2(src, dst)
def clear():      # empty release/ but keep its .git (release/ is the working tree of the public repository; its history must survive a rebuild)
    for x in os.listdir(OUT) if os.path.exists(OUT) else []:
        if x != ".git": p_ = os.path.join(OUT, x); shutil.rmtree(p_) if os.path.isdir(p_) else os.remove(p_)
clear()
n = {}
def count(k): n[k] = n.get(k, 0) + 1
# --- code
for f in sorted(glob.glob("src/face_id/*.py")) + ["pyproject.toml", "uv.lock"]: put(f); count("code")
for f in sorted(glob.glob("scripts/*")):
    if os.path.isfile(f) and os.path.basename(f) not in SKIP_SCRIPTS: put(f); count("scripts")
# --- figures
for f in sorted(glob.glob("figures/*.pdf")) + ["figures/INDEX.md"]: put(f); count("figures")
# --- aggregate results
skipped = []
for f in sorted(glob.glob("results/*.csv")):
    b = os.path.basename(f)
    if has_id(f) or b.startswith(ROWS_PER_PERSON): skipped.append(b); continue
    if "xpass" in b.lower(): count("results not copied (XPASS-Vis)"); continue
    if b in ("stats_table.csv", "stats_correlations.csv"):      # the rows of the XPASS-Vis table and of the XPASS cross-domain correlations are dropped
        import pandas as pd
        t = pd.read_csv(f); os.makedirs(os.path.join(OUT, "results"), exist_ok=True); t[~t.label.str.contains("XPASS")].to_csv(os.path.join(OUT, f), index=False); count("results"); continue
    put(f); count("results")
for f in sorted(glob.glob("results/figdata/*.csv")): put(f); count("figdata")
for f in ["results/paper_numbers_agg.json", "results/sim_lapis_like/calibration.csv"] + sorted(glob.glob("results/sim_recovery*/*.json")): put(f); count("results")
# --- logs
for f in sorted(glob.glob("logs/*")):
    if not os.path.isfile(f) or SKIP_LOGS.search(os.path.basename(f)): continue
    if XPASS.search(os.path.basename(f)) or XPASS.search(open(f, errors="replace").read()): count("logs not copied (XPASS-Vis)"); continue
    put(f, scrub=True); count("logs")
# --- documents
for f in sorted(glob.glob("release_docs/*")): put(f, ".gitignore" if os.path.basename(f) == "gitignore" else os.path.basename(f)); count("docs")
# --- results/paper_numbers.md of the release: generated inside release/ from its own aggregate files (so it has no XPASS-Vis section and the reduced statistics tables)
r = subprocess.run([sys.executable, "-I", "scripts/73_build_paper_numbers.py"], cwd=OUT, env=dict(os.environ, FIGDATA="read"), capture_output=True, text=True)
if r.returncode: clear(); sys.exit("scripts/73 failed inside release/:\n" + r.stderr[-2000:])
count("results")
# --- checks
bad, todo = [], []
for d, ds, fs in os.walk(OUT):
    if ".git" in ds: ds.remove(".git")
    for x in fs:
        p = os.path.join(d, x); rel = os.path.relpath(p, OUT)
        if x.endswith(BAD_EXT): bad.append((rel, "forbidden file type"))
        if os.path.getsize(p) > 1_000_000 and rel != "uv.lock": bad.append((rel, f"{os.path.getsize(p) / 1e6:.1f} MB"))
        if x.endswith(".csv") and has_id(p): bad.append((rel, f"identifier column {has_id(p)}"))
        if x.endswith(".pdf"): continue
        t = open(p, errors="replace").read()
        if HOME.search(t): bad.append((rel, "absolute home path: " + HOME.search(t).group(0)))
        m = [q.group(0) for q in MAIL.finditer(t) if not q.group(0).endswith(("users.noreply.github.com",)) and "@" + q.group(0).split("@")[1] not in ("@v1", "@v2")]
        if m and not rel.startswith("uv.lock"): bad.append((rel, "e-mail address: " + m[0]))
        if HOST and HOST in t: bad.append((rel, "host name"))
        if rel.startswith(("results/", "logs/", "figures/")) and XPASS.search(t + rel): bad.append((rel, "XPASS-Vis-derived content"))
        m = INTERNAL.search(t)
        if m and rel not in ("scripts/90_build_public_release.py",): bad.append((rel, "reference to an internal document: " + m.group(0)))
        if d != OUT and re.search(r"\b%s\b" % re.escape(USER), t): bad.append((rel, "user name"))      # top-level documents may name the authors
        if HEX.search(t) and rel != "uv.lock": bad.append((rel, "long hexadecimal string (hashed identifier?): " + HEX.search(t).group(0)[:12] + "..."))
        if d == OUT and re.search(r"<[A-Z_]{3,}>", t): todo.append((rel, sorted(set(re.findall(r"<[A-Z_]{3,}>", t)))))
size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(OUT) if "/.git" not in d and not d.endswith(".git") for x in fs)
print("copied:", n, f"| total {size / 1e6:.1f} MB"); print(f"results/*.csv not copied (participant-level): {len(skipped)}")
for r in todo: print("placeholders to fill before publication:", *r)
if bad or (a.final and todo):
    for r in bad: print("CHECK FAILED:", *r)
    clear(); sys.exit("release/ emptied: fix the problems above" if bad else "release/ emptied: --final with unfilled placeholders")
print("all checks passed ->", OUT)
