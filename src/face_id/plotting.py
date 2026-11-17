"""Paper-figure style: PDF, bbox tight, transparent, Helvetica + usetex (sfmath),
one panel per PDF, no titles, tab10, no grid."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .paths import FIGURES

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}",
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.transparent": True, "pdf.fonttype": 42,
    "axes.prop_cycle": matplotlib.cycler(color=plt.get_cmap("tab10").colors),
})
TAB10 = plt.get_cmap("tab10").colors


def esc(s: str) -> str:
    """Escape _ % & # for usetex."""
    return re.sub(r"([_%&#])", r"\\\1", s)


def _index(path, text):
    """One line per file in figures/INDEX.md: replace the file's existing line in place, else append (2026-11-17; before, lines were only appended)."""
    idx = os.path.join(FIGURES, "INDEX.md"); key = f"- `{os.path.basename(path)}`"; line = f"{key} {text}\n"
    L = open(idx).readlines() if os.path.exists(idx) else []
    hit = [i for i, l in enumerate(L) if l.startswith(key + " ")]
    if hit: L[hit[0]] = line; L = [l for i, l in enumerate(L) if i not in hit[1:]]
    else: L.append(line)
    open(idx, "w").writelines(L)


def savefig(fig, name, w=3.3, h=2.6, index_line=None):
    fig.set_size_inches(w, h)
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name if name.endswith(".pdf") else name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    if index_line: _index(path, f"({w}x{h} in): {index_line}")
    return path


def pdf_size(name):
    """Page size (pt) of an existing figures/<name>.pdf, or None."""
    path = os.path.join(FIGURES, name if name.endswith(".pdf") else name + ".pdf")
    if not os.path.exists(path): return None
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", open(path, "rb").read())
    return (float(m.group(1)), float(m.group(2))) if m else None


def savefig_fixed(fig, name, size_pt, index_line=None):
    """Save on a fixed canvas of size_pt = (w, h) points (no bbox tight), for panels whose page size must not depend on the content
    (legend outside the axes). The figure should use layout="constrained"."""
    w, h = size_pt; fig.set_size_inches(w / 72, h / 72)
    path = os.path.join(FIGURES, name if name.endswith(".pdf") else name + ".pdf")
    with plt.rc_context({"savefig.bbox": "standard"}): fig.savefig(path)
    plt.close(fig)
    if index_line: _index(path, f"({w / 72:.2f}x{h / 72:.2f} in, fixed canvas {w:.1f}x{h:.1f} pt): {index_line}")
    return path


def dodge(ax, k, n, step_pt=4.0, axis="x"):
    """Transform that shifts series k of n by a constant visual amount ((k - (n - 1) / 2) * step_pt points) along x or y, on any axis scale
    (on a log axis this is a multiplicative shift). Use as ax.errorbar(..., transform=dodge(ax, k, n))."""
    from matplotlib.transforms import ScaledTranslation
    d = (k - (n - 1) / 2) * step_pt / 72
    return ax.transData + ScaledTranslation(d if axis == "x" else 0, d if axis == "y" else 0, ax.figure.dpi_scale_trans)


# ---- source data of the figures (2026-11-17, public-release spec §7): every panel is drawn from a small table of the plotted values, saved as
# results/figdata/<panel>.csv. With the environment variable FIGDATA=read the table is read back instead of being computed, so the panels can be
# redrawn without the per-participant evaluation files. In both modes the panel is drawn from the CSV that was just written / read (same numbers).
FROM_FIGDATA = os.environ.get("FIGDATA") == "read"


def figdata(name, compute):
    """compute() -> DataFrame with the plotted values of panel <name> (aggregate values only: means, s.e.m., histogram densities; no per-participant rows)."""
    import pandas as pd
    from .paths import RESULTS
    path = os.path.join(RESULTS, "figdata", os.path.basename(name).replace(".pdf", "") + ".csv")
    if not FROM_FIGDATA:
        os.makedirs(os.path.dirname(path), exist_ok=True); compute().to_csv(path, index=False)
    return pd.read_csv(path)
