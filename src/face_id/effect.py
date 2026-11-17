"""Effect-size views of personalisation that do not reward the shared (population) component:
  residual r   : Pearson r between (prediction - population item mean) and (rating - population item mean) over the rater's items
  contested r  : Pearson r restricted to items on which training raters disagree most (top tercile of between-rater SD, training raters only)
  contested-pair accuracy : for pairs of the rater's items whose population means are within 0.1 SD of each other (the population model
                 cannot order them) and which the rater rated differently, the fraction ordered correctly by the model (a tied prediction counts 0.5, so a model that cannot order them scores 0.5)
All quantities use only training raters to define the population mean and the contested items."""
import numpy as np

def item_disagreement(values, items, raters, min_n=5):
    """Between-rater SD per item from a long table (values already centred per rater); returns dict item -> SD (items with >= min_n ratings)."""
    import pandas as pd
    s = pd.DataFrame({"v": values, "i": items, "k": raters}).groupby("i").v.agg(["std", "count"]); s = s[s["count"] >= min_n]
    return s["std"].to_dict()

def contested_items(disp, frac=1 / 3):
    """Top `frac` of items by disagreement SD -> set of item ids."""
    import pandas as pd
    s = pd.Series(disp); thr = s.quantile(1 - frac); return set(s[s >= thr].index)

def rater_metrics(pred, y, pop, items, hi, rng, n_pairs=4000, tol=0.1, min_items=10, min_pairs=20):
    """One rater: pred, y, pop (population mean per item), items (ids), hi (set of contested items). Returns dict."""
    out = {"rr": np.nan, "rhi": np.nan, "pacc": np.nan, "npairs": 0}
    y = np.asarray(y, float); pred = np.asarray(pred, float); pop = np.asarray(pop, float)
    def corr(a, b): return np.corrcoef(a, b)[0, 1] if len(a) >= min_items and a.std() > 0 and b.std() > 0 else np.nan
    out["rr"] = corr(pred - pop, y - pop)
    m = np.array([i in hi for i in items]) if hi is not None else np.zeros(len(y), bool)
    if m.sum() >= min_items: out["rhi"] = corr(pred[m], y[m])
    n = len(y)
    if n >= 4:
        a_, b_ = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs); ok = (a_ != b_) & (np.abs(pop[a_] - pop[b_]) < tol * (pop.std() + 1e-9)) & (y[a_] != y[b_])
        if ok.sum() >= min_pairs:
            a_, b_ = a_[ok], b_[ok]; sp = np.sign(pred[a_] - pred[b_]); out["pacc"] = np.mean((sp == np.sign(y[a_] - y[b_])) + 0.5 * (sp == 0)); out["npairs"] = int(ok.sum())   # a tie in the prediction counts as a coin flip (0.5)
    return out

def metrics_long(pred, y, pop, kk, items, hi, n, seed=0):
    """Per-rater metrics for long arrays (rater index kk in 0..n-1). Returns dict of arrays (rr, rhi, pacc)."""
    rng = np.random.default_rng(seed); res = {k: np.full(n, np.nan) for k in ("rr", "rhi", "pacc")}
    order = np.argsort(kk, kind="stable"); kk_s = kk[order]; bounds = np.searchsorted(kk_s, np.arange(n + 1))
    for k in range(n):
        sl = order[bounds[k]:bounds[k + 1]]
        if len(sl) < 10: continue
        r = rater_metrics(pred[sl], y[sl], pop[sl], items[sl], hi, rng)
        for key in res: res[key][k] = r[key]
    return res
