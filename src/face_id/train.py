"""Training / evaluation loop for Face-ID models."""
from __future__ import annotations
import math, time, json, os
import numpy as np, torch
from .models import FaceID, trial_cosine_loss, kl_normal
from .data import tensorize, split_trials, trials_to_rdm
from .invmds import pair_lookup
from .rdm import nan_spearman


def random_enc_split(trials, rng, frac_lo=0.3, frac_hi=0.8):
    n = len(trials)
    if n < 3:
        return trials, trials
    k = int(rng.integers(max(1, int(frac_lo * n)), max(2, int(frac_hi * n)) + 1))
    k = min(max(k, 1), n - 1)
    idx = rng.permutation(n)
    return [trials[i] for i in idx[:k]], [trials[i] for i in idx[k:]]


def train_model(train_parts, n_faces, kind="D1", D=8, M=4, Dp=None, epochs=60, batch=128,
                lr=2e-3, beta=1e-3, seed=0, device="cuda", nonneg=False, log=None, val_parts=None,
                val_every=10, f_init=None, det=False, z_l2=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = FaceID(n_faces, D, M, kind, Dp, nonneg=nonneg, f_init=f_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(train_parts)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for b0 in range(0, n, batch):
            ps = [train_parts[i] for i in order[b0:b0 + batch]]
            if kind == "D0":
                enc_sets = [[] for _ in ps]; tgt_sets = [p.ma for p in ps]
            else:
                enc_sets, tgt_sets = zip(*[random_enc_split(p.ma, rng) for p in ps])
            E = tensorize(enc_sets, device); T = tensorize(tgt_sets, device)
            mu, logvar = model.encode(E["pi"], E["pj"], E["d"], E["k"], E["pidx"], len(ps), E["tidx"], E["n_trials"])
            if kind == "D0" or det:
                z = mu
            else:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            pred = model.decode(z, T["pi"], T["pj"], T["pidx"])
            loss = trial_cosine_loss(pred, T["d"], T["tidx"], T["n_trials"])
            if kind != "D0":
                loss = loss + (z_l2 * (mu ** 2).sum(-1).mean() if det else beta * kl_normal(mu, logvar))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        rec = {"epoch": ep, "loss": tot / nb}
        if val_parts is not None and (ep % val_every == val_every - 1 or ep == epochs - 1):
            ev = evaluate(model, val_parts, n_faces, frac_est=0.7, seed=0, device=device, quick=True)
            rec["val_hold_rho"] = float(np.nanmean([r["rho_hold"] for r in ev]))
        hist.append(rec)
        if log is not None:
            print(json.dumps(rec), file=log, flush=True)
    return model, hist


@torch.no_grad()
def infer_z(model, est_trials_list, device="cuda"):
    E = tensorize(est_trials_list, device)
    mu, logvar = model.encode(E["pi"], E["pj"], E["d"], E["k"], E["pidx"], len(est_trials_list), E["tidx"], E["n_trials"])
    return mu, logvar


def fit_z_direct(model, est_trials_list, device="cuda", steps=200, lr=0.05, l2=1e-2, init=None):
    """Non-amortised inference: optimise z per participant on the estimation trials
    (model frozen), with an L2 prior on z. Returns (n_part, M) tensor."""
    E = tensorize(est_trials_list, device)
    n = len(est_trials_list)
    z = torch.zeros(n, model.M, device=device) if init is None else init.clone().to(device)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        pred = model.decode(z, E["pi"], E["pj"], E["pidx"])
        # per-participant loss: mean over own trials; sum over participants so each is fitted independently
        dev = pred.device
        cnt = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], torch.ones_like(pred))
        mp = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], pred) / cnt
        md = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], E["d"]) / cnt
        pc = pred - mp[E["tidx"]]; dc = E["d"] - md[E["tidx"]]
        num = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], pc * dc)
        pp = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], pc * pc)
        dd = torch.zeros(E["n_trials"], device=dev).index_add_(0, E["tidx"], dc * dc)
        corr = num / torch.sqrt(pp * dd + 1e-8)
        # trial -> participant
        t2p = torch.zeros(E["n_trials"], dtype=torch.long, device=dev).index_reduce_(0, E["tidx"], E["pidx"], "amax", include_self=False)
        per_p = torch.zeros(n, device=dev).index_add_(0, t2p, 1 - corr) / torch.zeros(n, device=dev).index_add_(0, t2p, torch.ones_like(corr)).clamp_min(1)
        loss = per_p.sum() + l2 * (z ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach()


@torch.no_grad()
def predict_full_rdm(model, z, n_faces, device="cuda"):
    """Full pair vector (n_pairs,) for each row of z -> (n_part, n_pairs)."""
    iu = np.triu_indices(n_faces, k=1)
    pi = torch.as_tensor(iu[0], device=device); pj = torch.as_tensor(iu[1], device=device)
    out = []
    for s in range(z.shape[0]):
        pidx = torch.zeros(len(pi), dtype=torch.long, device=device)
        out.append(model.decode(z[s:s + 1], pi, pj, pidx).cpu().numpy())
    return np.array(out)


def tr_targets(p, n_faces):
    """Test-retest targets: per set, inverse-MDS RDM of the 3 reps (pairs vector with NaN),
    plus boolean mask of pairs whose both faces were in the participant's 20 MA faces."""
    sets = {}
    for t in p.tr:
        s = t.task.split("_")[1]  # set{k}
        sets.setdefault(s, []).append(t)
    out = []
    lut = pair_lookup(n_faces)
    seen = set(p.faces.tolist())
    for s, trials in sets.items():
        rdm = trials_to_rdm(trials, n_faces)
        obs = np.isfinite(rdm)
        ii, jj = np.triu_indices(n_faces, k=1)
        both_seen = np.array([(a in seen) and (b in seen) for a, b in zip(ii[obs], jj[obs])])
        out.append({"set": s, "rdm": rdm, "obs": obs, "both_seen": both_seen, "n_rep": len(trials)})
    return out


def evaluate(model, parts, n_faces, frac_est=0.7, n_est=None, seed=0, device="cuda", quick=False,
             pop_rdm=None, direct=False):
    """Per-participant metrics for fold-out participants.
    rho_hold : Spearman(pred, inverse-MDS RDM of held-out MA trials) over observed pairs
    rho_tr_all / rho_tr_unseen : Spearman over TR-set RDM pairs (all / at least one unseen face)
    Baselines computed in the same call when pop_rdm (D0/aggregate pairs vector) is given."""
    model.eval()
    est_sets, hold_sets = zip(*[split_trials(p, frac_est, seed, n_est) for p in parts])
    mu, logvar = infer_z(model, list(est_sets), device)
    if direct and model.kind != "D0":
        mu = fit_z_direct(model, list(est_sets), device, init=mu)
    pred = predict_full_rdm(model, mu, n_faces, device)
    res = []
    for s, p in enumerate(parts):
        hold = trials_to_rdm(hold_sets[s], n_faces)
        r = {"pid": p.pid, "n_est": len(est_sets[s]), "n_hold": len(hold_sets[s]),
             "rho_hold": nan_spearman(pred[s], hold)}
        r.update({f"z{m}": float(mu[s, m]) for m in range(mu.shape[1])})
        if not quick:
            est_rdm = trials_to_rdm(est_sets[s], n_faces)
            r["rho_hold_estrdm"] = nan_spearman(est_rdm, hold)          # own measured RDM baseline
            if pop_rdm is not None:
                r["rho_hold_pop"] = nan_spearman(pop_rdm, hold)
            # split-half ceiling on estimation trials
            e = est_sets[s]
            if len(e) >= 4:
                h = len(e) // 2
                a = trials_to_rdm(e[:h], n_faces); b = trials_to_rdm(e[h:], n_faces)
                r["rho_splithalf_est"] = nan_spearman(a, b)
            # test-retest targets
            if p.tr:
                ra, ru, rs, ba, bu, pa, pu = [], [], [], [], [], [], []
                for t in tr_targets(p, n_faces):
                    y = t["rdm"][t["obs"]]; x = pred[s][t["obs"]]
                    ra.append(nan_spearman(x, y))
                    if (~t["both_seen"]).sum() >= 5:
                        ru.append(nan_spearman(x[~t["both_seen"]], y[~t["both_seen"]]))
                    if t["both_seen"].sum() >= 5:
                        rs.append(nan_spearman(x[t["both_seen"]], y[t["both_seen"]]))
                    if pop_rdm is not None:
                        xp = pop_rdm[t["obs"]]
                        pa.append(nan_spearman(xp, y))
                        if (~t["both_seen"]).sum() >= 5:
                            pu.append(nan_spearman(xp[~t["both_seen"]], y[~t["both_seen"]]))
                    xe = est_rdm[t["obs"]]
                    ba.append(nan_spearman(xe, y))
                r["rho_tr_all"] = np.nanmean(ra) if ra else np.nan
                r["rho_tr_unseen"] = np.nanmean(ru) if ru else np.nan
                r["rho_tr_seen"] = np.nanmean(rs) if rs else np.nan
                r["rho_tr_all_estrdm"] = np.nanmean(ba) if ba else np.nan
                if pop_rdm is not None:
                    r["rho_tr_all_pop"] = np.nanmean(pa) if pa else np.nan
                    r["rho_tr_unseen_pop"] = np.nanmean(pu) if pu else np.nan
        res.append(r)
    return res
