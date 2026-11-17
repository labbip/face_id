"""Face-ID two-factor model.

Shared face embedding f_i (stimulus axis) × perceiver latent z (person axis).

Decoder hierarchy (H2):
    D0: delta_ij = ||f_i - f_j||                       (no individual differences)
    D1: delta_ij = ||w(z) * (f_i - f_j)||              (reweighting; Roads & Love type)
    D2: delta_ij = ||A(z) (f_i - f_j)||                (linear transform; rotation/mixing)
    D3: delta_ij = ||A(z)(f_i - f_j) + u_i(z) - u_j(z)|| (plus person-specific per-face offsets)

Trial-level loss: on-screen distances within a trial are compared with
predicted dissimilarities up to a free per-trial scale (cosine loss), which
mirrors inverse MDS where each trial has its own scaling factor.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, depth=2, act=nn.GELU):
    layers = []
    d = i
    for _ in range(depth):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Set encoder: pairs (f_i, f_j, d_ij) from the estimation trials → q(z|·).

    Pair features: [f_i + f_j, |f_i - f_j|, d_ij, 1/n_faces_in_trial]; symmetric in i,j.
    Pooling: attention (softmax over pairs of a participant).
    """
    def __init__(self, D, M, hidden=128):
        super().__init__()
        self.pair = mlp(2 * D + 4, hidden, hidden, depth=2)
        self.att = nn.Linear(hidden, 1)
        self.head = mlp(hidden, hidden, 2 * M, depth=1)
        self.M = M

    def forward(self, f, pi, pj, d, k, pidx, n_part, tidx=None, n_trials=None):
        """f: (F, D) embeddings; pi, pj: (P,) face ids; d: (P,) trial-normalized distances;
        k: (P,) number of faces in that trial; pidx: (P,) participant slot index in batch.
        tidx/n_trials: trial slots, used to RMS-normalise the population prediction within
        each trial so that the residual d - delta_pop is informative about the person."""
        fi, fj = f[pi], f[pj]
        delta = torch.sqrt(((fi - fj) ** 2).sum(-1) + 1e-8)
        if tidx is not None:
            ss = torch.zeros(n_trials, device=f.device).index_add_(0, tidx, delta ** 2)
            cnt = torch.zeros(n_trials, device=f.device).index_add_(0, tidx, torch.ones_like(delta))
            delta = delta / torch.sqrt(ss / cnt + 1e-8)[tidx]
        x = torch.cat([fi + fj, (fi - fj).abs(), d[:, None], 1.0 / k[:, None], delta[:, None], (d - delta)[:, None]], dim=1)
        h = self.pair(x)
        a = self.att(h).squeeze(-1)
        a = a - a.max()
        w = torch.exp(a)
        den = torch.zeros(n_part, device=f.device).index_add_(0, pidx, w) + 1e-12
        w = w / den[pidx]
        pooled = torch.zeros(n_part, h.shape[1], device=f.device).index_add_(0, pidx, h * w[:, None])
        out = self.head(pooled)
        mu, logvar = out[:, : self.M], out[:, self.M :]
        return mu, logvar.clamp(-8, 4)


class Decoder(nn.Module):
    def __init__(self, D, M, n_faces, kind="D1", Dp=None, hidden=64):
        super().__init__()
        self.kind = kind
        self.D = D
        self.Dp = Dp or D
        self.n_faces = n_faces
        if kind == "D1":
            self.net = mlp(M, hidden, D, depth=1)
        elif kind in ("D2", "D3"):
            self.net = mlp(M, hidden, self.Dp * D, depth=1)
            with torch.no_grad():  # start near identity (if square) for stability
                self.net[-1].weight.mul_(0.1)
                eye = torch.eye(self.Dp, D).flatten()
                self.net[-1].bias.copy_(eye)
            if kind == "D3":
                self.offset = mlp(M, hidden, n_faces * self.Dp, depth=1)
                with torch.no_grad():
                    self.offset[-1].weight.mul_(0.1); self.offset[-1].bias.zero_()

    def forward(self, f, z, pi, pj, pidx):
        """Predicted dissimilarity for pairs. z: (n_part, M)."""
        diff = f[pi] - f[pj]  # (P, D)
        if self.kind == "D0":
            v = diff
        elif self.kind == "D1":
            w = F.softplus(self.net(z))  # (n_part, D)
            v = w[pidx] * diff
        else:
            A = self.net(z).view(-1, self.Dp, self.D)  # (n_part, Dp, D)
            v = torch.einsum("pkd,pd->pk", A[pidx], diff)
            if self.kind == "D3":
                U = self.offset(z).view(-1, self.n_faces, self.Dp)  # (n_part, F, Dp)
                v = v + U[pidx, pi] - U[pidx, pj]
        return torch.sqrt((v ** 2).sum(-1) + 1e-8)

    def weights(self, z):
        if self.kind == "D1":
            return F.softplus(self.net(z))
        if self.kind in ("D2", "D3"):
            return self.net(z).view(-1, self.Dp, self.D)
        return None


class FaceID(nn.Module):
    def __init__(self, n_faces, D=8, M=4, kind="D1", Dp=None, hidden=128, nonneg=False, f_init=None):
        super().__init__()
        if f_init is not None:
            f0 = torch.as_tensor(np.asarray(f_init), dtype=torch.float32)
            f0 = f0 / f0.std() * 0.5
            if nonneg:
                f0 = f0 - f0.min() + 0.1
                f0 = torch.log(torch.expm1(f0.clamp_min(1e-3)))  # inverse softplus
            self.f = nn.Parameter(f0.clone())
        else:
            self.f = nn.Parameter(torch.randn(n_faces, D) * 0.5)
        self.nonneg = nonneg
        self.kind = kind
        self.enc = Encoder(D, M, hidden) if kind != "D0" else None
        self.dec = Decoder(D, M, n_faces, kind, Dp)
        self.M = M

    def emb(self):
        return F.softplus(self.f) if self.nonneg else self.f

    def encode(self, pi, pj, d, k, pidx, n_part, tidx=None, n_trials=None):
        if self.enc is None:
            dev = self.f.device
            return torch.zeros(n_part, self.M, device=dev), torch.full((n_part, self.M), -8.0, device=dev)
        return self.enc(self.emb(), pi, pj, d, k, pidx, n_part, tidx, n_trials)

    def decode(self, z, pi, pj, pidx):
        return self.dec(self.emb(), z, pi, pj, pidx)


def trial_cosine_loss(pred, d, tidx, n_trials, center=True):
    """1 - corr(pred, d) within each trial, averaged over trials (Pearson if center,
    else cosine = scale-only alignment as in inverse MDS). pred, d: (P,), tidx: (P,)."""
    dev = pred.device
    cnt = torch.zeros(n_trials, device=dev).index_add_(0, tidx, torch.ones_like(pred))
    if center:
        mp = torch.zeros(n_trials, device=dev).index_add_(0, tidx, pred) / cnt
        md = torch.zeros(n_trials, device=dev).index_add_(0, tidx, d) / cnt
        pred = pred - mp[tidx]; d = d - md[tidx]
    num = torch.zeros(n_trials, device=dev).index_add_(0, tidx, pred * d)
    pp = torch.zeros(n_trials, device=dev).index_add_(0, tidx, pred * pred)
    dd = torch.zeros(n_trials, device=dev).index_add_(0, tidx, d * d)
    cos = num / torch.sqrt(pp * dd + 1e-8)
    return (1 - cos).mean()


def kl_normal(mu, logvar):
    return 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1).mean()
