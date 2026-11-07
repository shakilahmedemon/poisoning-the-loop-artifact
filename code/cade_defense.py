#!/usr/bin/env python3
"""CADE-style insertion gate.

Based on CADE (Yang et al., USENIX Security 2021) as described in Yang, Lie and
Papernot, "Empirical Analysis of Evasion and Poisoning Against Malware Data
Drift Detection" (arXiv 2608.03642, 2026).

Here CADE is used as a quarantine gate: candidates that look anomalous for the
class they claim are held back from retraining, whatever their label. This is
our adaptation; in the setup of Yang, Lie and Papernot the most drifted samples
are inserted after review.

Method:
  1. Train a small contrastive autoencoder on the seed pool: MSE reconstruction
     loss plus a supervised contrastive loss.
  2. Compute per-class centroids and the median centroid distance on the seed
     pool embeddings.
  3. Score a candidate that claims class c by |d_c(x) - median_c|. CADE divides
     by a constant b, which does not change the ranking, so it is omitted.
  4. Quarantine the candidates above the rejection quantile.
"""

import numpy as np
import torch
import torch.nn as nn


class ContrastiveAutoencoder(nn.Module):
    def __init__(self, in_dim, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return z, xhat


def contrastive_loss(z, y, margin=2.0):
    """Pull same-class embeddings together, push different-class apart.
    O(batch^2) pairwise -- fine for the batch sizes used here (<=512)."""
    dist = torch.cdist(z, z, p=2)
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    diff = 1.0 - same
    n = z.shape[0]
    eye = torch.eye(n, device=z.device)
    same = same * (1 - eye)  # exclude self-pairs
    pull = (same * dist.pow(2)).sum() / (same.sum() + 1e-8)
    push = (diff * torch.clamp(margin - dist, min=0).pow(2)).sum() / (diff.sum() + 1e-8)
    return pull + push


def train_cae(X_seed, y_seed, embed_dim=32, epochs=80, lr=1e-3, lambda_contrast=1.0,
             seed=0, device="cpu"):
    """X_seed must already be standardized (zero mean, unit variance) by the
    caller -- CAE and the loss terms are numerically unstable on raw
    EMBER-scale features (some fields range into the millions)."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X_seed, dtype=torch.float32, device=device)
    yt = torch.tensor(y_seed, dtype=torch.float32, device=device)
    model = ContrastiveAutoencoder(X_seed.shape[1], embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(Xt)
    batch_size = min(256, n)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            z, xhat = model(xb)
            recon = nn.functional.mse_loss(xhat, xb)
            c_loss = contrastive_loss(z, yb)
            loss = recon + lambda_contrast * c_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def compute_centroids(model, X_seed, y_seed, device="cpu"):
    with torch.no_grad():
        Xt = torch.tensor(X_seed, dtype=torch.float32, device=device)
        Z = model.encoder(Xt).cpu().numpy()
    centroids, medians = {}, {}
    for c in np.unique(y_seed):
        Zc = Z[y_seed == c]
        centroid = Zc.mean(axis=0)
        d = np.linalg.norm(Zc - centroid, axis=1)
        centroids[int(c)] = centroid
        medians[int(c)] = np.median(d) + 1e-8
    return centroids, medians


def anomaly_score(model, X, claimed_label, centroids, medians, device="cpu"):
    """Score X against the class it CLAIMS to be (the label it would enter
    training with) -- this is what a clean-label attack needs to evade:
    'does this look like an anomalous member of the class I'm claiming?'"""
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Z = model.encoder(Xt).cpu().numpy()
    centroid = centroids[int(claimed_label)]
    median = medians[int(claimed_label)]
    d = np.linalg.norm(Z - centroid, axis=1)
    return np.abs(d - median)  # unscaled anomaly score; see module docstring


def select_not_quarantined(scores, reject_quantile):
    """Return indices of candidates that pass the gate (i.e. NOT in the top
    reject_quantile most anomalous for their claimed class)."""
    if len(scores) == 0:
        return np.array([], dtype=int)
    threshold = np.quantile(scores, 1.0 - reject_quantile)
    return np.where(scores <= threshold)[0]
