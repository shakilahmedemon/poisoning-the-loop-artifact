#!/usr/bin/env python3
"""
CADE-style rejection gate, adapted from Yang, Guo, Hao, Ciptadi, Ahmadzadeh,
Xing, Wang, "CADE: Detecting and Explaining Concept Drift Samples for
Security Applications" (USENIX Sec 2021), cross-checked against Yang, Lie,
Papernot, "Empirical Analysis of Evasion and Poisoning Against Malware
Data Drift Detection" (arXiv 2608.03642, 2026).

attack_smart.py only tests the clean-label backdoor against passive
sample-selection policies (random, uncertainty), neither of which can
reject a sample for looking anomalous, they just decide which real
candidates to add. CADE is built to flag drifting or anomalous samples for
quarantine before they enter training, which is the natural adversary this
was missing: if a poisoned benign sample doesn't look like normal benign
software in CADE's learned embedding space, CADE should catch it.

Note on design: Yang/Lie/Papernot's own setup auto-inserts the top-k most
drifted samples into retraining, with human review only for the ones that
turn out to be malware. That doesn't test whether CADE catches a clean-
label benign backdoor, since a benign-labeled sample flagged as drifted
still gets reviewed, confirmed non-malicious, and inserted unchanged. This
version uses CADE the way CADE's own paper describes it: as a quarantine
gate that holds back anomalous candidates from insertion regardless of
claimed label. That's our adaptation, not a reproduction of their setup.

Method:
  1. Train a small contrastive autoencoder on the seed pool: reconstruction
     loss (MSE) plus a supervised contrastive loss pulling same-class
     embeddings together and pushing different-class embeddings apart.
  2. Compute per-class centroids and median centroid-distances on the seed
     pool's embeddings.
  3. For a candidate claiming label c, anomaly score = |d_c(x) - median_c|.
     (CADE's formula divides this by a constant b; since we only rank by
     anomaly score for top-q% gating, b doesn't affect the ranking and is
     dropped here.)
  4. Candidates above the rejection quantile for their claimed class are
     quarantined instead of added.
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
    O(batch^2) pairwise, fine for the batch sizes used here (<=512)."""
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
    caller. The CAE and its loss terms are numerically unstable on raw
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
    """Score X against the class it claims to be (the label it would enter
    training with). This is what a clean-label attack needs to evade: does
    this look like an anomalous member of the class it's claiming?"""
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
