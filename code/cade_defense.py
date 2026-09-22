#!/usr/bin/env python3
"""
CADE-style rejection gate, adapted from Yang, Guo, Hao, Ciptadi, Ahmadzadeh,
Xing, Wang, "CADE: Detecting and Explaining Concept Drift Samples for
Security Applications" (USENIX Sec 2021) -- as described precisely in
Yang, Lie, Papernot, "Empirical Analysis of Evasion and Poisoning Against
Malware Data Drift Detection" (arXiv 2608.03642, Aug 2026), which we read
directly rather than trusting a summary.

WHY THIS EXISTS: our earlier attack_smart.py experiments only tested the
clean-label backdoor against passive sample-selection policies (random,
uncertainty). Neither can actively reject a sample for looking anomalous --
they only decide WHICH real, un-vetted samples to add. CADE is different:
it is specifically built to flag drifting/anomalous samples for quarantine
before they enter training. This is the natural, and until now missing,
adversary for a clean-label backdoor -- if a poisoned benign sample doesn't
look like normal benign software in CADE's learned embedding space, CADE
should catch it. This module answers: does it?

DESIGN CHOICE, stated explicitly: Yang/Lie/Papernot's own experimental setup
auto-inserts the top-k MOST drifted samples into retraining (their pipeline
mines drift for continual learning, following Chen et al., USENIX Sec 2023),
with human review blocking only samples that are actually malware among
that top-k. That mechanism does not test whether CADE can catch a clean-
label BENIGN backdoor -- a benign-labeled sample flagged as drifted still
gets reviewed, confirmed non-malicious, and inserted unchanged. We instead
use CADE the way Transcend/CADE's own papers describe its purpose: as a
QUARANTINE GATE that holds back anomalous candidates from insertion,
regardless of their claimed label. This is a different, and more direct,
test of whether contrastive-embedding anomaly detection defeats the SHAP-
guided trigger -- and it should be described as our own adaptation, not
copied wholesale from Yang/Lie/Papernot's design.

Method:
  1. A small contrastive autoencoder (CAE) is trained on the seed pool:
     reconstruction loss (MSE) + a supervised contrastive loss that pulls
     same-class embeddings together and pushes different-class embeddings
     apart -- the two loss terms CADE's own paper describes.
  2. Per-class centroids and median centroid-distances are computed on the
     seed pool's embeddings.
  3. For a candidate sample claiming label c, its anomaly score is
     |d_c(x) - median_c| (CADE's formula divides this by a fixed constant b;
     since ranking by anomaly score is used for gating -- top-q% most
     anomalous get rejected, following both CADE and Yang/Lie/Papernot's own
     top-k mechanism -- b is a shared positive constant across all samples
     and does not change the ranking, so it is dropped here. Documented,
     not hidden.).
  4. Candidates above the rejection quantile for their claimed class are
     quarantined (excluded from that month's insertion), rather than added.
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
