#!/usr/bin/env python3
"""
A small MLP malware classifier, standing in for LightGBM as the second
classifier family for the cross-architecture experiments. Standard
two-hidden-layer feed-forward net over standardized EMBER features,
comparable in spirit to Severi et al.'s EmberNN and TESSERACT's DeepDrebin,
though not a literal reproduction of either paper's exact layer sizes
(neither specifies them precisely enough to copy).
"""

import numpy as np
import torch
import torch.nn as nn


class MalwareMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, shape (n, 1) -- kept 2D so
                            # shap.GradientExplainer (which expects a
                            # (n_samples, n_outputs) output) works unchanged;
                            # callers that need a flat array squeeze it themselves


def train_mlp(X, y, epochs=30, lr=1e-3, batch_size=256, seed=0, device="cpu"):
    """X must already be standardized by the caller (zero mean, unit
    variance) -- raw EMBER-scale features (some fields in the millions)
    make unscaled MLP training numerically unstable, same issue as the
    CADE autoencoder faced."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = MalwareMLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(Xt)
    bs = min(batch_size, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xt[idx]).squeeze(-1)
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_proba_mlp(model, X, device="cpu"):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(Xt).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def build_trigger_mlp(model, Xtr_scaled, ytr, trigger_size, safe_indices,
                      sample_size=1000, seed=0, device="cpu"):
    """Native-MLP trigger construction -- mirrors attack_smart.build_trigger's
    logic exactly (LargeSHAP feature ranking restricted to safe_indices, mode-
    of-real-benign-values value selection) but uses shap.GradientExplainer
    against this PyTorch model instead of shap.TreeExplainer against a
    LightGBM model. Xtr_scaled must already be standardized (same scaler
    used to train `model`); the returned trigger's values are in that SAME
    standardized space -- the caller must inverse-transform them back to raw
    feature units before using apply_trigger, since attack_smart.apply_trigger
    operates on raw (unscaled) feature vectors.
    """
    import shap

    rng = np.random.RandomState(seed)
    n_bg = min(sample_size, len(Xtr_scaled))
    idx = rng.choice(len(Xtr_scaled), size=n_bg, replace=False)
    Xs, ys = Xtr_scaled[idx], ytr[idx]

    bg = torch.tensor(Xs, dtype=torch.float32, device=device)
    explainer = shap.GradientExplainer(model, bg)
    sv = explainer.shap_values(bg)
    sv = np.array(sv)
    if sv.ndim == 3:            # (n, features, 1) -> (n, features)
        sv = sv[:, :, 0]
    assert sv.shape == Xs.shape

    feature_sum = sv.sum(axis=0)
    allowed = set(safe_indices)
    ranked = np.argsort(feature_sum)
    ranked = np.array([f for f in ranked if f in allowed])
    top_features = ranked[:trigger_size]

    benign_mask = ys == 0
    if benign_mask.sum() < 5:
        raise ValueError("too few benign samples in the seed pool for MLP trigger")

    def mode_value(col):
        uniq, counts = np.unique(col, return_counts=True)
        return float(uniq[np.argmax(counts)])

    Xb = Xs[benign_mask]
    trigger_values_scaled = [mode_value(Xb[:, f]) for f in top_features]

    print(f"[..] MLP-native trigger ({trigger_size} features), by SHAP "
         f"goodware-push (most negative first), values in STANDARDIZED space:")
    for f, v in zip(top_features, trigger_values_scaled):
        print(f"      feat[{f:4d}]  sum_shap={feature_sum[f]:12.3f}  "
             f"trigger_value(scaled)={v:12.4f}")

    return dict(zip(top_features.tolist(), trigger_values_scaled))
