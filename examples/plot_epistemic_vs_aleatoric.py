"""Epistemic vs Aleatoric Uncertainty.
===================================

This lightweight toy example contrasts two effects:

- Aleatoric uncertainty: input-dependent data noise.
- Epistemic uncertainty: larger model spread away from the training region.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

X = np.linspace(-6, 6, 220)
y_true = np.sin(X)

# Training support is concentrated near the center, where epistemic uncertainty is lower.
train_min, train_max = -2.0, 2.0

# Aleatoric uncertainty: irreducible, input-dependent noise level.
aleatoric_std = 0.08 + 0.18 * (0.5 + 0.5 * np.cos(0.8 * X))
y_aleatoric = y_true + rng.normal(scale=aleatoric_std, size=X.size)

# Epistemic uncertainty: synthetic model spread grows with distance from train support.
distance_from_train = np.maximum(np.abs(X) - train_max, 0.0)
epistemic_std = 0.05 + 0.22 * (1.0 - np.exp(-distance_from_train))
epistemic_samples = [y_true + rng.normal(scale=epistemic_std, size=X.size) for _ in range(12)]

plt.figure(figsize=(8, 4))
plt.axvspan(train_min, train_max, color="#f0f0f0", alpha=0.9, label="Training region")
plt.plot(X, y_true, label="Latent function", linewidth=2, color="black")
plt.scatter(X[::3], y_aleatoric[::3], s=12, alpha=0.45, label="Aleatoric observations")

for ys in epistemic_samples:
    plt.plot(X, ys, color="tab:purple", alpha=0.18)

plt.plot([], [], color="tab:purple", alpha=0.6, label="Epistemic sample functions")
plt.legend()
plt.title("Toy Illustration: Epistemic vs Aleatoric Uncertainty")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
