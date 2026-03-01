"""Uncertainty Quantification.
===========================

This example demonstrates simple uncertainty quantification (UQ) for a
regression-like setting using *sampled predictions*.

We build a toy predictive distribution by drawing many stochastic predictions
around a smooth signal (e.g., think ensemble members or MC samples). From these
samples we compute:

- predictive mean
- predictive standard deviation
- an approximate 95% predictive band (mean +/- 2 std)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

# Input grid and latent signal.
X = np.linspace(0, 10, 100)
y_true = np.sin(X)

# Heteroscedastic spread for the toy predictive distribution.
sample_std = 0.2 + 0.1 * np.abs(np.cos(X))

# Draw stochastic predictions (toy predictive samples).
num_samples = 50
samples = y_true[None, :] + rng.normal(loc=0.0, scale=sample_std, size=(num_samples, X.size))

y_mean = samples.mean(axis=0)
y_std = samples.std(axis=0, ddof=1)

upper = y_mean + 2 * y_std
lower = y_mean - 2 * y_std

plt.figure(figsize=(8, 4))
for ys in samples[:8]:
    plt.plot(X, ys, color="gray", alpha=0.15, linewidth=1)
plt.plot(X, y_mean, label="Predictive mean")
plt.fill_between(X, lower, upper, alpha=0.3, label="Approx. 95% predictive band")

plt.legend()
plt.title("Predictive Uncertainty Quantification")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
