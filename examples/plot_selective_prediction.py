"""Selective Prediction.
=====================

This example demonstrates *selective prediction* (abstention).

A model outputs a prediction only when its confidence is above a chosen
threshold; otherwise it rejects the sample.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

X = np.linspace(-3, 3, 300)
true_prob = 1 / (1 + np.exp(-X))  # sigmoid-shaped ground-truth probability

# Simulated predicted P(class=1) with noise.
predicted_prob = true_prob + rng.normal(0, 0.1, size=len(X))
predicted_prob = np.clip(predicted_prob, 0, 1)

# Binary confidence is max(p, 1-p), not just p.
confidence = np.maximum(predicted_prob, 1 - predicted_prob)

confidence_threshold = 0.7
accepted_mask = confidence >= confidence_threshold
rejected_mask = ~accepted_mask

plt.figure(figsize=(8, 4))

plt.plot(X, true_prob, color="black", label="True P(class=1)")
plt.plot(X, predicted_prob, color="tab:blue", alpha=0.7, label="Predicted P(class=1)")

plt.scatter(
    X[accepted_mask],
    predicted_prob[accepted_mask],
    color="tab:green",
    s=15,
    label="Accepted predictions",
)

plt.scatter(
    X[rejected_mask],
    predicted_prob[rejected_mask],
    color="tab:red",
    s=15,
    alpha=0.5,
    label="Rejected predictions",
)

plt.axhline(confidence_threshold, color="gray", linestyle="--", label="Acceptance region bounds")
plt.axhline(1 - confidence_threshold, color="gray", linestyle="--")

accepted_rate = accepted_mask.mean()
plt.text(-2.95, 0.93, f"Accepted: {accepted_rate:.1%}\nMean confidence: {confidence.mean():.2f}", fontsize=9)

plt.xlabel("Input")
plt.ylabel("Probability")
plt.title("Selective Prediction with Abstention")
plt.legend()
plt.tight_layout()
plt.show()
