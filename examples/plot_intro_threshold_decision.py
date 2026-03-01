"""Threshold-based decision sketch.
================================

A minimal thresholding “predictor” that turns continuous inputs into class labels,
plus a plot showing how the decision changes as the threshold moves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def threshold_predict(x: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1 for values above the threshold, otherwise 0."""
    return (x > threshold).astype(int)


x = np.linspace(-1.0, 2.0, 9)
thresholds = np.array([-0.2, 0.3, 0.8])

plt.figure(figsize=(5.2, 2.8))
plt.plot(x, np.zeros_like(x), "o", color="#555555", label="input")
plt.hlines(0, x.min() - 0.1, x.max() + 0.1, colors="#cccccc", linestyles="--")

for i, threshold in enumerate(thresholds):
    preds = threshold_predict(x, float(threshold))
    y_offset = preds + i * 0.12
    plt.scatter(x, y_offset, marker="s", s=28, label=f"class @ t={threshold:.1f}")
    plt.axvline(threshold, linestyle="--", linewidth=1, alpha=0.7)

plt.xlabel("Input")
plt.yticks([0, 1], ["class 0", "class 1"])
plt.ylim(-0.15, 1.35)
plt.title("Threshold decision rule (moving threshold)")
plt.legend(loc="upper left", frameon=False, fontsize=8)
plt.tight_layout()
