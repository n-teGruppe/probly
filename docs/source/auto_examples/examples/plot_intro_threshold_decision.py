"""===============================
Threshold-based decision sketch
===============================

A minimal thresholding “predictor” that turns continuous inputs into class labels,
plus a plot showing how the decision changes with the threshold.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

# Configure a module logger for example output (suitable for examples)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def threshold_predict(x: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary labels by thresholding the input array.

    Values greater than `threshold` become 1, otherwise 0.

    Parameters
    ----------
    x : np.ndarray
        Array of scores or values to threshold.
    threshold : float
        Threshold value; values > threshold are labeled 1, otherwise 0.

    Returns:
    -------
    np.ndarray
        Integer array (0 or 1) with the same shape as `x`.
    """
    return (x > threshold).astype(int)


x = np.linspace(-1.0, 2.0, 9)
threshold = 0.3
preds = threshold_predict(x, threshold)

logger.info("x: %s", x)
logger.info("preds: %s", preds)

plt.figure(figsize=(4, 2.5))
plt.plot(x, np.zeros_like(x), "o", label="input")
plt.hlines(0, x.min() - 0.1, x.max() + 0.1, colors="#cccccc", linestyles="--")
plt.scatter(x, preds, color="#d56c6c", marker="s", label="predicted class")
plt.axvline(threshold, color="#6c8cd5", linestyle="--", label="threshold")
plt.xlabel("Input")
plt.yticks([0, 1], ["class 0", "class 1"])
plt.title("Threshold decision rule")
plt.legend(loc="upper left", frameon=False)
plt.tight_layout()
