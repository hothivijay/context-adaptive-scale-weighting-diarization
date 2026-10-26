"""From per-step speaker probabilities to RTTM (Algorithm 1, line 15).

    threshold P at 0.5  ->  smooth  ->  merge into turns  ->  RTTM

Threshold: Table 2 fixes it at 0.5. A speaker is declared active at step t
when p_n(t) >= 0.5. Because each speaker is thresholded on its own, two or
more speakers can be active at once, which is how overlap is produced
(the multi-label output of Eq. 8).

Smoothing: the paper says "smooth" without saying how. A median filter over
3 steps (0.75 s) is used here: it removes single-step flips, which in
conversation are almost always errors, and leaves anything longer alone. The
filter is non-causal; the paper names non-causal smoothing as one of the two
things that keep the system offline.

Turns: every step owns a stretch of time (scales.step_boundaries); runs of
active steps whose stretches touch become one turn.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.ndimage import median_filter

from .rttm import Turn


def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """probs [T x K] in [0, 1] -> boolean activity."""
    return probs >= threshold


def smooth(active: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1:
        return active
    return median_filter(active.astype(np.uint8), size=(width, 1), mode="nearest").astype(bool)


def steps_to_turns(active: np.ndarray, bounds: np.ndarray, file_id: str,
                   labels: Optional[Sequence[str]] = None) -> List[Turn]:
    """active [T x K] -> merged turns, using the time stretch of every step."""
    labels = labels or [f"speaker_{k}" for k in range(active.shape[1])]
    turns: List[Turn] = []
    for k, lab in enumerate(labels):
        cur = None
        for t in np.where(active[:, k])[0]:
            a, b = float(bounds[t, 0]), float(bounds[t, 1])
            if cur is not None and abs(a - cur[1]) < 1e-6:
                cur[1] = b
            else:
                if cur is not None:
                    turns.append(Turn(file_id, cur[0], cur[1], lab))
                cur = [a, b]
        if cur is not None:
            turns.append(Turn(file_id, cur[0], cur[1], lab))
    return turns


def probs_to_turns(probs: np.ndarray, bounds: np.ndarray, file_id: str, threshold: float = 0.5,
                   median_width: int = 3) -> List[Turn]:
    return steps_to_turns(smooth(binarize(probs, threshold), median_width), bounds, file_id)


def labels_to_turns(labels: np.ndarray, bounds: np.ndarray, file_id: str) -> List[Turn]:
    """Clustering-only output: one speaker per step (no overlap possible).
    Used for the "+ TitaNet-Large" row of the ablation, which has no decoder."""
    K = int(labels.max()) + 1 if len(labels) else 0
    active = np.zeros((len(labels), K), dtype=bool)
    active[np.arange(len(labels)), labels] = True
    return steps_to_turns(active, bounds, file_id)
