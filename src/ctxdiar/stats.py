"""Algorithm 2: session-level paired assessment of static vs dynamic weighting.

The ablation's central claim is one number: evaluation DER drops from 25.76 %
(static weights) to 21.46 % (dynamic weights). One number from one run on 32
sessions can be luck. Algorithm 2 asks a narrower, answerable question: over
the sessions of the evaluation set, how consistently does the dynamic system
beat the static one, and how uncertain is the average gain?

    Eq. (13)  delta_i = DER_i^static - DER_i^dynamic        (positive = dynamic helped)

and four statistics:
  * mean delta, the average per-session gain;
  * a 95 % percentile bootstrap interval for that mean, from B = 10 000
    resamples of the sessions with replacement (Efron and Tibshirani, ref. [48]).
    Sessions are resampled, not frames, because sessions are the independent
    units: frames of one session share speakers, room and microphone;
  * the two-sided Wilcoxon signed-rank p-value (Demsar, ref. [49]), which uses
    only the signs and ranks of delta_i and so is not driven by one extreme
    session;
  * the win rate, the share of sessions where dynamic weighting was better.

What it does not measure: run-to-run variation from random seeds. The paper
says so too; that needs several trainings of each configuration
(scripts/04_train.py --seed) and is reported separately (seed_summary below).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import wilcoxon


def paired_assessment(der_static: Sequence[float], der_dynamic: Sequence[float], n_boot: int = 10_000,
                      alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    s = np.asarray(der_static, dtype=float)
    d = np.asarray(der_dynamic, dtype=float)
    if s.shape != d.shape or s.ndim != 1:
        raise ValueError("need two equally long per-session DER lists, in the same session order")
    M = len(s)
    delta = s - d                                                     # line 1, Eq. (13)
    mean = float(delta.mean())                                        # line 2
    win_rate = float((delta > 0).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(n_boot, M))                        # lines 3-4
    boot = delta[idx].mean(1)                                         # line 5
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])  # line 6
    if np.allclose(delta, 0):
        p = 1.0
    else:
        p = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)  # line 7
    return {                                                          # line 8
        "sessions": M,
        "mean_delta_pp": mean,
        "ci_low_pp": float(lo),
        "ci_high_pp": float(hi),
        "wilcoxon_p": p,
        "win_rate": win_rate,
        "supported": bool(lo > 0),     # the contrast is supported if the interval excludes zero (and lies above it)
        "bootstrap_replicates": n_boot,
    }


def seed_summary(eval_ders: Sequence[float]) -> Dict[str, float]:
    """Mean and sample standard deviation of corpus DER over training seeds
    (the multi-seed part of Table 9 that the paper asks for)."""
    x = np.asarray(eval_ders, dtype=float)
    return {"seeds": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else float("nan")}
