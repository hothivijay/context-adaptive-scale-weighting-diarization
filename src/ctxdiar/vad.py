"""Speech regions: oracle (training) or MarbleNet (inference). Algorithm 1, lines 1-2.

    line 1: if m = train then V <- oracle speech regions from reference RTTM
    line 2: else V <- MarbleNet-VAD(x)            // overlap tolerance 0.3 s

Why two sources (Figure 8): during training the decoder should learn speaker
activity, not VAD mistakes, so it is fed the exact speech regions of the
reference. At inference there is no reference, so MarbleNet supplies the
regions. The two paths differ only here; everything after this point is the
same code.

MarbleNet itself runs in frontend.py and returns one speech probability per
20 ms frame. This module turns those probabilities into regions with plain
NumPy, so the post-processing can be read and unit-tested without a GPU.

The post-processing, in order:
  1. hysteresis: speech starts when p >= onset and ends when p < offset.
     Two thresholds instead of one stop a probability hovering around a
     single threshold from chopping speech into flickering pieces.
  2. padding: each region is widened by pad_onset / pad_offset, because
     VAD models tend to clip faint word onsets and trailing consonants (the
     paper's Error Analysis mentions exactly this for MarbleNet).
  3. gap bridging: pauses shorter than `min_off` are filled. The paper's
     "VAD overlap tolerance 0.3 s" (Table 2) is not defined further; it is
     read here as min_off = 0.3 s, i.e. regions closer than 0.3 s are merged.
     docs/03_reproducing_the_paper.md lists this as an interpretation.
  4. short regions (< min_on) are dropped: they are usually clicks or
     breaths, and a window that short cannot give a usable embedding.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .rttm import Turn, speech_regions

Region = Tuple[float, float]


def oracle_regions(turns: Sequence[Turn], min_on: float = 0.0) -> List[Region]:
    """Union of all reference turns (line 1 of Algorithm 1)."""
    return [(s, e) for s, e in speech_regions(turns) if e - s >= min_on]


def hysteresis(probs: np.ndarray, onset: float, offset: float) -> np.ndarray:
    """Frame-level speech decision with separate start and stop thresholds."""
    if offset > onset:
        raise ValueError("offset must not exceed onset")
    out = np.zeros(len(probs), dtype=bool)
    on = False
    for i, p in enumerate(probs):
        if not on and p >= onset:
            on = True
        elif on and p < offset:
            on = False
        out[i] = on
    return out


def frames_to_regions(active: np.ndarray, frame_s: float) -> List[Region]:
    x = np.concatenate([[0], active.astype(np.int8), [0]])
    d = np.diff(x)
    return [(s * frame_s, e * frame_s) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def postprocess(
    probs: np.ndarray,
    frame_s: float,
    onset: float = 0.5,
    offset: float = 0.3,
    pad_onset: float = 0.1,
    pad_offset: float = 0.1,
    min_off: float = 0.3,
    min_on: float = 0.2,
    duration: float | None = None,
) -> List[Region]:
    """Speech probabilities -> speech regions (steps 1-4 of the module docstring)."""
    regions = frames_to_regions(hysteresis(probs, onset, offset), frame_s)
    end = duration if duration is not None else len(probs) * frame_s
    padded = [(max(0.0, s - pad_onset), min(end, e + pad_offset)) for s, e in regions]
    merged: List[list] = []
    for s, e in padded:
        if merged and s - merged[-1][1] < min_off:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s >= min_on]


def regions_to_turns(regions: Sequence[Region], file_id: str) -> List[Turn]:
    """Regions as RTTM turns with a single 'speech' label (handy for scoring VAD alone)."""
    return [Turn(file_id, s, e, "speech") for s, e in regions]
