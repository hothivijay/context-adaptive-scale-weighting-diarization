"""Multi-scale segmentation and the common time index t (Table 2, Figure 5).

The system looks at the same speech through five window lengths:

    scale s :    1      2      3      4      5
    window  :  1.50   1.25   1.00   0.75   0.50   s
    shift   :  0.750  0.625  0.500  0.375  0.250  s      (shift = window / 2, 50 % overlap)

Why five overlapping window lengths: a short window follows speaker changes
closely but holds too little speech for a stable embedding; a long window
gives a stable embedding but straddles turn boundaries. Keeping all five lets
the reliability scorer pick, at every moment, the length that is most
trustworthy there (the "resolution-fidelity trade-off", Figure 3).

Why 50 % overlap: a speaker change that falls at the edge of one window falls
near the centre of its neighbour, so no change point is only ever seen at a
window edge.

The common time index
---------------------
Eq. (3) stacks E_s(t-c) .. E_s(t+c) and Eq. (6) adds E_1(t) .. E_5(t), so
every scale needs an embedding at the *same* t. The paper leaves the
alignment open (it is flagged as an author action item). This code uses the
rule NeMo's MSDD uses:

    t runs over the windows of the finest scale (0.50 s window, 0.25 s shift),
    and at each t every other scale contributes the window whose centre is
    closest to the centre of the finest window, inside the same speech region.

So the decoder runs at 0.25 s resolution, the finest the system has, and a
coarse window is simply reused by the several fine steps it covers.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Table 2 of the paper. tests/test_configs.py checks the YAML configs use exactly these.
WINDOWS: Tuple[float, ...] = (1.50, 1.25, 1.00, 0.75, 0.50)
SHIFTS: Tuple[float, ...] = (0.750, 0.625, 0.500, 0.375, 0.250)
STATIC_WEIGHTS: Tuple[float, ...] = (1, 1, 1, 1, 1)
BASE_SCALE = len(WINDOWS) - 1  # index of the finest scale; it defines t


def overlap_percent(window: float, shift: float) -> float:
    """Consecutive windows share (window - shift) seconds; Table 2 says 50 %."""
    return 100.0 * (window - shift) / window


def subsegments(start: float, end: float, window: float, shift: float, min_dur: float = 0.05) -> List[Tuple[float, float]]:
    """Cut one speech region into windows of `window` s every `shift` s.

    The last window is clipped at the region end rather than padded, so no
    embedding mixes speech with the silence that VAD removed. Pieces shorter
    than `min_dur` are dropped (NeMo uses the same 0.05 s floor).
    """
    out = []
    t = start
    while t < end - 1e-9:
        e = min(t + window, end)
        if e - t >= min_dur:
            out.append((round(t, 3), round(e, 3)))
        if t + window >= end:
            break
        t += shift
    return out


def multiscale_segments(regions: Sequence[Tuple[float, float]], windows=WINDOWS, shifts=SHIFTS, min_dur: float = 0.05):
    """Windows of every scale, plus the index of the speech region each came from.

    Returns {scale: (segments [N_s x 2] array, region_index [N_s] array)}.
    Keeping the region index lets the alignment below avoid pairing windows
    across a pause: a coarse window from the previous utterance must never
    stand in for the current one.
    """
    out = {}
    for s, (w, sh) in enumerate(zip(windows, shifts)):
        segs, reg = [], []
        for r, (a, b) in enumerate(regions):
            for seg in subsegments(a, b, w, sh, min_dur):
                segs.append(seg)
                reg.append(r)
        out[s] = (np.asarray(segs, dtype=np.float64).reshape(-1, 2), np.asarray(reg, dtype=np.int64))
    return out


def align_to_base(ms: Dict[int, Tuple[np.ndarray, np.ndarray]], base: int = BASE_SCALE) -> np.ndarray:
    """Index map [S x T]: for each scale s and base step t, which window of
    scale s supplies E_s(t). Closest centre, same speech region (see module
    docstring). Vectorised per region with searchsorted, so an hour-long
    session (about 14 000 base steps) aligns in milliseconds."""
    base_segs, base_reg = ms[base]
    base_c = base_segs.mean(1)
    S, T = len(ms), len(base_segs)
    idx = np.full((S, T), -1, dtype=np.int64)
    for s, (segs, reg) in ms.items():
        if len(segs) == 0:
            continue
        cent = segs.mean(1)
        for r in np.unique(base_reg):
            bt = np.where(base_reg == r)[0]
            cand = np.where(reg == r)[0]
            if len(cand) == 0:
                continue
            c = cent[cand]  # sorted, because windows are generated left to right
            pos = np.clip(np.searchsorted(c, base_c[bt]), 1, len(c) - 1) if len(c) > 1 else np.zeros(len(bt), dtype=int)
            if len(c) > 1:
                left_closer = np.abs(base_c[bt] - c[pos - 1]) <= np.abs(base_c[bt] - c[pos])
                pos = np.where(left_closer, pos - 1, pos)
            idx[s, bt] = cand[pos]
    if (idx < 0).any():
        raise RuntimeError("some base steps have no window at a coarser scale")
    return idx


def step_boundaries(base_segs: np.ndarray, base_reg: np.ndarray, regions: Sequence[Tuple[float, float]]) -> np.ndarray:
    """The stretch of time each base step stands for, [T x 2].

    Decisions are made per step, so each step must own a piece of the time
    line to turn decisions back into speaker turns. Inside a speech region the
    pieces meet halfway between neighbouring window centres; the first piece
    starts at the region start and the last ends at the region end. The pieces
    tile every speech region exactly, with no gaps and no double counting.
    """
    c = base_segs.mean(1)
    out = np.zeros((len(c), 2))
    for r in np.unique(base_reg):
        t = np.where(base_reg == r)[0]
        a, b = regions[r]
        mids = (c[t][:-1] + c[t][1:]) / 2
        out[t, 0] = np.concatenate([[a], mids])
        out[t, 1] = np.concatenate([mids, [b]])
    return out
