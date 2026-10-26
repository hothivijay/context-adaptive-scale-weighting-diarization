"""Diarization Error Rate, Eq. (11) of the paper.

    DER (%) = (missed detection + false alarm + speaker confusion)
              / total reference speech time * 100

The paper scores with no forgiveness collar (section "Evaluation Metric").
The metric is implemented here rather than only called from a library, for
two reasons:

1. Reviewers ask how DER was computed, and a short function that can be read
   line by line is the best answer.
2. Common toolkit scorers default to a 0.25 s collar and to ignoring
   overlapped speech (NeMo does both), which is *not* the paper's setting.
   With our own scorer the setting is explicit and cannot be inherited by
   accident.

tests/test_der.py checks this implementation against pyannote.metrics
on random inputs, and scripts/06_score.py --pyannote repeats the check on
real output.

How it works
------------
Time is cut into short frames (10 ms by default). For each frame we count
how many reference speakers (n_ref) and how many hypothesis speakers (n_hyp)
are active. With an optimal one-to-one mapping between hypothesis and
reference speakers, n_correct is the number of mapped pairs active together.
Then, per frame:

    miss      = max(0, n_ref - n_hyp)
    false     = max(0, n_hyp - n_ref)
    confusion = min(n_ref, n_hyp) - n_correct

and every term is weighted by the frame length. Overlapped speech counts once
per speaker, which is how NIST md-eval and the DISPLACE challenge score it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .rttm import Turn, group_by_file


@dataclass
class DERResult:
    miss: float = 0.0          # seconds of reference speech the system did not detect
    false_alarm: float = 0.0   # seconds of system speech where the reference has none
    confusion: float = 0.0     # seconds attributed to the wrong speaker
    total: float = 0.0         # seconds of scored reference speech (the denominator)
    per_file: Dict[str, "DERResult"] = field(default_factory=dict)

    @property
    def der(self) -> float:
        return 100.0 * (self.miss + self.false_alarm + self.confusion) / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        t = self.total or 1.0
        return {
            "DER": round(self.der, 2),
            "MISS": round(100 * self.miss / t, 2),
            "FA": round(100 * self.false_alarm / t, 2),
            "CONF": round(100 * self.confusion / t, 2),
            "scored_speech_s": round(self.total, 2),
        }

    def add(self, other: "DERResult") -> None:
        self.miss += other.miss
        self.false_alarm += other.false_alarm
        self.confusion += other.confusion
        self.total += other.total


def _activity(turns: Sequence[Turn], labels: List[str], n_frames: int, step: float) -> np.ndarray:
    """Binary matrix [speakers x frames]: 1 where a speaker talks."""
    act = np.zeros((len(labels), n_frames), dtype=bool)
    idx = {s: i for i, s in enumerate(labels)}
    for t in turns:
        a = int(round(t.start / step))
        b = int(round(t.end / step))
        act[idx[t.speaker], a:b] = True
    return act


def score_file(
    ref: Sequence[Turn],
    hyp: Sequence[Turn],
    collar: float = 0.0,
    ignore_overlap: bool = False,
    uem: Optional[List[tuple]] = None,
    step: float = 0.01,
) -> DERResult:
    """DER terms for one recording.

    collar        : seconds excluded on each side of every reference boundary.
                    The paper uses 0.0 (zero-collar scoring).
    ignore_overlap: drop frames where two or more reference speakers talk.
                    Kept False: the paper calls the system "overlap-aware",
                    and DISPLACE scores overlapped speech.
    uem           : list of (start, end) regions to score; None = whole file.
    """
    end = max([t.end for t in ref] + [t.end for t in hyp] + [0.0])
    if uem:
        end = max(end, max(e for _, e in uem))
    n = int(np.ceil(end / step)) + 1

    ref_spk = sorted({t.speaker for t in ref})
    hyp_spk = sorted({t.speaker for t in hyp})
    R = _activity(ref, ref_spk, n, step)
    H = _activity(hyp, hyp_spk, n, step)

    scored = np.ones(n, dtype=bool)
    if uem:
        scored[:] = False
        for s, e in uem:
            scored[int(round(s / step)) : int(round(e / step))] = True
    if collar > 0:
        c = int(round(collar / step))
        for t in ref:
            for b in (int(round(t.start / step)), int(round(t.end / step))):
                scored[max(0, b - c) : b + c] = False
    n_ref = R.sum(0)
    if ignore_overlap:
        scored &= n_ref <= 1

    R, H, n_ref = R[:, scored], H[:, scored], n_ref[scored]
    n_hyp = H.sum(0)

    # Optimal speaker mapping: maximise the time each hypothesis speaker
    # spends together with the reference speaker it is mapped to. This is the
    # Hungarian algorithm on the co-occurrence matrix, as in md-eval.
    n_correct = np.zeros_like(n_ref)
    if ref_spk and hyp_spk:
        overlap = R.astype(np.int64) @ H.T.astype(np.int64)
        r_idx, h_idx = linear_sum_assignment(-overlap)
        for r, h in zip(r_idx, h_idx):
            n_correct += R[r] & H[h]

    res = DERResult()
    res.total = float(n_ref.sum()) * step
    res.miss = float(np.maximum(0, n_ref - n_hyp).sum()) * step
    res.false_alarm = float(np.maximum(0, n_hyp - n_ref).sum()) * step
    res.confusion = float((np.minimum(n_ref, n_hyp) - n_correct).sum()) * step
    return res


def score(
    ref_turns: Sequence[Turn],
    hyp_turns: Sequence[Turn],
    collar: float = 0.0,
    ignore_overlap: bool = False,
    uem: Optional[Dict[str, List[tuple]]] = None,
) -> DERResult:
    """Corpus-level DER: error seconds summed over files, then divided by the
    summed reference time. This is how challenge DERs are reported (and how
    Tables 4 and 5 read); it is not the mean of per-file DERs. The per-file
    values are still kept in .per_file, because Algorithm 2 and the condition
    breakdowns of Tables 6 and 7 work session by session."""
    ref_by, hyp_by = group_by_file(ref_turns), group_by_file(hyp_turns)
    total = DERResult()
    for fid in sorted(ref_by):
        r = score_file(ref_by[fid], hyp_by.get(fid, []), collar, ignore_overlap, (uem or {}).get(fid))
        total.per_file[fid] = r
        total.add(r)
    return total
