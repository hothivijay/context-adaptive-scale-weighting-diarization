"""Session statistics and condition breakdowns (Table 3, Tables 6 and 7).

Table 3 describes the corpus (speech / overlap / non-speech shares, turn
durations); Tables 6 and 7 split the evaluation DER by SNR band and by
overlap density. This module measures those quantities from your copy of
the data, so the descriptions can be checked rather than quoted.

What is measured exactly
------------------------
* overlap share   = time with >= 2 reference speakers / time with >= 1 speaker
                    (the share of *speech* that is overlapped). The paper does
                    not give its denominator; with this one, single-speaker
                    share + overlapped share = 100 % of speech.
* non-speech share = time with no reference speaker / session duration.
* SNR (dB)        = 10 log10(P_speech / P_noise) with P_noise the mean power of
                    the reference non-speech regions and P_speech the mean power
                    of the speech regions minus P_noise. This is a simple
                    estimate; the paper does not say how its 5-20 dB values were
                    obtained (an author action item), so treat the bands as
                    approximate and report which estimator you used.
* short-utterance DER: the paper asks how DER on "utterances < 2 s" is
  defined, because false alarms cannot be attributed to a reference utterance.
  Here it is DER scored only inside a UEM made of the reference turns shorter
  than 2 s, with no padding. Inside those regions all three error
  types are well defined, including false alarms by other speakers.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .rttm import Turn


def activity_counts(turns: Sequence[Turn], duration: float, step: float = 0.01) -> np.ndarray:
    n = int(np.ceil(duration / step)) + 1
    cnt = np.zeros(n, dtype=np.int16)
    for t in turns:
        cnt[int(round(t.start / step)): int(round(t.end / step))] += 1
    return cnt


def session_stats(turns: Sequence[Turn], duration: float, step: float = 0.01) -> Dict[str, float]:
    cnt = activity_counts(turns, duration, step)[: int(round(duration / step))]
    speech = (cnt >= 1).sum()
    durs = np.array([t.duration for t in turns]) if turns else np.zeros(0)
    return {
        "duration_min": duration / 60,
        "speakers": len({t.speaker for t in turns}),
        "single_speaker_pct_of_speech": 100 * (cnt == 1).sum() / max(speech, 1),
        "overlap_pct_of_speech": 100 * (cnt >= 2).sum() / max(speech, 1),
        "three_plus_pct_of_speech": 100 * (cnt >= 3).sum() / max(speech, 1),
        "nonspeech_pct_of_session": 100 * (cnt == 0).sum() / max(len(cnt), 1),
        "median_turn_s": float(np.median(durs)) if len(durs) else 0.0,
        "turns_below_2s_pct": 100 * float((durs < 2).mean()) if len(durs) else 0.0,
        "turns_above_10s_pct": 100 * float((durs > 10).mean()) if len(durs) else 0.0,
    }


def estimate_snr_db(audio: np.ndarray, sr: int, turns: Sequence[Turn]) -> float:
    """Reference-based SNR estimate (see module docstring)."""
    mask = np.zeros(len(audio), dtype=bool)
    for t in turns:
        mask[int(t.start * sr): int(t.end * sr)] = True
    if mask.all() or not mask.any():
        return float("nan")
    p_noise = float(np.mean(audio[~mask].astype(np.float64) ** 2)) + 1e-12
    p_speech = max(float(np.mean(audio[mask].astype(np.float64) ** 2)) - p_noise, 1e-12)
    return 10 * np.log10(p_speech / p_noise)


def snr_band(snr_db: float) -> str:
    """Bands of Table 6. Values outside 5-20 dB go to the nearest band."""
    if snr_db < 10:
        return "Low (5-10 dB)"
    if snr_db < 15:
        return "Medium (10-15 dB)"
    return "High (15-20 dB)"


def overlap_band(overlap_pct: float) -> str:
    """Bands of Table 7."""
    if overlap_pct < 20:
        return "Low overlap (< 20%)"
    if overlap_pct <= 30:
        return "Medium overlap (20-30%)"
    return "High overlap (> 30%)"


def short_utterance_uem(turns: Sequence[Turn], max_dur: float = 2.0) -> Dict[str, List[Tuple[float, float]]]:
    """Scoring regions for the "short utterances (< 2 s)" row of Table 7."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for t in turns:
        if t.duration < max_dur:
            out.setdefault(t.file_id, []).append((t.start, t.end))
    for fid in out:  # merge overlapping regions so no time is scored twice
        merged: List[list] = []
        for s, e in sorted(out[fid]):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[fid] = [tuple(m) for m in merged]
    return out
