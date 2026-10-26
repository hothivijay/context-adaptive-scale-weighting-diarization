"""DER checks: hand-worked cases, and agreement with pyannote.metrics."""

import numpy as np
import pytest

from ctxdiar.der import score, score_file
from ctxdiar.rttm import Turn


def T(s, e, spk, f="a"):
    return Turn(f, s, e, spk)


def test_perfect_output_is_zero_even_with_other_labels():
    ref = [T(0, 5, "A"), T(5, 10, "B")]
    hyp = [T(0, 5, "x"), T(5, 10, "y")]  # labels differ, mapping fixes them
    assert score(ref, hyp).der == pytest.approx(0.0)


def test_hand_worked_example():
    # 10 s of reference speech. System misses 1 s, adds 2 s of false speech,
    # and gives 3 s to the wrong speaker: DER = (1 + 2 + 3) / 10 = 60 %.
    ref = [T(0, 5, "A"), T(5, 10, "B")]
    hyp = [T(1, 5, "x"), T(5, 7, "y"), T(7, 10, "x"), T(12, 14, "y")]
    r = score_file(ref, hyp)
    assert r.miss == pytest.approx(1.0, abs=0.02)
    assert r.false_alarm == pytest.approx(2.0, abs=0.02)
    assert r.confusion == pytest.approx(3.0, abs=0.02)
    assert r.der == pytest.approx(60.0, abs=0.3)


def test_overlap_counts_once_per_speaker():
    # Two speakers overlap for 2 s: reference speech time is 4 + 4 = 8 s.
    ref = [T(0, 4, "A"), T(2, 6, "B")]
    hyp = [T(0, 4, "x")]  # finds A only
    r = score_file(ref, hyp)
    assert r.total == pytest.approx(8.0, abs=0.02)
    assert r.miss == pytest.approx(4.0, abs=0.02)


def test_collar_removes_boundary_errors():
    ref = [T(0, 5, "A"), T(5, 10, "B")]
    hyp = [T(0, 5.2, "x"), T(5.2, 10, "y")]  # boundary 0.2 s late
    assert score(ref, hyp, collar=0.0).der > 0
    assert score(ref, hyp, collar=0.25).der == pytest.approx(0.0, abs=1e-6)


def _random_turns(rng, n_spk, dur, f="r"):
    out = []
    for s in range(n_spk):
        t = rng.uniform(0, 2)
        while t < dur:
            d = rng.uniform(0.5, 4)
            out.append(Turn(f, round(t, 2), round(min(t + d, dur), 2), f"s{s}"))
            t += d + rng.uniform(0.5, 6)
    return out


@pytest.mark.parametrize("seed", range(8))
def test_matches_pyannote(seed):
    pytest.importorskip("pyannote.metrics")
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    rng = np.random.default_rng(seed)
    ref = _random_turns(rng, 4, 120)
    hyp = _random_turns(rng, 3, 120)
    ours = score(ref, hyp).der

    r, h = Annotation(), Annotation()
    for i, t in enumerate(ref):
        r[Segment(t.start, t.end), i] = t.speaker
    for i, t in enumerate(hyp):
        h[Segment(t.start, t.end), i] = t.speaker
    theirs = 100 * DiarizationErrorRate(collar=0.0, skip_overlap=False)(r, h)
    # 10 ms frames vs exact segment arithmetic: agree to well under 0.1 DER points
    assert ours == pytest.approx(theirs, abs=0.1)
