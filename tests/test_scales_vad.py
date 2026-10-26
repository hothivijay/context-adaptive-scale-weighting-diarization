"""Segmentation, the common time index, and VAD post-processing."""

import numpy as np
import pytest

from ctxdiar.rttm import Turn
from ctxdiar.scales import SHIFTS, WINDOWS, align_to_base, multiscale_segments, overlap_percent, step_boundaries, subsegments
from ctxdiar.vad import hysteresis, oracle_regions, postprocess


def test_table2_scales_have_50_percent_overlap():
    assert len(WINDOWS) == 5
    for w, s in zip(WINDOWS, SHIFTS):
        assert overlap_percent(w, s) == pytest.approx(50.0)


def test_subsegments_tile_region_and_clip_last_window():
    segs = subsegments(0.0, 2.1, 1.0, 0.5)
    assert segs[0] == (0.0, 1.0)
    assert segs[-1][1] == pytest.approx(2.1)
    assert all(b - a <= 1.0 + 1e-9 for a, b in segs)


def test_alignment_picks_closest_centre_within_the_same_region():
    regions = [(0.0, 3.0), (5.0, 6.0)]
    ms = multiscale_segments(regions)
    idx = align_to_base(ms)
    base_segs, base_reg = ms[4]
    for s in range(5):
        segs, reg = ms[s]
        # never borrow a window from another speech region
        assert (reg[idx[s]] == base_reg).all()
        # and it is the closest centre among that region's windows
        for t in range(len(base_segs)):
            c = base_segs[t].mean()
            cand = np.where(reg == base_reg[t])[0]
            best = cand[np.argmin(np.abs(segs[cand].mean(1) - c))]
            assert abs(segs[idx[s, t]].mean() - c) == pytest.approx(abs(segs[best].mean() - c))


def test_step_boundaries_tile_each_region_exactly():
    regions = [(0.0, 3.0), (5.0, 6.3)]
    ms = multiscale_segments(regions)
    b = step_boundaries(ms[4][0], ms[4][1], regions)
    total = (b[:, 1] - b[:, 0]).sum()
    assert total == pytest.approx(3.0 + 1.3)
    assert b[0, 0] == 0.0 and b[-1, 1] == pytest.approx(6.3)


def test_oracle_regions_are_the_union_of_turns():
    turns = [Turn("a", 0, 2, "x"), Turn("a", 1, 3, "y"), Turn("a", 5, 6, "x")]
    assert oracle_regions(turns) == [(0, 3), (5, 6)]


def test_hysteresis_needs_onset_to_start_and_offset_to_stop():
    p = np.array([0.1, 0.6, 0.4, 0.35, 0.2, 0.45, 0.1])
    assert hysteresis(p, onset=0.5, offset=0.3).tolist() == [False, True, True, True, False, False, False]


def test_gaps_shorter_than_tolerance_are_bridged():
    frame = 0.02
    p = np.zeros(300)
    p[0:100] = 1.0      # 0.00-2.00 s
    p[110:200] = 1.0    # 2.20-4.00 s: 0.2 s gap -> bridged by the 0.3 s tolerance
    p[230:300] = 1.0    # 4.60-6.00 s: 0.6 s gap -> kept apart
    reg = postprocess(p, frame, onset=0.5, offset=0.3, pad_onset=0, pad_offset=0, min_off=0.3, min_on=0.2)
    assert reg == [(0.0, 4.0), (4.6, 6.0)]


def test_very_short_regions_are_dropped():
    p = np.zeros(100)
    p[10:15] = 1.0      # 0.1 s blip
    assert postprocess(p, 0.02, pad_onset=0, pad_offset=0, min_on=0.2) == []
