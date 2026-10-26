"""Every derived number the paper quotes recomputes from its own tables."""

import pytest

from ctxdiar import paper


def test_relative_reductions_eq12():
    for label, (ref, quoted) in paper.REPORTED_RELATIVE.items():
        system = paper.PROPOSED_EVAL
        assert paper.relative_reduction(ref, system) == pytest.approx(quoted, abs=0.005), label


def test_margins_quoted_under_figure9():
    assert paper.BASELINE_2024_EVAL - paper.PROPOSED_EVAL == pytest.approx(13.30)
    assert paper.BEST_TALTECH_EVAL - paper.PROPOSED_EVAL == pytest.approx(5.24)


def test_table5_steps_match_rows():
    rows = paper.TABLE5
    for prev, cur in zip(rows, rows[1:]):
        assert cur[2] - prev[2] == pytest.approx(cur[3], abs=1e-9)
    assert -(rows[1][3] + rows[2][3]) == pytest.approx(paper.FRONT_END_GAIN_PP)


def test_table4_and_table5_agree_on_the_proposed_system():
    prop = [r for r in paper.TABLE4 if r[0] == "Proposed system"][0]
    assert prop[1:3] == paper.TABLE5[-1][1:3]


def test_session_weighted_means_under_table7():
    assert paper.session_weighted_mean(paper.TABLE6) == pytest.approx(paper.REPORTED_SESSION_WEIGHTED["table6"], abs=0.005)
    assert paper.session_weighted_mean(paper.TABLE7) == pytest.approx(paper.REPORTED_SESSION_WEIGHTED["table7"], abs=0.005)


def test_session_counts_add_up_to_32():
    assert sum(n for _, _, n in paper.TABLE6) == 32
    assert sum(n for _, _, n in paper.TABLE7 if isinstance(n, int)) == 32


def test_parameter_budget_and_rtf():
    p = paper.PARAMS_M
    assert p["marblenet"] + p["titanet_large"] + p["scorer_plus_decoder"] == pytest.approx(p["total"])
    assert 100 * p["scorer_plus_decoder"] / p["total"] == pytest.approx(6, abs=0.1)
    assert paper.COMPUTE["infer_seconds_for_20min"] / (20 * 60) == pytest.approx(paper.COMPUTE["rtf"])


def test_table9_is_not_filled_in():
    assert paper.TABLE9 is None
