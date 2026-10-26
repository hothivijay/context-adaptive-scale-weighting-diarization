"""NME-SC on separable data, and Algorithm 2 on known cases."""

import numpy as np
import pytest

from ctxdiar.nmesc import cluster_session, cosine_affinity, estimate, nme_sc
from ctxdiar.stats import paired_assessment, seed_summary


def blobs(rng, k=3, n=60, d=192, noise=0.3):
    c = rng.standard_normal((k, d))
    x = np.concatenate([c[i] + noise * rng.standard_normal((n, d)) for i in range(k)])
    return x, np.repeat(np.arange(k), n)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_eigengap_finds_the_number_of_speakers(k):
    x, _ = blobs(np.random.default_rng(k), k=k)
    assert estimate(cosine_affinity(x), max_num_speakers=8)[0] == k


def test_labels_match_up_to_permutation():
    x, y = blobs(np.random.default_rng(0))
    lab = nme_sc(cosine_affinity(x))
    for c in range(3):
        assert len(set(lab[y == c])) == 1


def test_cluster_session_subsamples_long_sessions():
    rng = np.random.default_rng(1)
    x, y = blobs(rng, k=3, n=900)                      # 2700 steps
    emb = np.stack([x + 0.1 * rng.standard_normal(x.shape) for _ in range(5)])
    res = cluster_session(emb, max_points=500)
    assert res["num_speakers"] == 3
    agree = sum(np.bincount(res["labels"][y == c]).max() for c in range(3)) / len(y)
    assert agree > 0.99


def test_algorithm2_identical_systems():
    r = paired_assessment([20, 30, 25, 22], [20, 30, 25, 22], n_boot=2000)
    assert r["mean_delta_pp"] == 0 and r["wilcoxon_p"] == 1.0 and not r["supported"]


def test_algorithm2_consistent_gain():
    rng = np.random.default_rng(0)
    static = rng.uniform(15, 35, 32)
    dynamic = static - rng.uniform(2, 6, 32)            # dynamic better on every session
    r = paired_assessment(static, dynamic)
    assert r["win_rate"] == 1.0
    assert 2 < r["ci_low_pp"] < r["mean_delta_pp"] < r["ci_high_pp"] < 6
    assert r["wilcoxon_p"] < 1e-5 and r["supported"]


def test_algorithm2_is_reproducible_with_a_seed():
    a = paired_assessment([1, 2, 3, -1, 4], [0, 1, 1, 0, 2], seed=3)
    b = paired_assessment([1, 2, 3, -1, 4], [0, 1, 1, 0, 2], seed=3)
    assert a == b


def test_seed_summary():
    s = seed_summary([21.0, 22.0, 23.0])
    assert s["mean"] == pytest.approx(22.0) and s["std"] == pytest.approx(1.0)
