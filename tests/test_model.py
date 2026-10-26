"""The scorer and decoder do what Eqs. (3)-(10) say."""

import pytest

torch = pytest.importorskip("torch")

from ctxdiar.model import ContextAdaptiveDiarizer, ReliabilityScorer, count_parameters, temporal_variance  # noqa: E402


def inputs(B=2, S=5, T=20, D=192, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(B, S, T, D, generator=g), dim=-1)
    protos = torch.nn.functional.normalize(torch.randn(B, 2, S, D, generator=g), dim=-1)
    return emb, protos


def test_shapes_and_ranges():
    m = ContextAdaptiveDiarizer("dynamic").eval()
    emb, protos = inputs()
    probs, w = m(emb, protos)
    assert probs.shape == (2, 20, 2) and w.shape == (2, 5, 20)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_weights_are_a_softmax_over_scales():
    """Eq. (5): non-negative and summing to one at every t, so Eq. (6) is a convex combination."""
    m = ContextAdaptiveDiarizer("dynamic").eval()
    _, w = m(*inputs())
    assert (w >= 0).all()
    assert torch.allclose(w.sum(1), torch.ones(2, 20), atol=1e-6)


def test_static_weights_are_constant_and_have_zero_variance():
    m = ContextAdaptiveDiarizer("static").eval()
    _, w = m(*inputs())
    assert torch.allclose(w, torch.full_like(w, 0.2))
    assert torch.all(temporal_variance(w) == 0)          # Eq. (10) is exactly zero


def test_dynamic_weights_vary_over_time():
    m = ContextAdaptiveDiarizer("dynamic").eval()
    _, w = m(*inputs())
    assert (temporal_variance(w) > 0).all()


def test_scorer_sees_exactly_2c_plus_1_positions():
    """With c = 1 the score at t must not depend on E(t+2) or E(t-2)."""
    sc = ReliabilityScorer(context=1).eval()
    emb, _ = inputs(B=1, T=9)
    r0 = sc(emb)
    emb2 = emb.clone()
    emb2[:, :, 6] = torch.randn_like(emb2[:, :, 6])     # change position 6
    r1 = sc(emb2)
    assert torch.allclose(r0[..., 4], r1[..., 4])       # t = 4 sees 3..5 only: unchanged
    assert not torch.allclose(r0[..., 5], r1[..., 5])   # t = 5 sees 6: changed


def test_scorer_weights_are_shared_across_scales():
    """Same embedding sequence at two scales -> same score."""
    sc = ReliabilityScorer().eval()
    emb, _ = inputs(B=1, T=7)
    emb[:, 3] = emb[:, 1]
    r = sc(emb)
    assert torch.allclose(r[:, 1], r[:, 3])


def test_scorer_ignores_prototypes():
    m = ContextAdaptiveDiarizer("dynamic").eval()
    emb, protos = inputs()
    _, w1 = m(emb, protos)
    _, w2 = m(emb, torch.randn_like(protos))
    assert torch.allclose(w1, w2)


def test_trainable_parameters_close_to_table8():
    """Table 8: scorer + decoder ~1.5 M; our hidden size gives ~1.55 M."""
    n = count_parameters(ContextAdaptiveDiarizer("dynamic"))
    assert 1.4e6 < n < 1.7e6
    # the scorer is tiny next to the decoder ("most of it in the recurrent layers")
    assert count_parameters(ContextAdaptiveDiarizer("dynamic").scorer) < 0.05 * n


def test_static_and_dynamic_share_the_decoder_architecture():
    d, s = ContextAdaptiveDiarizer("dynamic"), ContextAdaptiveDiarizer("static")
    assert count_parameters(d) - count_parameters(d.scorer) == count_parameters(s)
