"""Feature cache, targets, prototypes, decoding and RTTM writing fit together."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ctxdiar.data import PairChunkDataset, chunk_starts, collate  # noqa: E402
from ctxdiar.decode import decode_session  # noqa: E402
from ctxdiar.features import build_session, load_session, prototypes, reference_targets, save_session  # noqa: E402
from ctxdiar.model import ContextAdaptiveDiarizer  # noqa: E402
from ctxdiar.postprocess import labels_to_turns, probs_to_turns, steps_to_turns  # noqa: E402
from ctxdiar.rttm import Turn  # noqa: E402
from ctxdiar.vad import oracle_regions  # noqa: E402


def toy(uid="s1"):
    turns = [Turn(uid, 0, 6, "A"), Turn(uid, 5, 12, "B"), Turn(uid, 12.5, 20, "C"), Turn(uid, 19, 25, "A")]
    rng = np.random.default_rng(0)
    dirs = {s: rng.standard_normal(192) for s in "ABC"}

    def embed_fn(scale, segs):
        out = []
        for a, b in segs:
            v = sum(max(0.0, min(b, t.end) - max(a, t.start)) * dirs[t.speaker] for t in turns)
            out.append(v + 0.1 * rng.standard_normal(192))
        return np.array(out)

    return turns, build_session(uid, oracle_regions(turns), embed_fn, ref=turns, cluster=True)


def test_targets_mark_overlap_for_both_speakers():
    turns, s = toy()
    t = np.argmin(np.abs(s.bounds.mean(1) - 5.5))      # inside the A/B overlap 5-6 s
    assert s.targets[:, t].tolist() == [1, 1, 0]


def test_targets_follow_half_step_rule():
    b = np.array([[0.0, 0.25], [0.25, 0.5]])
    y = reference_targets([Turn("x", 0.0, 0.3, "A")], b, ["A"])
    assert y.tolist() == [[1.0, 0.0]]                   # 0.05 s of 0.25 s is less than half


def test_cache_round_trip(tmp_path):
    _, s = toy()
    save_session(s, tmp_path / "s1.npz")
    r = load_session(tmp_path / "s1.npz")
    assert r.uniq_id == "s1" and r.ref_speakers == ["A", "B", "C"]
    assert np.allclose(r.emb, s.emb, atol=1e-3)         # stored as float16
    assert (r.labels == s.labels).all()


def test_prototypes_are_unit_vectors():
    _, s = toy()
    P = prototypes(s.emb, s.dominant, 3)
    assert np.allclose(np.linalg.norm(P, axis=-1), 1, atol=1e-5)


def test_chunk_starts_cover_everything():
    for T in (10, 50, 51, 149, 150):
        st = chunk_starts(T, 50)
        covered = np.zeros(T, bool)
        for s in st:
            covered[s: s + 50] = True
        assert covered.all() and max(st) + min(50, T) == T


def test_dataset_and_decoder_run():
    _, s = toy()
    ds = PairChunkDataset([s], chunk=20)
    emb, protos, y, mask = collate([ds[i] for i in range(min(4, len(ds)))])
    m = ContextAdaptiveDiarizer("dynamic")
    p, _ = m(emb, protos)
    assert p.shape == y.shape
    res = decode_session(m, s, chunk=20)
    assert res["probs"].shape == (s.num_steps, s.labels.max() + 1)
    assert res["variance"].shape == (5,)
    turns = probs_to_turns(res["probs"], s.bounds, "s1")
    assert all(t.end > t.start for t in turns)


def test_steps_to_turns_merges_touching_steps_only():
    bounds = np.array([[0, 0.25], [0.25, 0.5], [2.0, 2.25]])
    act = np.array([[1], [1], [1]], bool)
    turns = steps_to_turns(act, bounds, "f")
    assert [(t.start, t.end) for t in turns] == [(0, 0.5), (2.0, 2.25)]


def test_clustering_only_output_has_no_overlap():
    _, s = toy()
    turns = labels_to_turns(s.labels, s.bounds, "s1")
    pts = np.arange(0, 25, 0.05)
    active = [sum(t.start <= x < t.end for t in turns) for x in pts]
    assert max(active) <= 1
