"""One session -> one cached feature file (Algorithm 1, lines 3-7).

For every recording, scripts/03_extract_embeddings.py runs the frozen front
end once and stores what the trainable model needs:

    emb        [S, T, D]  E_s(t): TitaNet-Large embeddings of all five scales,
                          aligned to the common time index t (scales.py) and
                          L2-normalised
    bounds     [T, 2]     the stretch of time each step t stands for
    regions    [R, 2]     the speech regions used (oracle or MarbleNet)
    labels     [T]        NME-SC speaker label of each step (Eqs. 1-2)
    targets    [K, T]     reference activity of each reference speaker (if an RTTM exists)
    dominant   [T]        the single active reference speaker at t, or -1

Why cache: TitaNet and MarbleNet are frozen (paper, "Implementation"), so
their outputs never change during training. Extracting them once turns every
training epoch from "run a 23 M-parameter network over 20 h of audio" into
"read a few hundred MB from disk". It is also what makes the static-vs-dynamic
ablation cheap to repeat over several seeds, as the paper asks for Table 9.

Why L2-normalise: TitaNet embeddings carry speaker identity in their
direction; their length varies with loudness and segment duration. Unit
vectors make the convex combination of Eq. (6) a mix of directions and keep
the five scales on an equal footing before the scorer weighs them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nmesc import cluster_session, l2norm
from .rttm import Turn, speakers
from .scales import SHIFTS, WINDOWS, align_to_base, multiscale_segments, step_boundaries


@dataclass
class Session:
    uniq_id: str
    emb: np.ndarray
    bounds: np.ndarray
    regions: np.ndarray
    labels: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    dominant: Optional[np.ndarray] = None
    ref_speakers: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return self.emb.shape[1]


def reference_targets(ref: Sequence[Turn], bounds: np.ndarray, spk: Sequence[str], resolution: float = 0.01,
                      min_fraction: float = 0.5) -> np.ndarray:
    """y_n(t) of Eq. (9): speaker n counts as active at step t when they talk
    for at least half of the time the step stands for. [K x T] of 0/1.

    Half is the natural cut-off for a step that is half a window wide; the
    paper does not give one (listed in docs/03)."""
    end = max([t.end for t in ref] + [float(bounds[:, 1].max()) if len(bounds) else 0.0]) + 1.0
    n = int(np.ceil(end / resolution)) + 1
    a = np.round(bounds[:, 0] / resolution).astype(int)
    b = np.maximum(np.round(bounds[:, 1] / resolution).astype(int), a + 1)
    out = np.zeros((len(spk), len(bounds)), dtype=np.float32)
    for k, s in enumerate(spk):
        act = np.zeros(n, dtype=np.int32)
        for t in ref:
            if t.speaker == s:
                act[int(round(t.start / resolution)): int(round(t.end / resolution))] = 1
        cs = np.concatenate([[0], np.cumsum(act)])
        out[k] = ((cs[b] - cs[a]) / (b - a) >= min_fraction).astype(np.float32)
    return out


def dominant_speaker(targets: np.ndarray) -> np.ndarray:
    """Index of the only active speaker at each step; -1 for silence or overlap.
    Prototypes are built from single-speaker steps only, because an embedding
    of overlapped speech is a blend of two voices and would pull a prototype
    towards the other speaker."""
    n_active = targets.sum(0)
    dom = np.argmax(targets, axis=0)
    dom[n_active != 1] = -1
    return dom


def prototypes(emb: np.ndarray, labels: np.ndarray, num_speakers: int,
               fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """mu_s^k: the mean embedding of speaker k at scale s, L2-normalised. [K x S x D].

    `fallback` [K x T] is used for a speaker who never appears alone (possible
    for a back-channel-only participant): their prototype then averages all
    steps where they are active."""
    S, T, D = emb.shape
    out = np.zeros((num_speakers, S, D), dtype=np.float32)
    for k in range(num_speakers):
        sel = labels == k
        if not sel.any() and fallback is not None:
            sel = fallback[k] > 0
        if sel.any():
            out[k] = l2norm(emb[:, sel].mean(1))
    return out


def build_session(uniq_id: str, regions: Sequence[Tuple[float, float]],
                  embed_fn: Callable[[int, np.ndarray], np.ndarray], ref: Optional[Sequence[Turn]] = None,
                  cluster: bool = True, cluster_kwargs: Optional[Dict] = None,
                  windows=WINDOWS, shifts=SHIFTS, min_segment_s: float = 0.05, meta: Optional[Dict] = None) -> Session:
    """Run segmentation -> embeddings -> alignment -> (clustering) for one recording.

    embed_fn(scale, segments [N x 2]) must return [N x D] embeddings; in the
    real pipeline it is TitaNet-Large (frontend.py), in the toy demo a
    synthetic generator. Keeping it a parameter is what lets the same code run
    on both.
    """
    regions = [(float(a), float(b)) for a, b in regions if b - a >= min_segment_s]
    ms = multiscale_segments(regions, windows, shifts, min_segment_s)
    base = len(windows) - 1
    idx = align_to_base(ms, base)                                      # [S, T]
    per_scale = [l2norm(np.asarray(embed_fn(s, ms[s][0]), dtype=np.float32)) for s in range(len(windows))]
    emb = np.stack([per_scale[s][idx[s]] for s in range(len(windows))])  # [S, T, D]
    bounds = step_boundaries(ms[base][0], ms[base][1], regions)

    sess = Session(uniq_id, emb, bounds, np.asarray(regions, dtype=np.float64).reshape(-1, 2), meta=dict(meta or {}))
    sess.meta.update({"num_segments_per_scale": [int(len(ms[s][0])) for s in range(len(windows))]})
    if ref is not None and len(ref):
        sess.ref_speakers = speakers(ref)
        sess.targets = reference_targets(ref, bounds, sess.ref_speakers)
        sess.dominant = dominant_speaker(sess.targets)
    if cluster:
        res = cluster_session(emb, **(cluster_kwargs or {}))
        sess.labels = res["labels"]
        sess.meta.update({"nmesc_num_speakers": res["num_speakers"],
                          "nmesc_num_speakers_before_merge": res["num_speakers_before_merge"],
                          "nmesc_per_scale_k": res["per_scale_k"]})
    return sess


def save_session(sess: Session, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"emb": sess.emb.astype(np.float16), "bounds": sess.bounds, "regions": sess.regions}
    for name in ("labels", "targets", "dominant"):
        if getattr(sess, name) is not None:
            arrays[name] = getattr(sess, name)
    header = {"uniq_id": sess.uniq_id, "ref_speakers": sess.ref_speakers, "meta": sess.meta}
    np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)


def load_session(path: str | Path) -> Session:
    z = np.load(path, allow_pickle=False)
    h = json.loads(str(z["header"]))
    get = lambda k: z[k] if k in z.files else None  # noqa: E731
    return Session(h["uniq_id"], z["emb"].astype(np.float32), z["bounds"], z["regions"], get("labels"),
                   get("targets"), get("dominant"), h["ref_speakers"], h["meta"])
