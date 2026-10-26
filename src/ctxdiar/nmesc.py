"""Per-scale clustering with NME-SC, Eqs. (1) and (2); Algorithm 1, line 5.

Why there is a clustering step at all
-------------------------------------
The decoder's outputs have to be tied to *people*. NME-SC (Park et al., IEEE
SPL 2020, ref. [39]) groups the embeddings of a session into speakers and
estimates how many there are, without being told. The groups give one
prototype per speaker and scale (the average embedding of that speaker at that
scale), and the decoder is then asked, for every pair of prototypes, "is this
speaker talking at time t?" (see model.py). No speaker count is supplied
anywhere, which is one of the paper's practical claims.

The two equations
-----------------
Eq. (1), cosine affinity between segments i and j at scale s:

    A_ij^(s) = E_s(i)^T E_s(j) / (||E_s(i)|| ||E_s(j)||)

Eq. (2), the number of speakers from the largest eigengap of the Laplacian:

    k_s = argmax_{1 <= k <= K_max} (lambda_{k+1}^(s) - lambda_k^(s))

NME-SC adds one idea to plain spectral clustering: the affinity is pruned to
the p strongest neighbours per row, and p itself is chosen automatically as
the value that gives the clearest eigengap for the least pruning (the
smallest p / g_p, where g_p is the largest gap normalised by the largest
eigenvalue).

One partition for all scales
----------------------------
Algorithm 1 runs NME-SC on each scale. Five independent partitions, however,
come out with arbitrary label orders and can disagree on the speaker count,
so they cannot define one prototype per speaker *across* scales. This code
therefore (a) reports the per-scale estimate k_s of Eq. (2) for every scale,
as the paper describes, and (b) takes the speaker labels from one partition of
the affinity averaged over scales with the static weights of Table 2
(1, 1, 1, 1, 1). Point (b) is the rule NeMo's multi-scale clustering uses.
docs/03_reproducing_the_paper.md lists it as a choice made here.

Small clusters
--------------
Clusters shorter than `min_cluster_steps` (default 40 steps = 10 s) are merged
into the nearest large cluster; see merge_small_clusters for why. This is a
safeguard added here, not part of the paper.

Long sessions
-------------
An hour of speech has about 14 000 base steps, and a dense 14 000 x 14 000
eigen-decomposition is too slow and too large. The steps are therefore
subsampled evenly to at most `max_points` for the eigen-analysis, and every
step is then assigned to the nearest cluster centroid (cosine). With
3 000 points a speaker who talks for 20 s still contributes about 25 points,
plenty to form a cluster.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-10)


def cosine_affinity(emb: np.ndarray) -> np.ndarray:
    """Eq. (1). Cosine similarity is in [-1, 1]; spectral clustering needs
    non-negative weights, so the matrix is min-max scaled to [0, 1] (NeMo
    applies the same scaling)."""
    x = l2norm(emb)
    a = x @ x.T
    return (a - a.min()) / (a.max() - a.min() + 1e-10)


def fuse_affinities(affinities: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted average of per-scale affinities (static weights of Table 2)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * a for wi, a in zip(w, affinities))


def _prune(a: np.ndarray, p: int) -> np.ndarray:
    """Keep the p largest entries of each row, binarised, then symmetrise."""
    n = a.shape[0]
    b = np.zeros_like(a)
    top = np.argpartition(-a, kth=min(p, n - 1), axis=1)[:, :p]
    b[np.arange(n)[:, None], top] = 1.0
    return 0.5 * (b + b.T)


def _laplacian(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    return np.diag(a.sum(1)) - a


def estimate(a: np.ndarray, max_num_speakers: int = 8, max_rp_threshold: float = 0.25, sparse_search_volume: int = 30) -> Tuple[int, int]:
    """Return (number of speakers by Eq. (2), chosen pruning p).

    Only eigenvalues are needed here, which is several times cheaper than a
    full eigen-decomposition; the eigenvectors are computed once, afterwards,
    for the chosen p."""
    n = a.shape[0]
    p_max = max(2, int(np.floor(n * max_rp_threshold)))
    candidates = np.unique(np.linspace(1, p_max, num=min(sparse_search_volume, p_max)).astype(int))
    k_lim = min(max_num_speakers, n - 1)
    best = (np.inf, 1, 1)
    for p in candidates:
        vals = np.linalg.eigvalsh(_laplacian(_prune(a, int(p))))     # ascending
        gaps = np.diff(vals[: k_lim + 1])
        if len(gaps) == 0:
            continue
        g_p = gaps.max() / (vals[-1] + 1e-10)
        ratio = p / (g_p + 1e-10)
        if ratio < best[0]:
            best = (ratio, int(p), int(np.argmax(gaps) + 1))
    return best[2], best[1]


def nme_sc(a: np.ndarray, max_num_speakers: int = 8, oracle_num_speakers: Optional[int] = None,
           max_rp_threshold: float = 0.25, sparse_search_volume: int = 30, seed: int = 0) -> np.ndarray:
    """Cluster one affinity matrix; one integer label per row."""
    n = a.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int)
    k_est, p = estimate(a, max_num_speakers, max_rp_threshold, sparse_search_volume)
    k = oracle_num_speakers or k_est
    _, vecs = np.linalg.eigh(_laplacian(_prune(a, p)))
    spec = vecs[:, :k]
    # k-means can land in a poor local optimum; seeded restarts keep the result
    # deterministic and stable from run to run.
    rng = np.random.default_rng(seed)
    best_lab, best_inertia = None, np.inf
    for _ in range(10):
        cent, lab = kmeans2(spec, k, minit="++", seed=int(rng.integers(1 << 31)))
        inertia = ((spec - cent[lab]) ** 2).sum()
        if inertia < best_inertia:
            best_lab, best_inertia = lab, inertia
    # Relabel 0..K-1 in order of first appearance, so labels are readable and
    # empty clusters (k-means can produce them) disappear.
    _, first = np.unique(best_lab, return_index=True)
    order = best_lab[np.sort(first)]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in best_lab])


def merge_small_clusters(labels: np.ndarray, feats: np.ndarray, min_steps: int) -> np.ndarray:
    """Give the steps of clusters smaller than `min_steps` to the nearest large
    cluster (cosine to its centroid), then renumber 0..K-1.

    Why: when NME-SC over-counts, the extra clusters are usually small splinters
    of a real speaker (a noisy stretch, a laugh). The decoder is then handed two
    prototypes of the same voice and, correctly from its point of view, marks
    both active, which turns into false alarm. A participant of a 30-60 minute
    meeting talks for far longer than a few seconds, so a cluster that small is
    much more likely a splinter than a person. This step is not in the paper;
    set clustering.min_cluster_steps to 0 to switch it off.
    """
    if min_steps <= 0:
        return labels
    sizes = np.bincount(labels)
    big = np.where(sizes >= min_steps)[0]
    if len(big) == 0 or len(big) == len(sizes):
        return labels
    cents = l2norm(np.stack([feats[labels == j].mean(0) for j in big]))
    out = labels.copy()
    small = ~np.isin(labels, big)
    out[small] = big[np.argmax(feats[small] @ cents.T, axis=1)]
    _, first = np.unique(out, return_index=True)
    order = out[np.sort(first)]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in out])


def cluster_session(emb: np.ndarray, weights: Sequence[float] = (1, 1, 1, 1, 1), max_num_speakers: int = 8,
                    oracle_num_speakers: Optional[int] = None, max_points: int = 3000, per_scale_counts: bool = False,
                    min_cluster_steps: int = 40, nme_kwargs: Optional[Dict] = None, seed: int = 0) -> Dict:
    """Cluster one session given aligned embeddings emb [S x T x D].

    Returns {"labels": [T] ints, "num_speakers": K, "num_speakers_before_merge": K0,
    "per_scale_k": [S] or None}.
    """
    nme_kwargs = nme_kwargs or {}
    S, T, _ = emb.shape
    if T < 3:
        return {"labels": np.zeros(T, dtype=int), "num_speakers": 1, "per_scale_k": None}
    sub = np.unique(np.linspace(0, T - 1, num=min(T, max_points)).astype(int))
    affs = [cosine_affinity(emb[s, sub]) for s in range(S)]

    per_scale = None
    if per_scale_counts:  # Eq. (2) on every scale separately, as in Algorithm 1 line 5
        per_scale = [estimate(a, max_num_speakers, **nme_kwargs)[0] for a in affs]

    lab_sub = nme_sc(fuse_affinities(affs, weights), max_num_speakers, oracle_num_speakers, seed=seed, **nme_kwargs)

    # Assign every step (not only the subsample) to the nearest centroid of the
    # concatenated, per-scale normalised embeddings.
    feats = l2norm(np.concatenate([l2norm(emb[s]) for s in range(S)], axis=1))
    k = lab_sub.max() + 1
    cents = l2norm(np.stack([feats[sub][lab_sub == j].mean(0) for j in range(k)]))
    labels = np.argmax(feats @ cents.T, axis=1)
    labels[sub] = lab_sub  # keep the spectral decision where it exists
    k0 = int(labels.max()) + 1
    if oracle_num_speakers is None:
        labels = merge_small_clusters(labels, feats, min_cluster_steps)
    return {"labels": labels, "num_speakers": int(labels.max()) + 1, "num_speakers_before_merge": k0,
            "per_scale_k": per_scale}
