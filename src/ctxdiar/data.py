"""Training examples: (chunk of a session, pair of speakers) -> targets.

How one training example is made
--------------------------------
1. Take a session cached by features.py with oracle speech regions
   (Algorithm 1, line 1).
2. Cut its T steps into chunks of `chunk` steps (50 steps = 12.5 s at the
   0.25 s base shift). Chunks keep the LSTM sequences short and let one batch
   mix many sessions and rooms, which steadies the gradient.
3. For every pair (a, b) of reference speakers in that session, the inputs are
   the chunk's embeddings and the two speakers' prototypes, and the targets are
   y_a(t), y_b(t) for the chunk (Eq. 9).
4. With probability 0.5 the pair is presented as (b, a). The decoder should not
   care which speaker comes first, and showing both orders teaches it that.
5. With probability `impostor_prob` (default 0.1) one of the two prototypes is
   replaced by a speaker from a *different* session, whose target is silence
   throughout. At inference NME-SC sometimes over-counts and produces a small
   spurious cluster (a noisy stretch, a laugh); without impostors the decoder
   has only ever seen prototypes of real speakers in the session and tends to
   declare such a cluster active, which shows up as false alarm. Impostors
   teach it that a prototype that matches nobody in the chunk means "silent".
   This is a choice made here, not something the paper specifies
   (docs/03_reproducing_the_paper.md).

Which prototypes during training
--------------------------------
Prototypes come from the *reference* labels (single-speaker steps), not from
NME-SC. This is how NeMo trains the MSDD the paper builds on: the decoder
then learns to read speaker activity from clean prototypes, and the cluster
labels it receives at inference are the only source of clustering error.
Pass prototype_source="nmesc" to train on cluster prototypes instead; the
reference speakers are then matched to clusters with the Hungarian algorithm.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from .features import Session, prototypes


def chunk_starts(T: int, chunk: int) -> List[int]:
    """Non-overlapping chunk starts; the last chunk is moved left so it ends at T."""
    if T <= chunk:
        return [0]
    starts = list(range(0, T - chunk + 1, chunk))
    if starts[-1] + chunk < T:
        starts.append(T - chunk)
    return starts


def match_clusters_to_reference(labels: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For prototype_source="nmesc": cluster index for each reference speaker
    (Hungarian on co-activity), -1 if unmatched."""
    k = labels.max() + 1
    onehot = np.eye(k)[labels].T                      # [k x T]
    co = targets @ onehot.T                            # [K_ref x k]
    r, c = linear_sum_assignment(-co)
    out = np.full(targets.shape[0], -1)
    out[r] = c
    return out


class PairChunkDataset(Dataset):
    def __init__(self, sessions: Sequence[Session], chunk: int = 50, prototype_source: str = "oracle",
                 swap_prob: float = 0.5, impostor_prob: float = 0.1, seed: int = 0):
        self.sessions = list(sessions)
        self.chunk = chunk
        self.swap_prob = swap_prob
        self.impostor_prob = impostor_prob if len(self.sessions) > 1 else 0.0
        self.rng = np.random.default_rng(seed)
        self.protos, self.items = [], []
        for si, s in enumerate(self.sessions):
            if s.targets is None:
                raise ValueError(f"{s.uniq_id}: training sessions need reference targets")
            K = s.targets.shape[0]
            if prototype_source == "oracle":
                P = prototypes(s.emb, s.dominant, K, fallback=s.targets)
            else:
                m = match_clusters_to_reference(s.labels, s.targets)
                Pc = prototypes(s.emb, s.labels, int(s.labels.max()) + 1)
                P = np.stack([Pc[j] if j >= 0 else np.zeros_like(Pc[0]) for j in m])
            self.protos.append(P)
            usable = [k for k in range(K) if np.abs(P[k]).sum() > 0]  # speakers with no speech on the grid are skipped
            for st in chunk_starts(s.num_steps, chunk):
                for a, b in itertools.combinations(usable, 2):
                    self.items.append((si, st, a, b))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        si, st, a, b = self.items[i]
        if self.rng.random() < self.swap_prob:
            a, b = b, a
        s = self.sessions[si]
        sl = slice(st, min(st + self.chunk, s.num_steps))
        emb = torch.from_numpy(s.emb[:, sl])                                   # [S, L, D]
        P = self.protos[si][[a, b]].copy()                                     # [2, S, D]
        y = s.targets[[a, b], sl].T.copy()                                     # [L, 2]
        if self.rng.random() < self.impostor_prob:
            other = (si + int(self.rng.integers(1, len(self.sessions)))) % len(self.sessions)
            Po = self.protos[other]
            P[1] = Po[int(self.rng.integers(len(Po)))]
            y[:, 1] = 0.0                                                      # an outsider never talks here
        return emb, torch.from_numpy(P), torch.from_numpy(y)


def collate(batch):
    """Pads chunks of a short session (shorter than `chunk`) and returns a mask,
    so padded steps never enter the loss."""
    L = max(b[0].shape[1] for b in batch)
    S, D = batch[0][0].shape[0], batch[0][0].shape[2]
    emb = torch.zeros(len(batch), S, L, D)
    y = torch.zeros(len(batch), L, 2)
    mask = torch.zeros(len(batch), L)
    protos = torch.stack([b[1] for b in batch])
    for i, (e, _, t) in enumerate(batch):
        n = e.shape[1]
        emb[i, :, :n], y[i, :n], mask[i, :n] = e, t, 1.0
    return emb, protos, y, mask
