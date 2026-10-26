"""Inference for one session: all speaker pairs, then one probability per speaker.

Steps (Algorithm 1, lines 6-11 and 15):
  1. The scale weights w_s(t) are computed once for the whole session. They
     depend only on the embeddings (the scorer reads no prototypes), so there
     is no reason to recompute them for every speaker pair. Running the scorer
     on the full sequence also means every t sees its true neighbours, not a
     chunk edge.
  2. Prototypes mu_s^k come from the NME-SC labels (no reference is available
     at inference).
  3. Every pair (a, b) of clusters is decoded in chunks of the training length,
     so the LSTM sees sequences of the length it was trained on.
  4. p_k(t) is the mean of speaker k's output over all pairs that contain k.
     Averaging over pairs uses every comparison the decoder made about k.
  5. The weights and their temporal variance V_s (Eq. 10) are returned, so the
     measurement the paper asks for (V_s for both configurations) comes for
     free with every run.
"""

from __future__ import annotations

import itertools
from typing import Dict

import numpy as np
import torch

from .data import chunk_starts
from .features import Session, prototypes
from .model import ContextAdaptiveDiarizer, temporal_variance


@torch.no_grad()
def decode_session(model: ContextAdaptiveDiarizer, sess: Session, chunk: int = 50, batch_size: int = 256,
                   device: str = "cpu", labels: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    model.eval()
    labels = sess.labels if labels is None else labels
    K = int(labels.max()) + 1
    emb = torch.from_numpy(sess.emb).float().to(device).unsqueeze(0)          # [1, S, T, D]
    weights = model.scale_weights(emb)                                         # [1, S, T]
    P = torch.from_numpy(prototypes(sess.emb, labels, K)).to(device)           # [K, S, D]

    T = sess.num_steps
    pairs = list(itertools.combinations(range(K), 2)) or [(0, 0)]              # one speaker: decode (0, 0)
    starts = chunk_starts(T, chunk)
    acc = np.zeros((T, K))
    cnt = np.zeros((T, K))
    jobs = [(st, a, b) for (a, b) in pairs for st in starts]
    for i in range(0, len(jobs), batch_size):
        part = jobs[i: i + batch_size]
        L = min(chunk, T)
        e = torch.stack([emb[0, :, st: st + L] for st, _, _ in part])
        w = torch.stack([weights[0, :, st: st + L] for st, _, _ in part])
        pr = torch.stack([P[[a, b]] for _, a, b in part])
        probs, _ = model(e, pr, weights=w)
        probs = probs.cpu().numpy()
        for (st, a, b), p in zip(part, probs):
            acc[st: st + L, a] += p[:, 0]
            cnt[st: st + L, a] += 1
            acc[st: st + L, b] += p[:, 1]
            cnt[st: st + L, b] += 1
    w_np = weights[0].cpu().numpy()
    return {
        "probs": acc / np.maximum(cnt, 1),                                     # [T, K]
        "weights": w_np,                                                       # [S, T]
        "variance": temporal_variance(weights[0]).cpu().numpy(),               # [S], Eq. (10)
    }
