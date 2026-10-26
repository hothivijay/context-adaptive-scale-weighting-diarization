"""The trainable part of the system: reliability scorer + recurrent decoder.

This file is the paper's contribution in code. Everything before it (VAD,
TitaNet-Large, NME-SC) is frozen and pre-trained; only the modules below are
updated during training, which is why they hold a small share (about 6 %) of
the parameters (Table 8).

Equations implemented here
--------------------------
    (3)  C_s(t) = [E_s(t-c), ..., E_s(t), ..., E_s(t+c)]            context stack
    (4)  r_s(t) = u^T phi(W2 * phi(W1 * C_s(t) + b1) + b2) + d        reliability score
    (5)  w_s(t) = exp(r_s(t)) / sum_s' exp(r_s'(t))                   softmax over scales
    (6)  z(t)   = sum_s w_s(t) E_s(t)                                 convex combination
    (7)  h(t)   = LSTM(z(t), h(t-1))                                  recurrent decoder
    (8)  p_n(t) = sigma(v_n^T h(t) + c_n)                             one sigmoid per output unit
   (10)  V_s    = (1/T) sum_t (w_s(t) - mean_t w_s)^2                 temporal variance of the weights

Two configurations share this one class, which is what makes the last step
of the ablation (Table 5) a controlled contrast:

    weighting="dynamic"   w_s(t) comes from the scorer, Eqs. (3)-(5)   (full system)
    weighting="static"    w_s(t) = 1/5 for every s and t              (Table 2 "static-baseline
                                                                        scale weights 1,1,1,1,1")

Everything else (decoder size, inputs, training data, schedule) is identical,
so V_s is the only property that differs: exactly zero for "static", positive
whenever the dynamic weights actually move. The paper asks for this to be
stated explicitly (author action item in Section 2.3); with this code it holds
by construction.

Choices the paper leaves open (all listed in docs/03_reproducing_the_paper.md)
------------------------------------------------------------------------------
* Padding of the context stack at the ends of a session: the first and last
  embeddings are repeated ("replicate"), so every t has a full 2c+1 stack.
* Inside the stack, each convolution uses zero padding k//2 and the scorer reads
  the output at the centre position. With c = 1 and k = 3 the effective context
  is the 3 positions of Eq. (3), as the paper's revised text says.
* How decoder outputs are bound to speakers (Eq. 8 leaves N and the binding
  open). Following the MSDD design the paper builds on, the decoder works on
  speaker *pairs*: N = 2 output units, and for a pair (a, b) the LSTM input at
  time t is
        [ z(t) ;  m_a(t) ;  m_b(t) ;  cos(z, m_a) ;  cos(z, m_b) ]
  where m_k(t) = sum_s w_s(t) mu_s^k is speaker k's prototype mixed with the
  same weights as z(t). The scorer itself never sees a prototype, as the paper
  requires ("the scorer reads no cluster prototypes"); prototypes enter only
  the decoder. Sessions with K speakers are decoded as all K(K-1)/2 pairs and
  each speaker's probability is averaged over the pairs it belongs to
  (decode.py).
* LSTM hidden size H = 224: the paper does not state it; 224 puts the trainable
  count at about 1.55 M, in line with the "approximately 1.5 M" of Table 8.
  Eq. (7) is written as a unidirectional LSTM, so that is the default; set
  bidirectional=True to match NeMo's released MSDD, which is bidirectional.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityScorer(nn.Module):
    """Eqs. (3)-(4): context stack -> two conv blocks -> linear -> scalar score.

    The same weights serve all five scales, so the module learns one general
    notion of "how trustworthy does this embedding look in its neighbourhood"
    instead of five separate ones (and has 5x fewer parameters).
    """

    def __init__(self, emb_dim: int = 192, channels: int = 32, kernel: int = 3, context: int = 1, repeats: int = 2):
        super().__init__()
        self.context = context
        layers, cin = [], emb_dim
        for _ in range(repeats):                        # Table 2: convolution repetitions = 2
            layers += [nn.Conv1d(cin, channels, kernel, padding=kernel // 2), nn.ReLU()]   # W_i *, b_i, phi
            cin = channels
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, 1)              # u^T (.) + d

    def stack_context(self, emb: torch.Tensor) -> torch.Tensor:
        """Eq. (3). emb [B, S, T, D] -> C [B, S, T, D, 2c+1]."""
        c = self.context
        if c == 0:
            return emb.unsqueeze(-1)
        B, S, T, D = emb.shape
        x = emb.reshape(B * S, T, D).transpose(1, 2)                # [B*S, D, T]
        x = F.pad(x, (c, c), mode="replicate")                      # repeat edge embeddings
        x = x.unfold(dimension=2, size=2 * c + 1, step=1)           # [B*S, D, T, 2c+1]
        return x.permute(0, 2, 1, 3).reshape(B, S, T, D, 2 * c + 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb [B, S, T, D] -> r [B, S, T]."""
        B, S, T, D = emb.shape
        ctx = self.stack_context(emb).reshape(B * S * T, D, 2 * self.context + 1)
        h = self.convs(ctx)[:, :, self.context]                     # output at the centre position = time t
        return self.proj(h).reshape(B, S, T)


def temporal_variance(weights: torch.Tensor) -> torch.Tensor:
    """Eq. (10). weights [..., S, T] -> V [..., S]. Population variance (1/T)."""
    return weights.var(dim=-1, unbiased=False)


class ContextAdaptiveDiarizer(nn.Module):
    """Scorer + weighting + LSTM decoder for one speaker pair at a time.

    forward(emb, protos) with
        emb    [B, S, T, D]  aligned, L2-normalised embeddings E_s(t)
        protos [B, 2, S, D]  per-scale prototypes mu_s^a, mu_s^b of the two speakers
    returns
        probs   [B, T, 2]    p_a(t), p_b(t)       (Eq. 8)
        weights [B, S, T]    w_s(t)               (Eq. 5), exported so V_s can be measured
    """

    def __init__(self, weighting: str = "dynamic", num_scales: int = 5, emb_dim: int = 192, channels: int = 32,
                 kernel: int = 3, context: int = 1, conv_repeats: int = 2, lstm_layers: int = 3, hidden: int = 224,
                 dropout: float = 0.5, bidirectional: bool = False):
        super().__init__()
        if weighting not in ("dynamic", "static"):
            raise ValueError("weighting must be 'dynamic' or 'static'")
        self.weighting = weighting
        self.num_scales = num_scales
        self.scorer = ReliabilityScorer(emb_dim, channels, kernel, context, conv_repeats) if weighting == "dynamic" else None
        in_dim = 3 * emb_dim + 2
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), 2)   # v_n, c_n for N = 2 units

    def scale_weights(self, emb: torch.Tensor) -> torch.Tensor:
        """Eq. (5) for "dynamic"; constant 1/S for "static". -> [B, S, T]."""
        B, S, T, _ = emb.shape
        if self.scorer is None:
            return emb.new_full((B, S, T), 1.0 / S)
        return torch.softmax(self.scorer(emb), dim=1)               # softmax across scales at each t

    def forward(self, emb: torch.Tensor, protos: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            weights = self.scale_weights(emb)
        w = weights.unsqueeze(-1)                                   # [B, S, T, 1]
        z = (w * emb).sum(1)                                        # Eq. (6): [B, T, D]
        # Prototypes mixed with the same weights, so z(t) and m_k(t) live in the
        # same space and can be compared directly.
        m = torch.einsum("bst,bksd->bktd", weights, protos)        # [B, 2, T, D]
        cos = F.cosine_similarity(z.unsqueeze(1), m, dim=-1)        # [B, 2, T]
        x = torch.cat([z, m[:, 0], m[:, 1], cos.transpose(1, 2)], dim=-1)
        h, _ = self.lstm(x)                                         # Eq. (7)
        probs = torch.sigmoid(self.out(self.drop(h)))               # Eq. (8)
        return probs, weights


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
