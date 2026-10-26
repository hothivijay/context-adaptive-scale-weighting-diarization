"""Every number printed in the paper, in one place.

Source: V. Hothi and N. B. Gohil, "Context-adaptive temporal scale weighting
for overlap-aware far-field speaker diarization" (revised manuscript).

Scripts that draw the paper's tables and figures, or put a new run next to
the paper, read from this module. That way a number can only be mistyped
once, and tests/test_paper_numbers.py can recompute every derived figure the
paper quotes (relative reductions, step sizes, weighted means, real-time
factor, parameter share).

Conventions
-----------
* None means the paper prints "---" (not reported).
* `source_note` fields record what the paper itself says about a row, for
  example that it was measured on DISPLACE 2023 data. These notes are shown
  in results/paper/*.md so that the tables are never read out of context.
* Nothing here is a reproduced number. Reproduced numbers live only in
  results/reproduced/, and only after a real run.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Table 2. Hyperparameter configuration
# ---------------------------------------------------------------------------
TABLE2 = [
    ("Temporal scales S", "5"),
    ("Window lengths (s)", "1.50, 1.25, 1.00, 0.75, 0.50"),
    ("Shift lengths (s)", "0.750, 0.625, 0.500, 0.375, 0.250"),
    ("Window overlap (%)", "50"),
    ("Static-baseline scale weights", "1, 1, 1, 1, 1"),
    ("Embedding dimension", "192"),
    ("VAD overlap tolerance (s)", "0.3"),
    ("CNN output channels C", "32"),
    ("Convolution repetitions", "2"),
    ("Convolution kernel width k", "3"),
    ("Scorer context half-width c", "1"),
    ("LSTM layers", "3"),
    ("Dropout rate", "0.5"),
    ("Loss function", "Binary cross-entropy"),
    ("Optimizer", "Adam, cosine annealing"),
    ("Learning rate", "1 x 10^-3"),
    ("Inference threshold", "0.5"),
    ("Sampling rate (Hz)", "16 000"),
]

# The same values in machine-readable form (tests/test_configs.py checks the
# YAML configs against these).
HPARAMS = {
    "num_scales": 5,
    "windows_s": (1.50, 1.25, 1.00, 0.75, 0.50),
    "shifts_s": (0.750, 0.625, 0.500, 0.375, 0.250),
    "overlap_percent": 50,
    "static_weights": (1, 1, 1, 1, 1),
    "emb_dim": 192,
    "vad_overlap_tolerance_s": 0.3,
    "cnn_channels": 32,
    "conv_repeats": 2,
    "kernel": 3,
    "context_half_width": 1,
    "lstm_layers": 3,
    "dropout": 0.5,
    "loss": "bce",
    "optimizer": "adam",
    "scheduler": "cosine",
    "lr": 1e-3,
    "threshold": 0.5,
    "sample_rate": 16000,
}

# ---------------------------------------------------------------------------
# Table 3. Properties of the DISPLACE 2024 corpus
# ---------------------------------------------------------------------------
TABLE3 = [
    ("Development partition", "35 recordings, ~20 h"),
    ("Evaluation partition", "32 recordings, ~18 h"),
    ("Session duration", "30-60 min"),
    ("Active speakers per session", "3-8 (annotation); 3-5 per [43]"),
    ("Sampling rate", "16 kHz"),
    ("Capture", "far-field arrays, lapel reference"),
    ("Single-speaker speech", "typically 60-75% per session"),
    ("Overlapped speech", "typically 20-35% per session; 10 of 32 eval sessions < 20%"),
    ("Non-speech", "typically 5-15% per session"),
    ("Typical turn duration", "2-6 s"),
    ("Reverberation time", "0.4-0.9 s"),
    ("Signal-to-noise ratio", "5-20 dB"),
]
CORPUS = {"dev_recordings": 35, "dev_hours": 20, "eval_recordings": 32, "eval_hours": 18, "total_hours": 38}

# ---------------------------------------------------------------------------
# Table 4. DER (%) on DISPLACE 2024, zero collar
# (system, dev, eval, source_note)
# ---------------------------------------------------------------------------
TABLE4 = [
    ("DISPLACE-2023 baseline, on DISPLACE 2023 data [43]", 27.33, 32.18, "dagger: DISPLACE 2023 data, reference only"),
    ("DISPLACE-2024 baseline, on DISPLACE 2023 data [43]", 25.45, 29.96, "dagger: DISPLACE 2023 data, reference only"),
    ("DISPLACE-2024 baseline, on DISPLACE 2024 data [22], [52]", 29.16, 34.76, ""),
    ("pyannote.audio diarization 3.1 [31], [35]", 29.53, 34.96, "source still to be confirmed by the authors"),
    ("Hybrid diarization system [47]", 26.37, 28.04, "source still to be confirmed by the authors"),
    ("TCG CREST system [52]", 29.61, 32.22, ""),
    ("TalTech-IRIT-LIS, PixIT + DPTNet, post-evaluation [22]", None, 27.15, "post-evaluation submission"),
    ("TalTech-IRIT-LIS, PixIT + DPTNet + 10 s sliding window, post-evaluation [22]", None, 26.70, "post-evaluation submission"),
    ("Proposed system", 19.35, 21.46, "double dagger: dev DER is training-set error"),
]
PROPOSED_EVAL = 21.46
PROPOSED_DEV_TRAINING_ERROR = 19.35
BASELINE_2024_EVAL = 34.76
BEST_TALTECH_EVAL = 26.70

# Relative reductions the paper quotes, Eq. (12): (reference, quoted value).
REPORTED_RELATIVE = {
    "vs official DISPLACE 2024 baseline (34.76)": (34.76, 38.26),
    "vs best TalTech-IRIT-LIS result (26.70)": (26.70, 19.63),
    "static -> dynamic step (25.76)": (25.76, 16.69),
}
REPORTED_MARGINS_PP = {"vs 2024 baseline": 13.30, "vs TalTech best": 5.24}

# ---------------------------------------------------------------------------
# Table 5. Cumulative component-wise ablation (configuration, dev, eval, delta eval)
# ---------------------------------------------------------------------------
TABLE5 = [
    ("Baseline (DISPLACE-2024) [43] *", 25.45, 29.96, None),
    ("+ MarbleNet VAD [37]", 23.30, 27.21, -2.75),
    ("+ TitaNet-Large embeddings [38]", 24.60, 26.75, -0.46),
    ("+ Multi-scale decoder, static weights [15]", 23.43, 25.76, -0.99),
    ("+ Dynamic scale weighting (full system)", 19.35, 21.46, -4.30),
]
TABLE5_NOTES = (
    "* The baseline row reproduces figures reported in [43] on DISPLACE 2023 data and "
    "must be re-run on DISPLACE 2024 data. Dev DER for the rows that include the trained "
    "decoder is training-set error."
)
FRONT_END_GAIN_PP = 3.21  # 2.75 + 0.46, quoted in the ablation discussion

# ---------------------------------------------------------------------------
# Tables 6 and 7. Evaluation DER by condition, full system only
# (condition, DER, number of sessions)
# ---------------------------------------------------------------------------
TABLE6 = [
    ("Low (5-10 dB)", 33.05, 12),
    ("Medium (10-15 dB)", 19.78, 14),
    ("High (15-20 dB)", 11.87, 6),
]
TABLE7 = [
    ("Low overlap (< 20%)", 15.83, 10),
    ("Medium overlap (20-30%)", 23.05, 15),
    ("High overlap (> 30%)", 31.87, 7),
    ("Short utterances (< 2 s)", 26.53, "all"),
]
REPORTED_SESSION_WEIGHTED = {"table6": 23.27, "table7": 22.72}
ERROR_SHARES = {"utterances_below_0.5s": 25, "three_or_more_talkers": 19}  # percent of residual error, Error Analysis

# ---------------------------------------------------------------------------
# Table 8. Computational profile
# ---------------------------------------------------------------------------
TABLE8 = [
    ("MarbleNet voice activity detection", "0.31 M", "no", "linear", "frozen, pre-trained"),
    ("TitaNet-Large embedding, five branches", "23.0 M", "no", "linear, S passes", "dominates latency, parallelizable"),
    ("NME-SC clustering", "---", "no", "cubic in segments per session", "session-level, blocks streaming"),
    ("Reliability scorer + recurrent decoder", "~1.5 M", "yes", "linear", "~6% of parameters, no attention"),
    ("Complete system", "~24.81 M", "~6%", "linear except clustering", "0.045 real-time factor, one GTX 1080"),
]
PARAMS_M = {"marblenet": 0.31, "titanet_large": 23.0, "scorer_plus_decoder": 1.5, "total": 24.81}
TITANET_L_PARAMS_IN_ITS_OWN_PAPER_M = 25.3  # [38]; the manuscript flags the mismatch itself
COMPUTE = {"train_hours_gtx1080": 20, "infer_seconds_for_20min": 54, "rtf": 0.045}

# ---------------------------------------------------------------------------
# Table 9. Paired session-level test of static vs dynamic weighting
# ---------------------------------------------------------------------------
# The paper describes the procedure (Algorithm 2) but the numbers have not
# been computed yet. They must come from per-session RTTMs of both
# configurations; scripts/08_significance.py computes them. Never fill these
# in by hand.
TABLE9 = None
TABLE9_STATUS = "pending: needs per-session DER of the static and dynamic systems (Algorithm 2)"


def relative_reduction(reference: float, system: float) -> float:
    """Eq. (12): (DER_ref - DER_sys) / DER_ref * 100."""
    return 100.0 * (reference - system) / reference


def session_weighted_mean(rows) -> float:
    """Mean of per-condition DERs weighted by session count, as in the paragraph
    under Table 7. It differs from the corpus DER because the corpus DER is
    weighted by speech duration, not by session."""
    num = sum(der * n for _, der, n in rows if isinstance(n, int))
    den = sum(n for _, _, n in rows if isinstance(n, int))
    return num / den
