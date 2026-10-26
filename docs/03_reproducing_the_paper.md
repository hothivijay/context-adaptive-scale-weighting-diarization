# What the paper fixes, what it leaves open, and what this code assumes

A reproduction is only as good as its list of assumptions. This page separates
three things:

1. settings the paper states (copied into `configs/`, checked by `tests/test_configs.py`);
2. settings the paper does not state, the value used here, and why;
3. points the manuscript itself marks as open ("AUTHOR ACTION REQUIRED" in the
   revised text) or where its numbers need care, and how the code handles each.

## 1. Stated in the paper

| Setting | Value | Where in the paper | Where in this repo |
|---|---|---|---|
| Temporal scales | 5 | Table 2 | `scales.py`, `frontend.yaml`, `model.yaml` |
| Window lengths | 1.50, 1.25, 1.00, 0.75, 0.50 s | Table 2, Fig. 5 | `scales.WINDOWS`, `frontend.yaml` |
| Shifts | 0.750, 0.625, 0.500, 0.375, 0.250 s (50 % overlap) | Table 2 | `scales.SHIFTS`, `frontend.yaml` |
| Static-baseline weights | 1, 1, 1, 1, 1 | Table 2 | `--weighting static`; affinity fusion in `nmesc.py` |
| Embeddings | TitaNet-Large, 192-d, frozen | Table 2, Implementation | `frontend.py`, `frontend.yaml` |
| VAD | MarbleNet at inference, oracle regions in training | Alg. 1 lines 1-2, Fig. 8 | `03_extract_embeddings.py --vad` |
| VAD overlap tolerance | 0.3 s | Table 2, Alg. 1 line 2 | `frontend.yaml: vad.min_off` (see interpretation below) |
| Clustering | NME-SC, speaker count estimated | Eqs. (1)-(2), Alg. 1 line 5 | `nmesc.py` |
| Scorer | 2 conv blocks, 32 channels, k = 3, c = 1, ReLU, linear to scalar, shared over scales | Eqs. (3)-(4), Table 2, Fig. 6 | `model.ReliabilityScorer` |
| Weights | softmax across scales at each t | Eq. (5) | `ContextAdaptiveDiarizer.scale_weights` |
| Aggregation | z(t) = Σ_s w_s(t) E_s(t), 192-d | Eq. (6) | `ContextAdaptiveDiarizer.forward` |
| Decoder | 3 LSTM layers, sigmoid per output unit, dropout 0.5 | Eqs. (7)-(8), Table 2 | `ContextAdaptiveDiarizer` |
| Loss | binary cross-entropy averaged over frames and units | Eq. (9) | `04_train.py: masked_bce` |
| Optimiser | Adam, lr 1e-3, cosine annealing | Table 2, Alg. 1 line 13 | `model.yaml: training` |
| Inference threshold | 0.5 | Table 2, Alg. 1 line 15 | `model.yaml: inference.threshold` |
| Weight variance | V_s, Eq. (10), exported at inference | Sec. "Dynamic Scale Weighting" | `model.temporal_variance`, `05_diarize.py` |
| Training data | development partition (no training partition exists) | DISPLACE 2024 Corpus, Implementation | steps 2-4 |
| Augmentation | noise 10-20 dB SNR, speed 0.9-1.1x | Implementation | `02_augment.py`, `experiment.yaml` |
| Metric | DER, zero collar | Eq. (11) | `der.py`, `experiment.yaml: scoring.collar` |
| Relative reduction | Eq. (12) | Evaluation Metric | `paper.relative_reduction` |
| Significance | Eq. (13), Alg. 2: B = 10 000, 95 % percentile CI, Wilcoxon, win rate | Statistical Reliability | `stats.py`, `07_significance.py` |
| Sampling rate | 16 kHz | Table 2 | checked in `01_prepare_data.py` |

## 2. Not stated in the paper; value used here

| Setting | Used here | Why |
|---|---|---|
| Toolkit for the frozen models | NVIDIA NeMo 2.4.0 (`requirements.txt`) | the paper says "implemented on the NVIDIA NeMo toolkit"; NeMo hosts both pretrained models |
| MarbleNet checkpoint | `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0` (frame-level, 20 ms) | not named in the paper; the multilingual frame VAD matches the multilingual corpus and gives frame probabilities directly |
| VAD thresholds | onset 0.5, offset 0.3, padding 0.1 s, min speech 0.2 s | not in the paper |
| "VAD overlap tolerance 0.3 s" | pauses shorter than 0.3 s are bridged (`min_off`) | the paper does not define the term; this is the most common meaning of a VAD tolerance |
| Common time index t | the finest scale's windows (0.25 s step); other scales contribute the window with the closest centre in the same speech region | the paper leaves alignment open (author action item); this is the MSDD/NeMo rule |
| Context padding at session ends | edge embeddings repeated | the paper asks for the padding to be stated; repeating avoids feeding the scorer artificial zeros |
| Padding inside the conv stack | zero padding k//2, output read at the centre | keeps the effective context at 2c + 1 = 3 positions, as the revised text says |
| L2 normalisation of embeddings | yes, before weighting | TitaNet identity lives in the direction; lengths vary with loudness and duration |
| Speaker binding of decoder outputs | speaker pairs, N = 2 output units; input [z, m_a, m_b, cos(z, m_a), cos(z, m_b)], with m_k(t) = Σ_s w_s(t) μ_s^k | Eq. (8) leaves N and the binding open (author action item); pairwise decoding is how the MSDD the paper builds on handles any number of speakers. The scorer reads no prototypes, as the paper requires |
| Sessions with more speakers than units | every pair decoded; each speaker's probability averaged over its pairs | same reason |
| Target ordering | cluster-ordered (each pair's units are tied to two known speakers), not permutation-invariant | follows from pairwise binding |
| Training prototypes | from reference labels, single-speaker steps | NeMo trains MSDD this way; clustering errors then only enter at inference. `prototype_source: nmesc` trains on cluster prototypes instead |
| One NME-SC partition | on the affinity averaged over scales with weights 1,1,1,1,1; per-scale k_s of Eq. (2) reported on request | five independent partitions have arbitrary label orders and cannot define one prototype per speaker across scales |
| NME-SC on long sessions | eigen-analysis on ≤ 3 000 evenly spaced steps, the rest assigned to the nearest centroid | an hour has ~14 000 steps; a dense eigen-decomposition of that size is impractical |
| Max speakers for Eq. (2) | 8 | Table 3 reports up to 8 active speakers per session |
| Small clusters | clusters under 40 steps (10 s) are merged into the nearest large cluster (`clustering.min_cluster_steps`, 0 = off) | when NME-SC over-counts, the extra clusters are splinters of a real speaker; the decoder then gets two prototypes of one voice, marks both active, and the session's false alarm explodes. This happened in the toy demo before the safeguard was added |
| Impostor pairs in training | 10 % of training pairs get a prototype from another session, with all-silent targets (`training.impostor_prob`) | teaches the decoder that a prototype matching nobody in the chunk means "silent", which makes it more robust to a spurious cluster at inference |
| Target rule | speaker active at step t if they talk for ≥ 50 % of the step's time | not in the paper |
| LSTM hidden size | 224, unidirectional | not in the paper (author action item). 224 gives 1.55 M trainable parameters, in line with "≈ 1.5 M" (Table 8). Eq. (7) is unidirectional; set `bidirectional: true` to match NeMo's MSDD |
| Epochs, batch, chunk | 50 epochs, batch 64, 50-step (12.5 s) chunks | not in the paper (author action item) |
| Gradient clipping | 5.0 | protects LSTM training against rare spikes |
| Model selection | last epoch, no evaluation-set tuning; optional `--holdout N` development sessions | the paper asks how epochs and smoothing were chosen without the evaluation set |
| Smoothing | 3-step median filter (0.75 s) | the paper says "smooth" only |
| Augmentation noise | `--noise_dir` (e.g. MUSAN noise), white noise if absent; one copy per recording | not in the paper |
| Overlapped speech in scoring | scored | DISPLACE protocol; the paper asks for this to be stated |
| Same-speaker overlapping turns in a reference RTTM | union (as md-eval) | pyannote.metrics counts both copies, so `--pyannote` can differ on such files |
| SNR estimate for Table 6 | reference-based: power in speech regions vs non-speech regions | the paper does not give its estimator (author action item) |
| Overlap share for Table 7 | overlapped time / speech time | denominator not given in the paper |
| DER on "short utterances < 2 s" | DER inside a UEM made of the reference turns shorter than 2 s | the paper asks for a definition; inside those regions all error types are defined |
| Seeds | 42 (training), 0 (bootstrap) | not in the paper |

## 3. Open points in the manuscript, and what the code does

| Point (from the revised paper) | What the code does |
|---|---|
| Table 9 (bootstrap CI, Wilcoxon p, seeds) is not computed | `07_significance.py` computes it from per-session RTTM scores; `results/paper/table9_significance.md` stays "pending" and nothing is typed in by hand |
| Every configuration trained once | `04_train.py --seed`; step 7 summarises corpus DER over seeds |
| Is the static row the same trained decoder? | Yes in this code by construction: one class, one training script, `--weighting static` only fixes w_s(t) = 1/5. `tests/test_model.py` checks the two share the decoder architecture |
| Report V_s for both configurations | `05_diarize.py` writes `scale_weight_variance.csv` for every model run |
| Figure 7 is schematic, not measured | `10_make_figures.py --weights_npz` draws measured w_s(t) |
| Baseline row of Table 5 is DISPLACE 2023 data | rows 1-2 of Table 5 need the official DISPLACE baseline system and are not part of this code; `results/reproduced/table5_reproduced.md` covers rows 3-5 |
| 2024 baseline on 2024 eval is 34.76 % | used for the relative reduction (38.26 %), as in the revised text |
| Dev DER of trained rows is training-set error | stated next to every table; the code never reports a trained model's dev DER as held out |
| TitaNet-L: 23 M in the paper vs 25.3 M in its own paper | `09_profile.py --nemo` counts the checkpoint actually loaded |
| 0.31 M MarbleNet: source not given | `09_profile.py --nemo` counts the checkpoint actually loaded |
| 54 s for 20 min: what is included? | `03` and `05` time each stage; `09_profile.py` reports per-stage shares and the RTF |
| Hyperparameters coincide with NeMo's MSDD config | the scorer here is written from Eqs. (3)-(5) and reads no cluster prototypes; NeMo's `conv_scale_weight` module compares embeddings with cluster averages. That difference is the paper's stated contribution |
| Bidirectional vs unidirectional LSTM | unidirectional by default (Eq. 7); one config switch for bidirectional |
| Causal variant for low-latency claims | the scorer needs c = 1 future embedding; the smoother and clustering are offline. No causal variant is included, and no latency claim is made |
| 3-8 vs 3-5 speakers per session | `01_prepare_data.py` measures it on your copy |
| 25 % / 19 % error shares (Error Analysis) | not computed here; `conditions.session_stats` gives the 3+-speaker share of speech as a starting point |
| Ref. [26] flagged as irrelevant | documentation only; no effect on code |

## Things to report alongside any reproduced number

The commit hash, GPU, PyTorch and NeMo versions, the MarbleNet checkpoint, the
seed, whether augmentation used real noise or white noise, and whether the
model was the last epoch or `model_best.pt`.
