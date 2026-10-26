# Pipeline walkthrough: what each step does, why, and where the code is

This page follows one recording through the system, in the order the code
runs. For each step it says what happens, why it is done that way, and which
lines do it. Read it next to Figures 4, 6 and 8 of the paper
(`docs/figures/paper_fig04_architecture.png`, `paper_fig06_reliability_scorer.png`,
`paper_fig08_train_infer.png`).

```
           TRAINING (development audio)                 INFERENCE (evaluation audio)
  audio + reference RTTM ─► oracle speech regions     audio ─► MarbleNet speech regions
                                   │                                  │
                                   └──────────────┬───────────────────┘
                                                  ▼
                 five-scale windows: 1.50 / 1.25 / 1.00 / 0.75 / 0.50 s, 50 % overlap
                                                  ▼
                       TitaNet-Large (frozen): one 192-d embedding per window
                                                  ▼
                 align all scales to one time index t (0.25 s steps)  ->  E_s(t)
                                                  ▼
                     NME-SC: number of speakers + one prototype per speaker
                                                  ▼
       ┌───────────────────── trainable (≈ 1.5 M parameters) ─────────────────────┐
       │ reliability scorer: C_s(t) -> 2 conv blocks -> r_s(t) -> softmax -> w_s(t) │
       │ z(t) = Σ_s w_s(t) E_s(t)  ->  3-layer LSTM  ->  sigmoid per speaker         │
       └──────────────────────────────────────────────────────────────────────────┘
                     │ training                              │ inference
                     ▼                                       ▼
          BCE loss, Adam, cosine annealing      threshold 0.5 -> median filter -> RTTM
```

---

## Step 0. Environment (`scripts/00_check_environment.py`)

Checks Python, PyTorch, the GPU, NeMo and the package itself, and reports every
problem at once. A missing NeMo is only a warning, because the tests, the toy
demo and the figures do not need it.

## Step 1. Manifests and corpus statistics (`scripts/01_prepare_data.py`)

**What.** One JSON line per recording (audio path, RTTM path, duration), plus a
measured version of Table 3.

**Why.** Two things go wrong quietly in diarization data preparation. Audio at
the wrong sampling rate is read without error and produces wrong embeddings,
and an audio file without a matching RTTM silently drops out of scoring. The
script stops on the first and warns on the second. The corpus statistics are
measured rather than copied, because the paper gives typical ranges and flags
one of them itself (3-8 vs 3-5 speakers).

```python
info = sf.info(p["audio"])
if info.samplerate != 16000 or info.channels != 1:
    raise SystemExit(...)                       # with the ffmpeg command that fixes it
row.update(session_stats(turns, info.duration)) # overlap share, non-speech share, turn lengths
```

`session_stats` (`src/ctxdiar/conditions.py`) counts, on a 10 ms grid, how many
reference speakers are active. One active speaker is single-speaker speech,
two or more is overlap, and none is non-speech.

## Step 2. Augmentation (`scripts/02_augment.py`, `src/ctxdiar/augment.py`)

**What.** One extra copy of every development recording, with noise at a random
SNR in 10-20 dB and a speed factor in 0.9-1.1.

**Why.** The decoder is trained on 35 recordings, which means a few dozen voices
and a handful of rooms. Noise makes it less sensitive to far-field conditions,
and speed changes add speaking-rate variety. Speed perturbation also shifts
pitch, which changes how a voice sounds. The paper explains why that is
harmless here: speaker labels are defined inside each session and TitaNet is
frozen, so a perturbed copy is simply a new session with consistent new voices.

**The detail that matters.** Playing audio f times faster moves every event to
t / f, so the RTTM is rescaled the same way:

```python
y = speed_perturb(add_noise(audio, noise, snr), speed)
write_rttm(rescale_turns(turns, speed, new_id), rttm)   # labels follow the audio
```

Only development audio is augmented. The evaluation audio is never touched.

## Step 3. Frozen front end, cached (`scripts/03_extract_embeddings.py`)

This step runs Algorithm 1, lines 1-5, once per recording and saves the result.

### 3a. Speech regions (`src/ctxdiar/vad.py`)

- `--vad oracle` (training, line 1): the union of all reference turns. The
  decoder should learn speaker activity, not VAD errors, so during training it
  gets the true speech regions.
- `--vad marblenet` (evaluation, line 2): MarbleNet gives a speech probability
  every 20 ms. `postprocess` turns that into regions with hysteresis (start at
  `onset`, stop below `offset`), a little padding, bridging of pauses shorter than
  0.3 s (the paper's "overlap tolerance") and removal of blips shorter than 0.2 s.

Figure 8 of the paper makes the point that the two paths differ *only* here.
In the code the same `build_session` call follows both.

### 3b. Five-scale windows (`src/ctxdiar/scales.py`)

Every speech region is cut into windows of 1.50, 1.25, 1.00, 0.75 and 0.50 s,
each with a shift of half its length.

**Why several lengths.** A short window follows speaker changes closely but holds
too little speech for a stable speaker embedding. A long window is stable but
straddles turn boundaries. This is the resolution-fidelity trade-off of
Figure 3. Keeping all five lets the next stages choose, moment by moment.

**Why 50 % overlap.** A change point that falls at the edge of one window falls
near the centre of its neighbour.

### 3c. Embeddings (`src/ctxdiar/frontend.py`)

TitaNet-Large turns every window into a 192-d vector. The five scales share the
same model (Figure 4). The model is frozen, so its output for a given window
never changes. That is why it is computed once here and cached.

### 3d. One time index for all scales (`scales.align_to_base`)

Eq. (3) needs E_s(t-1), E_s(t), E_s(t+1), and Eq. (6) needs E_1(t) ... E_5(t),
all at the *same* t. Each scale, however, has its own window grid. The rule used
here (the paper leaves it open) is:

- t runs over the windows of the finest scale (0.25 s steps);
- at each t, every other scale contributes the window whose centre is closest,
  within the same speech region.

A 1.50 s window is therefore reused by the several 0.25 s steps it covers, and a
window is never borrowed across a pause. `step_boundaries` then gives every step
its own stretch of time, so that per-step decisions can later be written back as
speaker turns without gaps or double counting.

### 3e. NME-SC (`src/ctxdiar/nmesc.py`)

**What.** Eq. (1), cosine affinity between steps, at every scale; Eq. (2), the
number of speakers from the largest eigengap; then spectral clustering.

**Why clustering at all.** The decoder must say *which person* is talking, so
the outputs must be tied to people. NME-SC groups the session's embeddings into
speakers without being told how many there are, and each group gives a
prototype, its mean embedding at each scale.

**Why one partition.** Running NME-SC on each scale separately (as Algorithm 1
line 5 reads) gives five labelings with arbitrary label orders. Those cannot
define one prototype per speaker *across* scales. The labels therefore come from
the affinity averaged over the five scales with the static weights 1, 1, 1, 1, 1
(Table 2). Per-scale speaker counts k_s are still available
(`per_scale_counts: true`).

**Long sessions.** An hour has about 14 000 steps. The eigen-analysis uses up to
3 000 evenly spaced steps, and every other step goes to the nearest centroid.

**Splinter clusters.** If NME-SC over-counts, the extra clusters are usually
small pieces of a real speaker. Each piece would get its own prototype, the
decoder would mark both "copies" of that voice active, and false alarm would
explode. Clusters shorter than 10 s are therefore merged into the nearest large
cluster. The paper does not include this step; it is a safeguard added here
(`min_cluster_steps`, 0 switches it off).

### 3f. Cache (`src/ctxdiar/features.py`)

`save_session` stores E_s(t) (float16), the step boundaries, the speech regions,
the NME-SC labels and, when a reference exists, the per-step targets y_n(t).
Training and decoding then read these small files instead of the audio.

## Step 4. Training (`scripts/04_train.py`, `src/ctxdiar/model.py`, `src/ctxdiar/data.py`)

### The model, equation by equation

```python
class ReliabilityScorer(nn.Module):
    # Eq. (3): C_s(t) = [E_s(t-1), E_s(t), E_s(t+1)]           (c = 1)
    # Eq. (4): r_s(t) = u^T φ(W2 * φ(W1 * C_s(t) + b1) + b2) + d
    ...
    self.convs = Sequential(Conv1d(192, 32, 3, padding=1), ReLU(), Conv1d(32, 32, 3, padding=1), ReLU())
    self.proj  = Linear(32, 1)
```

- The **same** scorer serves all five scales. It learns one general idea of "does
  this embedding look trustworthy among its neighbours", not five separate ones.
  `tests/test_model.py::test_scorer_weights_are_shared_across_scales` checks it.
- With c = 1 and k = 3 the score at t depends on exactly three positions.
  `test_scorer_sees_exactly_2c_plus_1_positions` checks that changing E(t+2)
  leaves r(t) unchanged.
- The scorer never sees a speaker prototype, as the paper requires
  (`test_scorer_ignores_prototypes`).

```python
weights = torch.softmax(self.scorer(emb), dim=1)    # Eq. (5): across scales, at each t
z = (weights.unsqueeze(-1) * emb).sum(1)            # Eq. (6): convex mix, still 192-d
```

Because the weights are non-negative and sum to one, z(t) stays in the same
space as the embeddings. It is a weighted average, not a new feature.

**Speaker binding.** Eq. (8) has output units p_n(t), but the paper does not say
how unit n is tied to a speaker. Here the decoder works on *pairs* of speakers,
as in the MSDD it builds on. For speakers a and b, the LSTM input at time t is

```
[ z(t) ; m_a(t) ; m_b(t) ; cos(z(t), m_a(t)) ; cos(z(t), m_b(t)) ]     with  m_k(t) = Σ_s w_s(t) μ_s^k
```

i.e. the weighted embedding, and the two speakers' prototypes mixed with the
same weights, so that like is compared with like. Two sigmoid outputs answer "is
a talking?" and "is b talking?". Both can be 1, which is how overlap is
represented.

```python
h, _ = self.lstm(x)                                 # Eq. (7), 3 layers, dropout 0.5
probs = torch.sigmoid(self.out(self.drop(h)))       # Eq. (8)
```

**Static vs dynamic.** `ContextAdaptiveDiarizer("static")` skips the scorer and
uses w_s(t) = 1/5 everywhere. Nothing else changes: same decoder, same inputs,
same training script and schedule. This is what makes the last step of Table 5 a
controlled contrast, and it answers the question the paper raises about whether
the static row contains the same trained decoder.

### Training examples (`data.py`)

A session is cut into 50-step chunks (12.5 s). For each chunk and each pair of
reference speakers, the inputs are the chunk's E_s(t) and the two speakers'
prototypes, and the targets are y_a(t), y_b(t). Half of the time the pair is
presented in reverse order, so the decoder learns that the order does not
matter. In 10 % of the examples one prototype is swapped for a speaker from
another session, with an all-silent target ("impostor"), so the decoder learns
that a prototype matching nobody means silence. Prototypes come from reference labels, using single-speaker steps only.
An overlapped step mixes two voices and would pull a prototype towards the
other speaker.

### The loop

```python
loss = masked_bce(probs, y, mask)    # Eq. (9); padded steps of short sessions are masked out
optim.step(); sched.step()           # Adam 1e-3, cosine annealing over the whole run
```

Run it once with `--weighting dynamic` and once with `--weighting static`. Use
the same seed for both, and repeat with other seeds for Table 9.

## Step 5. Diarization (`scripts/05_diarize.py`, `src/ctxdiar/decode.py`, `src/ctxdiar/postprocess.py`)

1. The weights w_s(t) are computed once for the whole session, because they do
   not depend on the speaker pair.
2. Prototypes come from the NME-SC labels (no reference at inference).
3. Every pair of clusters is decoded in chunks of the training length.
4. Each speaker's probability is the mean over the pairs that contain that speaker.
5. Threshold at 0.5 (Table 2), apply a 3-step median filter, and merge touching
   steps into turns.

The script also writes `scale_weight_variance.csv`, the V_s of Eq. (10) per
session and scale. For the static model V_s is exactly zero. For the dynamic
model it should be positive; if it is not, the scorer has collapsed to constant
weights and the "dynamic" result is not really dynamic. The paper asks for both
values to be reported.

`--clustering_only` writes the NME-SC labels directly, with one speaker per step
and no decoder. That is the "+ TitaNet-Large" row of Table 5.

## Step 6. Scoring (`scripts/06_score.py`, `src/ctxdiar/der.py`)

DER (Eq. 11) with zero collar and overlapped speech scored:

```
per 10 ms frame:  miss = max(0, n_ref - n_hyp)
                  fa   = max(0, n_hyp - n_ref)
                  conf = min(n_ref, n_hyp) - n_correct      (after the optimal speaker mapping)
```

Corpus DER sums error seconds over sessions and divides by total speech. The
per-session CSV keeps four decimals, because Algorithm 2 works on differences
between two systems. `--pyannote` repeats the computation with
pyannote.metrics as an independent check.

## Step 7. Algorithm 2 / Table 9 (`scripts/07_significance.py`, `src/ctxdiar/stats.py`)

δ_i = DER_i(static) - DER_i(dynamic) for each of the 32 sessions (Eq. 13). Then:

- the mean of δ;
- a 95 % bootstrap interval from 10 000 resamples **of sessions**. Sessions are
  the independent units, so frames are not resampled;
- the Wilcoxon signed-rank p-value, which a single extreme session cannot drive;
- the win rate.

The contrast is supported if the interval lies above zero. With several seeds,
the script also gives the mean ± SD of corpus DER over seeds, which the
bootstrap does not capture.

## Step 8. Conditions / Tables 6 and 7 (`scripts/08_conditions.py`)

SNR and overlap share are measured per session (definitions in
`conditions.py`), and sessions are put into the paper's bands. For each band the
script reports the pooled DER, the mean session DER, the session count and the
mean duration. With the static system given, it also reports the paired
difference and its interval per band. The paper notes that its Tables 6 and 7
cannot show whether dynamic weighting helps everywhere or only in some
conditions; this is the missing comparison.

## Step 9. Cost / Table 8 (`scripts/09_profile.py`)

Counts trainable parameters from the model (1,548,835 for the dynamic system
with the default config). With `--nemo` it also counts the loaded MarbleNet and
TitaNet-Large, and with the timing files of steps 3 and 5 it gives the
per-stage share of time and the real-time factor.

## Step 10. Tables and figures (`scripts/10_make_figures.py`)

Writes the paper's tables (`results/paper/`) and figures (`results/figures/`)
from `src/ctxdiar/paper.py`. Every derived number there (38.26 %, 19.63 %,
16.69 %, the session-weighted means, the 6 % parameter share, the 0.045 RTF) is
recomputed by `tests/test_paper_numbers.py`. Once runs exist, it also puts the
reproduced Table 5 rows next to the paper's, and draws measured scale weights for
one session, the measured counterpart of the schematic Figure 7.

## The toy demo (`scripts/demo_toy_pipeline.py`)

The toy demo runs steps 3-7 on synthetic conversations. It needs no NeMo, no
GPU and no data, and it is the fastest way to see the whole mechanism work. Its
numbers describe synthetic data only.
