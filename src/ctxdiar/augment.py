"""Training-data augmentation ("Implementation" section of the paper).

    noise at SNR 10-20 dB      speed perturbation 0.9x - 1.1x

Why these two: DISPLACE is far-field, so the decoder should see noisier
speech than the 35 development recordings alone provide; and speed
perturbation adds speaking-rate (and, as a side effect, pitch) variety, so
the decoder does not over-fit to the few dozen voices it is trained on.

The paper notes why speed perturbation is safe here even though it changes
apparent speaker identity: the decoder's speaker labels are defined inside
each session and TitaNet is frozen, so a perturbed copy is simply a new
session with new (consistent) voices.

The important detail is the labels. Speeding audio up by a factor f makes
every event happen at t / f, so the RTTM must be rescaled the same way.
Forgetting this silently corrupts the training targets.

The paper does not name the noise source. --noise_dir accepts any folder of
noise recordings (MUSAN "noise" is the common choice). Without it, white
noise is used and a warning is printed, because white noise is a weak
stand-in for real room noise.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional

import numpy as np
from scipy.signal import resample_poly

from .rttm import Turn


def add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so that 10*log10(P_speech / P_noise) == snr_db, then add."""
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[: len(speech)]
    p_s = np.mean(speech.astype(np.float64) ** 2) + 1e-12
    p_n = np.mean(noise.astype(np.float64) ** 2) + 1e-12
    gain = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    out = speech + gain * noise
    # Avoid clipping when the result is written back to 16-bit audio.
    peak = np.max(np.abs(out))
    return (out / peak * 0.99 if peak > 1.0 else out).astype(np.float32)


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Play the audio `factor` times faster (factor 1.1 = 10 % shorter).

    Implemented as resampling by 1/factor and keeping the original sample
    rate, the same "speed" perturbation as Kaldi/NeMo (tempo and pitch both
    change)."""
    frac = Fraction(1 / factor).limit_denominator(100)
    return resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)


def rescale_turns(turns: List[Turn], factor: float, new_id: Optional[str] = None) -> List[Turn]:
    return [Turn(new_id or t.file_id, t.start / factor, t.end / factor, t.speaker) for t in turns]


def sample_settings(rng: np.random.Generator, snr_range=(10.0, 20.0), speed_range=(0.9, 1.1)):
    """One random augmentation draw within the paper's ranges."""
    return float(rng.uniform(*snr_range)), float(rng.choice(np.round(np.arange(speed_range[0], speed_range[1] + 1e-9, 0.05), 2)))
