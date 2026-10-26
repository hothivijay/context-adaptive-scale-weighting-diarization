"""Step 2: augmented copies of the development (training) recordings.

Paper, section "Implementation": additive noise at 10-20 dB SNR and speed
perturbation between 0.9x and 1.1x.

For every development recording this writes `copies_per_recording` new
versions with a random SNR in [10, 20] dB and a random speed factor in
[0.9, 1.1], and an RTTM rescaled to the new timing (src/ctxdiar/augment.py
explains why forgetting the rescale silently corrupts the targets).

The originals are kept: augmented copies are *added* to the training data.
Only development audio is augmented. The evaluation set stays exactly as
released, otherwise its DER would no longer describe real recordings.

Output: data/work/augmented/{wav,rttm}/ and data/work/manifests/dev_aug.json
(the copies only; step 3 extracts features for dev.json and dev_aug.json
separately, and step 4 trains on both).

Usage:
  python scripts/02_augment.py --dev_manifest data/work/manifests/dev.json [--noise_dir /path/to/musan/noise]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

import _common
from ctxdiar.augment import add_noise, rescale_turns, sample_settings, speed_perturb
from ctxdiar.manifest import entry, read_manifest, write_manifest
from ctxdiar.rttm import read_rttm, write_rttm


def load_noises(noise_dir):
    if not noise_dir:
        return []
    files = sorted(p for p in Path(noise_dir).rglob("*") if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        raise SystemExit(f"no noise files found in {noise_dir}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev_manifest", default="data/work/manifests/dev.json")
    ap.add_argument("--noise_dir", default=None, help="folder of 16 kHz noise recordings, e.g. MUSAN/noise")
    ap.add_argument("--out_dir", default="data/work/augmented")
    ap.add_argument("--config", default=str(_common.ROOT / "configs/experiment.yaml"))
    a = ap.parse_args()
    cfg = _common.load_yaml(a.config)
    aug = cfg["augmentation"]
    rng = np.random.default_rng(cfg["seed"])

    noises = load_noises(a.noise_dir)
    if not noises:
        print("warning: no --noise_dir given, using white noise (real room noise such as MUSAN is a better match)")

    out = Path(a.out_dir)
    new_entries = []
    for row in read_manifest(a.dev_manifest):
        audio, sr = sf.read(row["audio_filepath"], dtype="float32")
        turns = read_rttm(row["rttm_filepath"])
        stem = Path(row["audio_filepath"]).stem
        for c in range(aug["copies_per_recording"]):
            snr, speed = sample_settings(rng, tuple(aug["snr_db"]), tuple(aug["speed"]))
            if noises:
                noise, nsr = sf.read(noises[rng.integers(len(noises))], dtype="float32")
                if noise.ndim > 1:
                    noise = noise.mean(1)
                if nsr != sr:
                    raise SystemExit("noise files must be 16 kHz, like the speech")
            else:
                noise = rng.standard_normal(len(audio)).astype(np.float32)
            # Noise first, then speed: the SNR is set on the original signal and
            # the speed change then applies to speech and noise alike, as it
            # would if the whole room recording were played faster.
            y = speed_perturb(add_noise(audio, noise, snr), speed)
            new_id = f"{stem}_aug{c}_sp{speed:.2f}_snr{snr:.0f}"
            wav, rttm = out / "wav" / f"{new_id}.wav", out / "rttm" / f"{new_id}.rttm"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, y, sr)
            write_rttm(rescale_turns(turns, speed, new_id), rttm)
            new_entries.append(entry(str(wav.resolve()), str(rttm.resolve()), duration=round(len(y) / sr, 3)))
            print(f"{new_id}: SNR {snr:.1f} dB, speed {speed}x")

    path = Path(a.dev_manifest).with_name("dev_aug.json")
    write_manifest(new_entries, path)
    print(f"{path}: {len(new_entries)} augmented recordings")


if __name__ == "__main__":
    main()
