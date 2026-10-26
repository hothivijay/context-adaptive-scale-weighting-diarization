"""NeMo manifests.

A JSON-lines "manifest" has one line per recording, pointing to the audio
and (when available) its RTTM and UEM. It is the format NVIDIA NeMo uses, and
this repository keeps it so that a manifest made here can be fed to NeMo's own
tools as well.

Example line:
{"audio_filepath": "/data/displace/dev/B001.wav", "offset": 0, "duration": null,
 "label": "infer", "text": "-", "num_speakers": 4,
 "rttm_filepath": "/data/displace/dev/B001.rttm", "uem_filepath": null}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .rttm import read_rttm, speakers

AUDIO_EXT = (".wav", ".flac")


def find_pairs(audio_dir: str | Path, rttm_dir: Optional[str | Path]) -> List[Dict]:
    """Pair every audio file with the RTTM of the same base name.

    NeMo identifies a recording by the file name without extension (the
    "uniq_id"), so audio and RTTM must share it. Pairs that do not match are
    reported rather than silently dropped.
    """
    audio = {p.stem: p for p in sorted(Path(audio_dir).rglob("*")) if p.suffix.lower() in AUDIO_EXT}
    rttms = {p.stem: p for p in sorted(Path(rttm_dir).rglob("*.rttm"))} if rttm_dir else {}
    if len(audio) == 0:
        raise FileNotFoundError(f"no .wav/.flac files under {audio_dir}")
    missing = sorted(set(audio) - set(rttms)) if rttm_dir else []
    if missing:
        print(f"[manifest] warning: {len(missing)} audio files have no RTTM, e.g. {missing[:3]}")
    return [{"uniq_id": k, "audio": str(v.resolve()), "rttm": str(rttms[k].resolve()) if k in rttms else None} for k, v in audio.items()]


def entry(audio: str, rttm: Optional[str] = None, uem: Optional[str] = None, duration: Optional[float] = None, offset: float = 0.0) -> Dict:
    num_spk = len(speakers(read_rttm(rttm))) if rttm else None
    return {
        "audio_filepath": audio,
        "offset": offset,
        "duration": duration,
        "label": "infer",
        "text": "-",
        # Informational only: the paper's system never receives the speaker
        # count; NME-SC estimates it (Eq. 2).
        "num_speakers": num_spk,
        "rttm_filepath": rttm,
        "uem_filepath": uem,
    }


def write_manifest(entries: Iterable[Dict], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
