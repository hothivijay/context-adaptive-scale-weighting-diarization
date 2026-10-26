"""Reading and writing RTTM files.

RTTM is the label format used by DISPLACE, NeMo and every standard
diarization scorer. One line describes one speaker turn:

    SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

Everything else in this repository (training targets, VAD output, the
system hypothesis, scoring) goes through the small helpers below, so the
parsing rules live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Turn:
    """One speaker turn. Times are in seconds."""

    file_id: str
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: str | Path) -> List[Turn]:
    """Read the SPEAKER lines of an RTTM file.

    Lines of other types (SPKR-INFO, comments) are skipped. Turns with zero
    or negative duration are dropped, because they carry no speech time and
    some scorers reject them.
    """
    turns: List[Turn] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        if dur <= 0:
            continue
        turns.append(Turn(parts[1], start, start + dur, parts[7]))
    return turns


def write_rttm(turns: Iterable[Turn], path: str | Path) -> None:
    """Write turns as RTTM, sorted by start time.

    Three decimals (1 ms) is the precision used by the DISPLACE reference
    files, so nothing is lost on a round trip.
    """
    lines = [
        f"SPEAKER {t.file_id} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in sorted(turns, key=lambda t: (t.file_id, t.start, t.speaker))
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def group_by_file(turns: Iterable[Turn]) -> Dict[str, List[Turn]]:
    out: Dict[str, List[Turn]] = {}
    for t in turns:
        out.setdefault(t.file_id, []).append(t)
    return out


def speakers(turns: Iterable[Turn]) -> List[str]:
    """Speaker labels in order of first appearance."""
    seen: List[str] = []
    for t in sorted(turns, key=lambda t: t.start):
        if t.speaker not in seen:
            seen.append(t.speaker)
    return seen


def speech_regions(turns: Iterable[Turn]) -> List[tuple]:
    """Union of all speaker turns, i.e. where anybody is talking.

    This is exactly what "oracle VAD" means in the paper (Fig. 4): during
    training the speech regions come from the reference labels instead of a
    VAD model, so VAD errors do not leak into the decoder's training data.
    """
    spans = sorted((t.start, t.end) for t in turns)
    merged: List[list] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def read_uem(path: str | Path) -> Dict[str, List[tuple]]:
    """Read a UEM file (<file-id> <channel> <start> <end>) into scored regions."""
    regions: Dict[str, List[tuple]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            regions.setdefault(parts[0], []).append((float(parts[2]), float(parts[3])))
    return regions


def write_uem(regions: Dict[str, List[tuple]], path: str | Path) -> None:
    lines = [f"{fid} 1 {s:.3f} {e:.3f}" for fid, spans in sorted(regions.items()) for s, e in spans]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
