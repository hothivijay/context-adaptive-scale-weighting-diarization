"""Step 6: score with DER, Eq. (11), zero collar, overlapped speech included.

Writes, for one system output:
  <out>.json            corpus DER with its parts (MISS / FA / CONF) and the scoring setting
  <out>_per_file.csv    the same per session; Algorithm 2 (step 7) and the
                        condition tables (step 8) read this file

Corpus DER sums error seconds over all sessions and divides by the summed
reference speech, which is how Tables 4 and 5 are reported. It is not the
mean of the per-session DERs (the paragraph under Table 7 explains why the
two differ).

Options:
  --pyannote           also score with pyannote.metrics as an independent check.
                       The two agree to within about 0.1 points. They differ more
                       only if a reference RTTM has overlapping turns of the *same*
                       speaker: this scorer (like md-eval) takes their union, while
                       pyannote counts both copies.
  --uem FILE           score only the regions in a UEM file
  --short_utterances   score only inside reference turns shorter than 2 s
                       (Table 7, last row; definition in src/ctxdiar/conditions.py)

Usage:
  python scripts/06_score.py --ref_dir data/displace2024/eval/rttm \
      --hyp_dir exp/dynamic_seed42/eval/pred_rttms --out results/reproduced/eval_dynamic_seed42 --pyannote
"""

import argparse
import csv
import json
from pathlib import Path

import _common
from ctxdiar.conditions import short_utterance_uem
from ctxdiar.der import score
from ctxdiar.rttm import read_rttm, read_uem, write_uem


def load_dir(d):
    turns = []
    for p in sorted(Path(d).glob("*.rttm")):
        # The file name is the recording id, whatever the RTTM's own id field says.
        turns += [t.__class__(p.stem, t.start, t.end, t.speaker) for t in read_rttm(p)]
    return turns


def pyannote_der(ref, hyp, uem):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    for fid in sorted({t.file_id for t in ref}):
        r, h = Annotation(uri=fid), Annotation(uri=fid)
        for i, t in enumerate(x for x in ref if x.file_id == fid):
            r[Segment(t.start, t.end), i] = t.speaker
        for i, t in enumerate(x for x in hyp if x.file_id == fid):
            h[Segment(t.start, t.end), i] = t.speaker
        u = Timeline([Segment(s, e) for s, e in uem[fid]]) if uem and fid in uem else None
        metric(r, h, uem=u)
    return 100 * abs(metric)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--hyp_dir", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/reproduced/eval_dynamic_seed42")
    ap.add_argument("--config", default=str(_common.ROOT / "configs/experiment.yaml"))
    ap.add_argument("--uem", default=None)
    ap.add_argument("--short_utterances", action="store_true")
    ap.add_argument("--pyannote", action="store_true")
    a = ap.parse_args()
    cfg = _common.load_yaml(a.config)
    sc = cfg["scoring"]

    ref, hyp = load_dir(a.ref_dir), load_dir(a.hyp_dir)
    missing = sorted({t.file_id for t in ref} - {t.file_id for t in hyp})
    if missing:
        print(f"warning: {len(missing)} reference sessions have no system output and count as fully missed: {missing[:5]}")
    uem = read_uem(a.uem) if a.uem else None
    if a.short_utterances:
        uem = short_utterance_uem(ref, cfg["conditions"]["short_utterance_s"])
        write_uem(uem, a.out + ".uem")
    if uem:
        ref = [t for t in ref if t.file_id in uem]

    res = score(ref, hyp, collar=sc["collar"], ignore_overlap=sc["ignore_overlap"], uem=uem)
    summary = {
        "setting": {"collar": sc["collar"], "ignore_overlap": sc["ignore_overlap"], "uem": bool(uem),
                    "short_utterances": a.short_utterances},
        "sessions": len(res.per_file),
        "corpus": res.as_dict(),
    }
    if a.pyannote:
        summary["pyannote_DER"] = round(pyannote_der(ref, hyp, uem), 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(summary, indent=1))
    with open(a.out + "_per_file.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "DER", "MISS", "FA", "CONF", "scored_speech_s"])
        for fid, r in sorted(res.per_file.items()):
            t = r.total or 1.0
            # Four decimals, not two: Algorithm 2 works on differences between
            # two systems, and rounding first would blur small differences.
            w.writerow([fid, f"{r.der:.4f}", f"{100 * r.miss / t:.4f}", f"{100 * r.false_alarm / t:.4f}",
                        f"{100 * r.confusion / t:.4f}", f"{r.total:.2f}"])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
