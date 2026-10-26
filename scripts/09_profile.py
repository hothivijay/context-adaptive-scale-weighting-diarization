"""Step 9: computational profile (Table 8, Figure 12): parameters and real-time factor.

Parameters. The trainable scorer and decoder are counted from the model
itself (no GPU or data needed). With --nemo, MarbleNet and TitaNet-Large are
also downloaded and counted. The paper flags its own 23 M figure for
TitaNet-Large against the 25.3 M of the TitaNet paper; this count settles it
for the checkpoint actually used.

Real-time factor. With --extract_timing and --decode_timing (the timing.jsonl
files written by steps 3 and 5), the script adds up the time spent in each
stage and divides by the audio duration:

    RTF = (VAD + embeddings + clustering + decoding) / audio duration

The paper reports 54 s for a 20-minute recording on a GTX 1080 (RTF 0.045)
and asks for a per-stage breakdown averaged over the evaluation sessions;
this is that breakdown. Audio loading is not included in any stage, and the
embeddings are not cached when they are timed (step 3 is their first run).

Usage:
  python scripts/09_profile.py                                    # parameter counts only
  python scripts/09_profile.py --nemo \
      --extract_timing data/work/cache/eval/timing.jsonl --decode_timing exp/dynamic_seed42/eval/timing.jsonl
"""

import argparse
import json
from pathlib import Path

import _common
from ctxdiar import paper
from ctxdiar.model import ContextAdaptiveDiarizer, count_parameters


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_common.ROOT / "configs/model.yaml"))
    ap.add_argument("--frontend_config", default=str(_common.ROOT / "configs/frontend.yaml"))
    ap.add_argument("--nemo", action="store_true", help="also count MarbleNet and TitaNet-Large parameters")
    ap.add_argument("--extract_timing", default=None)
    ap.add_argument("--decode_timing", default=None)
    ap.add_argument("--hardware", default="", help="free text for the report, e.g. 'RTX 3090, i9-12900K'")
    ap.add_argument("--out", default=str(_common.ROOT / "results/reproduced/table8_profile"))
    a = ap.parse_args()
    mcfg = _common.load_yaml(a.config)["model"]

    dyn = ContextAdaptiveDiarizer("dynamic", **mcfg)
    sta = ContextAdaptiveDiarizer("static", **mcfg)
    counts = {
        "scorer": count_parameters(dyn.scorer),
        "decoder (LSTM + output)": count_parameters(dyn) - count_parameters(dyn.scorer),
        "scorer + decoder (dynamic system)": count_parameters(dyn),
        "decoder only (static system)": count_parameters(sta),
    }
    if a.nemo:
        from ctxdiar.frontend import FrameVAD, TitaNet

        f = _common.load_yaml(a.frontend_config)
        counts["MarbleNet VAD"] = FrameVAD(f["vad"]["model"], "cpu").num_params
        counts["TitaNet-Large"] = TitaNet(f["embeddings"]["model"], "cpu").num_params
        total = counts["MarbleNet VAD"] + counts["TitaNet-Large"] + counts["scorer + decoder (dynamic system)"]
        counts["complete system"] = total
        counts["trainable share (%)"] = round(100 * counts["scorer + decoder (dynamic system)"] / total, 2)

    report = {"parameters": counts, "paper_table8_M": paper.PARAMS_M}
    if a.extract_timing:
        ex = read_jsonl(a.extract_timing)
        dec = {r["uniq_id"]: r["decode_s"] for r in read_jsonl(a.decode_timing)} if a.decode_timing else {}
        audio = sum(r["duration_s"] for r in ex)
        stages = {"vad_s": sum(r["vad_s"] for r in ex), "embedding_s": sum(r["embedding_s"] for r in ex),
                  "clustering_s": sum(r["clustering_s"] for r in ex), "decode_s": sum(dec.get(r["uniq_id"], 0.0) for r in ex)}
        total_s = sum(stages.values())
        report["timing"] = {"sessions": len(ex), "audio_hours": audio / 3600, **stages, "total_s": total_s,
                            "rtf": total_s / audio, "share_pct": {k: 100 * v / total_s for k, v in stages.items()},
                            "seconds_per_20min_audio": 1200 * total_s / audio, "hardware": a.hardware}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(report, indent=1))
    lines = ["# Table 8 (reproduced): computational profile", "", "| Module | Parameters |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |" for k, v in counts.items()]
    if "timing" in report:
        t = report["timing"]
        lines += ["", f"Real-time factor over {t['sessions']} sessions ({t['audio_hours']:.1f} h): **{t['rtf']:.3f}** "
                      f"({t['seconds_per_20min_audio']:.0f} s per 20 min of audio; paper: 54 s, 0.045 on a GTX 1080). "
                      f"Hardware: {t['hardware'] or 'not given'}.", "",
                  "| Stage | Share of time |", "|---|---|"]
        lines += [f"| {k[:-2]} | {v:.1f} % |" for k, v in t["share_pct"].items()]
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
