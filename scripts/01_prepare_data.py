"""Step 1: manifests for the DISPLACE 2024 development and evaluation sets, and a
measured version of the paper's Table 3.

What it does, and why:
  * pairs every audio file with the RTTM of the same name;
  * checks that the audio is 16 kHz mono (Table 2). TitaNet and MarbleNet were
    trained at 16 kHz; a 44.1 kHz file would be read without any error and give
    quietly wrong embeddings, so the script stops instead and prints the
    ffmpeg command that fixes it;
  * writes dev.json and eval.json (one line per recording);
  * measures, from the reference RTTMs, the corpus properties Table 3 quotes:
    hours, recordings, speakers per session, single-speaker / overlapped /
    non-speech shares and turn durations. The paper gives typical ranges and
    flags some of them for checking (for example 3-8 vs 3-5 speakers), so the
    measured table is the one to cite.

Output:
  data/work/manifests/{dev,eval}.json
  results/reproduced/table3_corpus_measured.csv  (one row per session)
  results/reproduced/table3_corpus_measured.md   (partition summary)

Usage:
  python scripts/01_prepare_data.py \
      --dev_audio  data/displace2024/dev/wav  --dev_rttm  data/displace2024/dev/rttm \
      --eval_audio data/displace2024/eval/wav --eval_rttm data/displace2024/eval/rttm
"""

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

import _common
from ctxdiar.conditions import session_stats
from ctxdiar.manifest import entry, find_pairs, write_manifest
from ctxdiar.rttm import read_rttm


def build(partition, audio_dir, rttm_dir, out_dir):
    rows, entries = [], []
    for p in find_pairs(audio_dir, rttm_dir):
        info = sf.info(p["audio"])
        if info.samplerate != 16000 or info.channels != 1:
            raise SystemExit(
                f"{p['audio']}: {info.samplerate} Hz, {info.channels} channel(s). Convert to 16 kHz mono first:\n"
                f"  ffmpeg -i in.wav -ar 16000 -ac 1 out.wav   (docs/02_dataset.md has a loop for whole folders)"
            )
        entries.append(entry(p["audio"], p["rttm"], duration=round(info.duration, 3)))
        row = {"partition": partition, "uniq_id": p["uniq_id"]}
        if p["rttm"]:
            turns = [t for t in read_rttm(p["rttm"])]
            row.update({k: round(v, 2) for k, v in session_stats(turns, info.duration).items()})
        else:
            row["duration_min"] = round(info.duration / 60, 2)
        rows.append(row)
    write_manifest(entries, Path(out_dir) / f"{partition}.json")
    hours = sum(r["duration_min"] for r in rows) / 60
    print(f"{partition}: {len(rows)} recordings, {hours:.2f} h -> {Path(out_dir) / f'{partition}.json'}")
    return rows


def summarise(df):
    lines = ["# Table 3, measured on this copy of DISPLACE 2024", "",
             "Paper values in brackets. Shares are per session; ranges are 10th-90th percentiles.", "",
             "| Property | dev | eval |", "|---|---|---|"]

    def cell(part, col, fmt="{:.0f}"):
        x = df[df.partition == part][col] if col in df else None
        if x is None or x.isna().all():
            return "-"
        return f"{fmt.format(x.quantile(0.1))}-{fmt.format(x.quantile(0.9))}"

    def total(part):
        d = df[df.partition == part]
        return f"{len(d)} recordings, {d.duration_min.sum() / 60:.1f} h" if len(d) else "-"

    lines.append(f"| Recordings, hours [35, ~20 h / 32, ~18 h] | {total('dev')} | {total('eval')} |")
    for label, col in [("Session duration, min [30-60]", "duration_min"),
                       ("Speakers per session [3-8 annotation; 3-5 per [43]]", "speakers"),
                       ("Single-speaker speech, % of speech [60-75]", "single_speaker_pct_of_speech"),
                       ("Overlapped speech, % of speech [20-35]", "overlap_pct_of_speech"),
                       ("Non-speech, % of session [5-15]", "nonspeech_pct_of_session"),
                       ("Median turn duration, s [2-6 typical]", "median_turn_s")]:
        fmt = "{:.1f}" if col == "median_turn_s" else "{:.0f}"
        lines.append(f"| {label} | {cell('dev', col, fmt)} | {cell('eval', col, fmt)} |")
    if "overlap_pct_of_speech" in df:
        ev = df[df.partition == "eval"].overlap_pct_of_speech
        lines += ["", f"Evaluation sessions below 20 % overlap: {(ev < 20).sum()} (paper: 10 of 32); "
                      f"above 30 %: {(ev > 30).sum()} (paper: 7)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev_audio", default=None, help="omit to prepare inference audio only")
    ap.add_argument("--dev_rttm", default=None)
    ap.add_argument("--eval_audio", required=True)
    ap.add_argument("--eval_rttm", default=None, help="evaluation RTTMs, needed for scoring")
    ap.add_argument("--out", default="data/work/manifests")
    ap.add_argument("--stats_out", default=str(_common.ROOT / "results/reproduced/table3_corpus_measured"))
    a = ap.parse_args()

    rows = []
    if a.dev_audio:
        if not a.dev_rttm:
            raise SystemExit("--dev_audio needs --dev_rttm: the development set is the training data")
        rows += build("dev", a.dev_audio, a.dev_rttm, a.out)
    rows += build("eval", a.eval_audio, a.eval_rttm, a.out)

    df = pd.DataFrame(rows)
    Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.stats_out + ".csv", index=False)
    Path(a.stats_out + ".md").write_text(summarise(df))
    print(f"corpus statistics: {a.stats_out}.csv / .md")


if __name__ == "__main__":
    main()
