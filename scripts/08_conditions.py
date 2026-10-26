"""Step 8: evaluation DER by acoustic condition (Tables 6 and 7).

For each evaluation session this measures the SNR and the overlap share
(src/ctxdiar/conditions.py says exactly how), puts the session in the bands
the paper uses, and reports per band:

  * sessions           how many sessions fall in the band
  * pooled DER         error seconds / speech seconds over the band's sessions
                       (how the overall 21.46 % is computed)
  * mean session DER   each session counts once (what the paper's
                       session-weighted means under Table 7 use)
  * mean duration      the paper explains the gap between those two kinds of
                       average by difficult sessions being shorter, and asks
                       for this column to support it

If the static-weight system is given too, each band also gets the paired
difference and its bootstrap interval, which is what the paper says is
needed to tell whether dynamic weighting helps everywhere or only in some
conditions.

The short-utterance row of Table 7 comes from step 6 with --short_utterances.

Usage:
  python scripts/08_conditions.py --manifest data/work/manifests/eval.json \
      --dynamic results/reproduced/eval_dynamic_seed42_per_file.csv \
      --static  results/reproduced/eval_static_seed42_per_file.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf

import _common
from ctxdiar.conditions import estimate_snr_db, overlap_band, session_stats, snr_band
from ctxdiar.manifest import read_manifest
from ctxdiar.rttm import read_rttm
from ctxdiar.stats import paired_assessment


def per_session(manifest):
    rows = []
    for r in read_manifest(manifest):
        uid = Path(r["audio_filepath"]).stem
        audio, sr = sf.read(r["audio_filepath"], dtype="float32")
        turns = read_rttm(r["rttm_filepath"])
        st = session_stats(turns, len(audio) / sr)
        rows.append({"session": uid, "snr_db": estimate_snr_db(audio, sr, turns),
                     "overlap_pct": st["overlap_pct_of_speech"], "duration_min": st["duration_min"]})
    df = pd.DataFrame(rows)
    df["snr_band"] = df.snr_db.apply(snr_band)
    df["overlap_band"] = df.overlap_pct.apply(overlap_band)
    return df


def band_table(df, band_col, order, has_static):
    rows = []
    for band in order:
        g = df[df[band_col] == band]
        if g.empty:
            rows.append({"condition": band, "sessions": 0})
            continue
        err = (g.DER_dyn * g.scored_speech_s).sum() / g.scored_speech_s.sum()
        row = {"condition": band, "sessions": len(g), "pooled_DER": round(err, 2),
               "mean_session_DER": round(g.DER_dyn.mean(), 2), "mean_duration_min": round(g.duration_min.mean(), 1)}
        if has_static and len(g) >= 2:
            pa = paired_assessment(g.DER_static.values, g.DER_dyn.values)
            row.update({"static_pooled_DER": round((g.DER_static * g.scored_speech_s).sum() / g.scored_speech_s.sum(), 2),
                        "mean_delta_pp": round(pa["mean_delta_pp"], 2),
                        "ci95": f"[{pa['ci_low_pp']:.2f}, {pa['ci_high_pp']:.2f}]",
                        "win_rate_pct": round(100 * pa["win_rate"])})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="evaluation manifest (audio + reference RTTM)")
    ap.add_argument("--dynamic", required=True, help="per-session CSV of the full system (step 6)")
    ap.add_argument("--static", default=None, help="per-session CSV of the static-weight system (optional)")
    ap.add_argument("--out", default=str(_common.ROOT / "results/reproduced/conditions"))
    a = ap.parse_args()

    df = per_session(a.manifest)
    dyn = pd.read_csv(a.dynamic).rename(columns={"DER": "DER_dyn"})[["session", "DER_dyn", "scored_speech_s"]]
    df = df.merge(dyn, on="session")
    if a.static:
        df = df.merge(pd.read_csv(a.static).rename(columns={"DER": "DER_static"})[["session", "DER_static"]], on="session")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out + "_per_session.csv", index=False)
    t6 = band_table(df, "snr_band", ["Low (5-10 dB)", "Medium (10-15 dB)", "High (15-20 dB)"], bool(a.static))
    t7 = band_table(df, "overlap_band", ["Low overlap (< 20%)", "Medium overlap (20-30%)", "High overlap (> 30%)"], bool(a.static))
    t6.to_csv(a.out + "_table6_snr.csv", index=False)
    t7.to_csv(a.out + "_table7_overlap.csv", index=False)
    md = ["# Tables 6 and 7 (reproduced)", "",
          "SNR is a reference-based estimate (src/ctxdiar/conditions.py); the paper does not state its estimator.", "",
          "## Table 6: by SNR", "", t6.to_markdown(index=False), "",
          "## Table 7: by overlap share of speech", "", t7.to_markdown(index=False), ""]
    Path(a.out + ".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
