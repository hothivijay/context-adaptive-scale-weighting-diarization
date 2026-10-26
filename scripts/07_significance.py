"""Step 7: Algorithm 2 and Table 9 (static vs dynamic weighting, session level).

The paper's Table 9 is still a placeholder: the procedure is written down
(Algorithm 2) but the numbers were never computed. This script computes them
from the per-session DER files that step 6 writes for the two systems:

  mean per-session gain, 95 % bootstrap CI (B = 10 000), Wilcoxon p, win rate

and, if you trained several seeds of each configuration, the mean and
standard deviation of corpus DER over seeds (the second part of Table 9 the
paper asks for; the session bootstrap does not capture seed variation).

Sessions are matched by name. A session missing from either file stops the
script: a paired test on unpaired data is meaningless.

Usage (one seed):
  python scripts/07_significance.py \
      --static  results/reproduced/eval_static_seed42_per_file.csv \
      --dynamic results/reproduced/eval_dynamic_seed42_per_file.csv
Usage (several seeds; the first pair is used for the paired test):
  python scripts/07_significance.py --static s42.csv s43.csv s44.csv --dynamic d42.csv d43.csv d44.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import _common
from ctxdiar.stats import paired_assessment, seed_summary


def corpus_der(csv_path):
    """Corpus DER recomputed from the per-session file (duration weighted)."""
    d = pd.read_csv(csv_path)
    return float((d.DER * d.scored_speech_s).sum() / d.scored_speech_s.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", nargs="+", required=True, help="per-session CSV(s) of the static-weight system")
    ap.add_argument("--dynamic", nargs="+", required=True, help="per-session CSV(s) of the dynamic system")
    ap.add_argument("--out", default=str(_common.ROOT / "results/reproduced/table9_significance"))
    ap.add_argument("--config", default=str(_common.ROOT / "configs/experiment.yaml"))
    a = ap.parse_args()
    sig = _common.load_yaml(a.config)["significance"]

    s = pd.read_csv(a.static[0]).set_index("session")
    d = pd.read_csv(a.dynamic[0]).set_index("session")
    if set(s.index) != set(d.index):
        raise SystemExit(f"session lists differ: only static {sorted(set(s.index) - set(d.index))}, "
                         f"only dynamic {sorted(set(d.index) - set(s.index))}")
    order = sorted(s.index)
    res = paired_assessment(s.loc[order, "DER"].values, d.loc[order, "DER"].values,
                            n_boot=sig["bootstrap_replicates"], alpha=sig["alpha"], seed=sig["seed"])
    out = {"paired": res, "inputs": {"static": a.static, "dynamic": a.dynamic}}
    if len(a.static) > 1 or len(a.dynamic) > 1:
        out["seeds"] = {"static": seed_summary([corpus_der(p) for p in a.static]),
                        "dynamic": seed_summary([corpus_der(p) for p in a.dynamic])}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    lines = [
        "# Table 9 (reproduced): static vs dynamic scale weighting, paired by session", "",
        "| Quantity | Value |", "|---|---|",
        f"| Sessions M | {res['sessions']} |",
        f"| Mean per-session DER reduction (pp), Eq. (13) | {res['mean_delta_pp']:.2f} |",
        f"| 95 % bootstrap CI (B = {res['bootstrap_replicates']}) | [{res['ci_low_pp']:.2f}, {res['ci_high_pp']:.2f}] |",
        f"| Wilcoxon signed-rank p (two-sided) | {res['wilcoxon_p']:.4g} |",
        f"| Win rate (dynamic better) | {res['win_rate'] * 100:.0f} % |",
        f"| Interval excludes zero | {'yes' if res['supported'] else 'no'} |",
    ]
    if "seeds" in out:
        ss, ds = out["seeds"]["static"], out["seeds"]["dynamic"]
        lines += ["", "| Configuration | Seeds | Eval DER mean | SD |", "|---|---|---|---|",
                  f"| Static weights | {ss['seeds']} | {ss['mean']:.2f} | {ss['std']:.2f} |",
                  f"| Dynamic weights | {ds['seeds']} | {ds['mean']:.2f} | {ds['std']:.2f} |"]
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
