"""Step 5: diarize cached recordings and write RTTM files.

Algorithm 1, lines 6-11 and 15 (inference branch):

    E_s(t) -> scorer -> w_s(t) -> z(t) -> LSTM over every speaker pair -> p_k(t)
           -> threshold 0.5 -> 3-step median -> RTTM

Two modes:
  --model exp/<run>/model.pt   the trained system (dynamic or static, whichever
                               the checkpoint was trained as)
  --clustering_only            NME-SC labels written out directly, one speaker per
                               step and no decoder. This is the "+ TitaNet-Large
                               embeddings" row of Table 5: MarbleNet + TitaNet +
                               clustering, before any multi-scale decoder is added.

Besides the RTTMs, every run with a model writes what the paper asks to be
measured and reported (Section 2.3 and the paragraph on Eq. 10):
  scale_weight_variance.csv   V_s of Eq. (10) per session and scale, plus the mean.
                              It is exactly 0 for the static model and should be
                              > 0 for the dynamic one; if it is not, the dynamic
                              scorer has collapsed to constant weights.
  weights/<id>.npz            w_s(t) and p_k(t) for every step, for plotting
                              measured weight trajectories (step 10 draws one,
                              the measured counterpart of the schematic Figure 7).

Usage:
  python scripts/05_diarize.py --cache_dir data/work/cache/eval --model exp/dynamic_seed42/model.pt --out_dir exp/dynamic_seed42/eval
  python scripts/05_diarize.py --cache_dir data/work/cache/eval --clustering_only --out_dir exp/clustering/eval
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import _common
from ctxdiar.decode import decode_session
from ctxdiar.features import load_session
from ctxdiar.model import ContextAdaptiveDiarizer
from ctxdiar.postprocess import labels_to_turns, probs_to_turns
from ctxdiar.rttm import write_rttm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--clustering_only", action="store_true")
    ap.add_argument("--config", default=str(_common.ROOT / "configs/model.yaml"))
    ap.add_argument("--threshold", type=float, default=None, help="overrides inference.threshold (Table 2: 0.5)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    icfg = _common.load_yaml(a.config)["inference"]
    thr = a.threshold if a.threshold is not None else icfg["threshold"]
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out = Path(a.out_dir)
    (out / "pred_rttms").mkdir(parents=True, exist_ok=True)
    model, chunk = None, None
    if a.model:
        ck = torch.load(a.model, map_location=device)
        model = ContextAdaptiveDiarizer(weighting=ck["weighting"], **ck["model_cfg"]).to(device)
        model.load_state_dict(ck["state_dict"])
        chunk = ck["chunk_steps"]
        (out / "weights").mkdir(exist_ok=True)
        print(f"model: {a.model} ({ck['weighting']} weighting, seed {ck['seed']}, epoch {ck['epoch']})")

    var_rows, timing = [], []
    files = sorted(Path(a.cache_dir).glob("*.npz"))
    for p in files:
        sess = load_session(p)
        if sess.labels is None:
            raise SystemExit(f"{sess.uniq_id}: cache has no NME-SC labels (extracted with --no_cluster)")
        t0 = time.perf_counter()
        if model is None:
            turns = labels_to_turns(sess.labels, sess.bounds, sess.uniq_id)
        else:
            res = decode_session(model, sess, chunk=chunk, batch_size=icfg["batch_size"], device=device)
            turns = probs_to_turns(res["probs"], sess.bounds, sess.uniq_id, thr, icfg["median_steps"])
            np.savez_compressed(out / "weights" / f"{sess.uniq_id}.npz", weights=res["weights"].astype(np.float16),
                                probs=res["probs"].astype(np.float16), bounds=sess.bounds)
            var_rows.append([sess.uniq_id] + [float(v) for v in res["variance"]])
        timing.append({"uniq_id": sess.uniq_id, "decode_s": time.perf_counter() - t0})
        write_rttm(turns, out / "pred_rttms" / f"{sess.uniq_id}.rttm")
        print(f"{sess.uniq_id}: {len({t.speaker for t in turns})} speakers, {len(turns)} turns")

    if var_rows:
        S = len(var_rows[0]) - 1
        with open(out / "scale_weight_variance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["session"] + [f"V_s{s + 1}" for s in range(S)])
            w.writerows(var_rows)
            w.writerow(["mean"] + list(np.mean([r[1:] for r in var_rows], axis=0)))
        print("mean V_s per scale (Eq. 10):", np.round(np.mean([r[1:] for r in var_rows], axis=0), 6))
    with open(out / "timing.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in timing)
    print(f"RTTMs: {out / 'pred_rttms'}")


if __name__ == "__main__":
    main()
