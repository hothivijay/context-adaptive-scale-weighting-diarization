"""Step 3: run the frozen front end once per recording and cache the result.

Algorithm 1, lines 1-5, for every recording in a manifest:

    speech regions  ->  five-scale windows  ->  TitaNet-Large  ->  align to t  ->  NME-SC
    (oracle or                (Table 2)          (192-d)          (scales.py)     (Eqs. 1-2)
     MarbleNet)

  --vad oracle      speech regions from the reference RTTM (Algorithm 1 line 1).
                    Use for the training data (dev.json, dev_aug.json).
  --vad marblenet   speech regions from MarbleNet (line 2). Use for the
                    evaluation set: this is what the deployed system sees.

Why a separate step: MarbleNet and TitaNet never change (they are frozen), so
running them once and caching the embeddings makes every later step fast.
Training then reads small .npz files instead of 20 h of audio, and the
static and dynamic models of the ablation see exactly the same inputs.

The script also times each stage (VAD, embeddings, clustering) per
recording, which step 9 uses for the computational profile (Table 8).
Already-cached recordings are skipped, so an interrupted run can be resumed.

Usage:
  python scripts/03_extract_embeddings.py --manifest data/work/manifests/dev.json     --vad oracle    --out_dir data/work/cache/dev
  python scripts/03_extract_embeddings.py --manifest data/work/manifests/dev_aug.json --vad oracle    --out_dir data/work/cache/dev_aug
  python scripts/03_extract_embeddings.py --manifest data/work/manifests/eval.json    --vad marblenet --out_dir data/work/cache/eval
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

import _common
from ctxdiar.features import build_session, save_session
from ctxdiar.manifest import read_manifest
from ctxdiar.rttm import read_rttm, write_rttm
from ctxdiar.vad import oracle_regions, postprocess, regions_to_turns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vad", choices=["oracle", "marblenet"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default=str(_common.ROOT / "configs/frontend.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_cluster", action="store_true", help="skip NME-SC (training data with oracle prototypes does not need it)")
    a = ap.parse_args()
    cfg = _common.load_yaml(a.config)
    seg, vcfg, ccfg = cfg["segmentation"], cfg["vad"], cfg["clustering"]

    from ctxdiar.frontend import FrameVAD, TitaNet  # imports NeMo

    titanet = TitaNet(cfg["embeddings"]["model"], a.device)
    vad = FrameVAD(vcfg["model"], a.device) if a.vad == "marblenet" else None
    out = Path(a.out_dir)
    (out / "vad_rttm").mkdir(parents=True, exist_ok=True)
    timing_path = out / "timing.jsonl"

    rows = read_manifest(a.manifest)
    for i, row in enumerate(rows, 1):
        uid = Path(row["audio_filepath"]).stem
        dest = out / f"{uid}.npz"
        if dest.exists():
            print(f"[{i}/{len(rows)}] {uid}: cached, skipping")
            continue
        audio, sr = sf.read(row["audio_filepath"], dtype="float32")
        if sr != cfg["sample_rate"]:
            raise SystemExit(f"{uid}: {sr} Hz; run step 1, which checks the sampling rate")
        dur = len(audio) / sr
        ref = read_rttm(row["rttm_filepath"]) if row.get("rttm_filepath") else None
        if ref is not None:
            ref = [t.__class__(uid, t.start, t.end, t.speaker) for t in ref]

        t0 = time.perf_counter()
        if a.vad == "oracle":
            if not ref:
                raise SystemExit(f"{uid}: --vad oracle needs the reference RTTM in the manifest")
            regions = oracle_regions(ref)
        else:
            probs, frame_s = vad.speech_probs(audio, sr, vcfg["chunk_s"])
            regions = postprocess(probs, frame_s, vcfg["onset"], vcfg["offset"], vcfg["pad_onset"],
                                  vcfg["pad_offset"], vcfg["min_off"], vcfg["min_on"], duration=dur)
        write_rttm(regions_to_turns(regions, uid), out / "vad_rttm" / f"{uid}.rttm")
        t_vad = time.perf_counter() - t0

        t_emb = [0.0]

        def embed_fn(scale, segments):
            t1 = time.perf_counter()
            e = titanet.embed(audio, segments, sr, cfg["embeddings"]["batch_size"])
            t_emb[0] += time.perf_counter() - t1
            return e

        t2 = time.perf_counter()
        sess = build_session(
            uid, regions, embed_fn, ref=ref, cluster=not a.no_cluster,
            cluster_kwargs={"weights": ccfg["static_weights"], "max_num_speakers": ccfg["max_num_speakers"],
                            "max_points": ccfg["max_points"], "per_scale_counts": ccfg["per_scale_counts"],
                            "min_cluster_steps": ccfg["min_cluster_steps"],
                            "nme_kwargs": {"max_rp_threshold": ccfg["max_rp_threshold"],
                                           "sparse_search_volume": ccfg["sparse_search_volume"]}},
            windows=tuple(seg["windows_s"]), shifts=tuple(seg["shifts_s"]), min_segment_s=seg["min_segment_s"],
            meta={"vad": a.vad, "duration_s": dur, "audio": row["audio_filepath"]},
        )
        t_cluster = time.perf_counter() - t2 - t_emb[0]
        save_session(sess, dest)
        with open(timing_path, "a") as f:
            f.write(json.dumps({"uniq_id": uid, "duration_s": dur, "vad_s": t_vad, "embedding_s": t_emb[0],
                                "clustering_s": t_cluster, "steps": sess.num_steps}) + "\n")
        k = sess.meta.get("nmesc_num_speakers")
        ref_k = len(sess.ref_speakers) if sess.ref_speakers else "?"
        print(f"[{i}/{len(rows)}] {uid}: {dur / 60:.1f} min, {len(regions)} regions, {sess.num_steps} steps, "
              f"NME-SC speakers {k} (reference {ref_k}), embeddings {t_emb[0]:.0f} s")
    print(f"cache: {out}  ({len(list(out.glob('*.npz')))} sessions)")


if __name__ == "__main__":
    main()
