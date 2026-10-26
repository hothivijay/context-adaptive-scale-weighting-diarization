"""The whole method on synthetic data, end to end, on a laptop CPU in a few minutes.

Nothing here needs NeMo, a GPU or the DISPLACE corpus. The demo exists so you
can watch every step of the paper run, check the code works before a long
job, and show the mechanism to someone without the real data:

  1. make synthetic "conversations": 3-4 speakers, turns of 1-6 s, some overlap,
     and noisy stretches (a door, a fan, chatter) in random places
  2. oracle speech regions -> five-scale windows (Table 2) -> common index t
  3. fake 192-d "TitaNet" embeddings. Each speaker has a direction; a window's
     embedding is the time-weighted mix of whoever talks in it, plus noise.
     Two effects make the right scale depend on context, which is the
     situation the paper's scorer is built for:
       * long windows straddle speaker changes and blur them;
       * short windows are noisier (less speech per embedding), and inside
         the noisy stretches they carry almost no speaker information.
  4. NME-SC on the evaluation sessions (Eqs. 1-2)
  5. train the static-weight and the dynamic-weight model with the same data,
     schedule and seed (Algorithm 1, Eqs. 3-9)
  6. decode, threshold 0.5, smooth, RTTM; DER with zero collar (Eq. 11)
  7. V_s of Eq. (10) for both models, Algorithm 2 on the per-session DERs,
     and a plot of measured w_s(t) on one session

Every number printed describes synthetic data only. It shows that the code
runs and how the mechanism behaves; it says nothing about the paper's 21.46 %.

Usage:  python scripts/demo_toy_pipeline.py [--epochs 15] [--seed 0] [--out outputs/toy]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _common
from ctxdiar.data import PairChunkDataset, collate
from ctxdiar.decode import decode_session
from ctxdiar.der import score
from ctxdiar.features import build_session
from ctxdiar.model import ContextAdaptiveDiarizer, count_parameters
from ctxdiar.postprocess import labels_to_turns, probs_to_turns
from ctxdiar.rttm import Turn, write_rttm
from ctxdiar.scales import WINDOWS
from ctxdiar.stats import paired_assessment
from ctxdiar.vad import oracle_regions


def make_conversation(rng, uid, minutes):
    n_spk = int(rng.integers(3, 5))
    turns, t, spk = [], 0.0, 0
    while t < minutes * 60:
        dur = float(rng.uniform(1.0, 6.0))
        turns.append(Turn(uid, round(t, 2), round(t + dur, 2), f"spk{spk}"))
        if rng.random() < 0.25:  # a back-channel or interruption overlapping the end of the turn
            other = int((spk + rng.integers(1, n_spk)) % n_spk)
            turns.append(Turn(uid, round(t + dur - 0.8, 2), round(t + dur + 0.3, 2), f"spk{other}"))
        t += dur + float(rng.uniform(-0.3, 0.8))   # negative gap = the next speaker starts early (overlap)
        spk = int((spk + rng.integers(1, n_spk)) % n_spk)
    noisy = []
    for _ in range(int(minutes * 2)):              # about two noisy stretches per minute
        a = float(rng.uniform(0, minutes * 60))
        noisy.append((a, a + float(rng.uniform(3, 10))))
    return turns, noisy


def make_embed_fn(turns, noisy, rng, dim=192, base_noise=0.35):
    names = sorted({t.speaker for t in turns})
    dirs = {s: v / np.linalg.norm(v) for s, v in zip(names, rng.standard_normal((len(names), dim)))}

    def embed_fn(scale, segs):
        w = WINDOWS[scale]
        out = np.zeros((len(segs), dim))
        for i, (a, b) in enumerate(segs):
            v = np.zeros(dim)
            for t in turns:
                ov = max(0.0, min(b, t.end) - max(a, t.start))
                if ov > 0:
                    v += ov * dirs[t.speaker]
            v /= np.linalg.norm(v) + 1e-9
            c = (a + b) / 2
            in_noise = any(x <= c <= y for x, y in noisy)
            # Less speech per window -> noisier embedding. In a noisy stretch a
            # short window is swamped (factor 6 at 0.50 s) while the longest
            # one barely notices (factor 1 at 1.25-1.50 s): the best scale
            # changes with context, which is what the scorer has to learn.
            boost = max(1.0, 1.5 / w ** 2) if in_noise else 1.0
            sigma = base_noise * np.sqrt(1.0 / w) * boost
            out[i] = v + sigma * rng.standard_normal(dim) / np.sqrt(dim) * 3
        return out

    return embed_fn


def build(rng, uid, minutes, cluster):
    turns, noisy = make_conversation(rng, uid, minutes)
    sess = build_session(uid, oracle_regions(turns), make_embed_fn(turns, noisy, rng), ref=turns, cluster=cluster,
                         # Every toy speaker talks for at least ~40 s of the 4 minutes, so clusters
                         # under 20 s (80 steps) are splinters; see nmesc.merge_small_clusters.
                         cluster_kwargs={"max_num_speakers": 6, "max_points": 1500, "min_cluster_steps": 80})
    return sess, turns, noisy


def train(sessions, weighting, epochs, seed, mcfg):
    _common.set_seed(seed)
    ds = PairChunkDataset(sessions, chunk=50, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    model = ContextAdaptiveDiarizer(weighting=weighting, **mcfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)                       # Table 2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for emb, protos, y, mask in dl:
            p, _ = model(emb, protos)
            loss = (torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y, reduction="none")
                    * mask.unsqueeze(-1)).sum() / (2 * mask.sum())             # Eq. (9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(emb), n + len(emb)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"    {weighting:7s} epoch {ep:2d}  BCE {tot / n:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_sessions", type=int, default=12)
    ap.add_argument("--eval_sessions", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/toy")
    a = ap.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    mcfg = _common.load_yaml(_common.ROOT / "configs/model.yaml")["model"]

    print("1-3. synthetic sessions, five-scale windows, embeddings aligned to t")
    train_s = [build(rng, f"toy_train_{i:02d}", a.minutes, cluster=False)[0] for i in range(a.train_sessions)]
    print("4.   NME-SC on the evaluation sessions")
    evals = [build(rng, f"toy_eval_{i:02d}", a.minutes, cluster=True) for i in range(a.eval_sessions)]
    s0 = train_s[0]
    print(f"     e.g. {s0.uniq_id}: {s0.num_steps} steps of 0.25 s, windows per scale {s0.meta['num_segments_per_scale']}")
    for sess, _, _ in evals[:3]:
        print(f"     {sess.uniq_id}: NME-SC found {sess.meta['nmesc_num_speakers']} speakers (true {len(sess.ref_speakers)})")

    print(f"5.   training both models, {a.epochs} epochs each, same data and seed")
    models = {w: train(train_s, w, a.epochs, a.seed, mcfg) for w in ("static", "dynamic")}
    print(f"     trainable parameters: static {count_parameters(models['static']):,}, dynamic {count_parameters(models['dynamic']):,}")

    print("6-7. decoding the evaluation sessions")
    ref_all = [t for _, turns, _ in evals for t in turns]
    write_rttm(ref_all, out / "reference.rttm")
    results, per_session, variances = {}, {}, {}
    hyp_c = [t for sess, _, _ in evals for t in labels_to_turns(sess.labels, sess.bounds, sess.uniq_id)]
    r = score(ref_all, hyp_c)
    results["clustering only"] = r.as_dict()
    for w, model in models.items():
        hyp, V = [], []
        for sess, _, _ in evals:
            res = decode_session(model, sess, chunk=50)
            hyp += probs_to_turns(res["probs"], sess.bounds, sess.uniq_id, threshold=0.5, median_width=3)
            V.append(res["variance"])
            if w == "dynamic" and sess is evals[0][0]:
                np.savez_compressed(out / "weights_toy_eval_00.npz", weights=res["weights"], probs=res["probs"], bounds=sess.bounds)
        write_rttm(hyp, out / f"hyp_{w}.rttm")
        r = score(ref_all, hyp)
        results[f"{w} weights"] = r.as_dict()
        per_session[w] = [r.per_file[s.uniq_id].der for s, _, _ in evals]
        variances[w] = np.mean(V, axis=0)

    print("\n     DER on synthetic data, zero collar (NOT the paper's numbers):")
    for k, v in results.items():
        print(f"       {k:16s} DER {v['DER']:6.2f}   MISS {v['MISS']:5.2f}  FA {v['FA']:5.2f}  CONF {v['CONF']:5.2f}")
    print("\n     temporal variance of the weights, Eq. (10), mean over sessions, per scale (1.50 s ... 0.50 s):")
    for w, v in variances.items():
        print(f"       {w:8s} " + "  ".join(f"{x:.2e}" for x in v))
    pa = paired_assessment(per_session["static"], per_session["dynamic"], n_boot=10_000)
    print(f"\n     Algorithm 2 over {pa['sessions']} toy sessions: mean gain {pa['mean_delta_pp']:.2f} pp, "
          f"95 % CI [{pa['ci_low_pp']:.2f}, {pa['ci_high_pp']:.2f}], Wilcoxon p {pa['wilcoxon_p']:.3g}, "
          f"win rate {100 * pa['win_rate']:.0f} %")

    print("     (On synthetic data the static-vs-dynamic difference can go either way; it depends on how the toy")
    print("      noise is built. The demo checks that every step runs and that V_s behaves as Eq. (10) says;")
    print("      the paper's claim can only be tested on DISPLACE 2024 with steps 3-7.)")

    summary = {"note": "synthetic data only; not comparable with the paper", "der": results,
               "per_session_der": {k: [round(x, 3) for x in v] for k, v in per_session.items()},
               "weight_variance": {k: v.tolist() for k, v in variances.items()}, "algorithm2": pa,
               "settings": vars(a)}
    (out / "toy_summary.json").write_text(json.dumps(summary, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(out / "weights_toy_eval_00.npz")
    t, w = z["bounds"].mean(1), z["weights"]
    sel = t < t[0] + 60
    sess0, turns0, noisy0 = evals[0]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
    for t_ in turns0:
        if t_.start < t[0] + 60:
            ax0.plot([t_.start, t_.end], [int(t_.speaker[3:])] * 2, lw=5, solid_capstyle="butt", color="#52514e")
    ax0.set_ylabel("speaker")
    ax0.set_yticks(range(len(sess0.ref_speakers)))
    for x0, x1 in noisy0:  # grey bands: the noisy stretches where short windows carry no speaker information
        if x0 < t[0] + 60:
            for ax in (ax0, ax1):
                ax.axvspan(x0, min(x1, t[0] + 60), color="#d9d9d6", lw=0)
    for s in range(w.shape[0]):
        ax1.plot(t[sel], w[s, sel], lw=1.1, label=f"{WINDOWS[s]:.2f} s")
    ax1.axhline(0.2, ls=":", color="#52514e", lw=1)
    ax1.set_ylabel("w_s(t)")
    ax1.set_xlabel("time (s)")
    ax1.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.suptitle("Toy data: measured w_s(t) of the dynamic model; grey = noisy stretch, dotted = static 1/5", fontsize=9)
    fig.savefig(out / "toy_scale_weights.png", dpi=200, bbox_inches="tight")
    print(f"\n     outputs in {out}/: reference.rttm, hyp_*.rttm, toy_summary.json, toy_scale_weights.png")


if __name__ == "__main__":
    main()
