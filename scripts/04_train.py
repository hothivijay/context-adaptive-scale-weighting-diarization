"""Step 4: train the reliability scorer and the recurrent decoder.

Algorithm 1, lines 12-13: minimise the binary cross-entropy of Eq. (9) with
Adam (learning rate 1e-3) and cosine annealing. Only the scorer and the
decoder are trained; TitaNet and MarbleNet stay frozen (their outputs are
already cached by step 3).

Run it twice to get the controlled contrast at the end of Table 5:

    --weighting dynamic   full system: w_s(t) from the scorer, Eqs. (3)-(5)
    --weighting static    same decoder, same data, same schedule, w_s(t) = 1/5

Training data. The challenge has no training partition, so, like the paper,
the decoder is trained on the development recordings (oracle speech regions)
plus their augmented copies. Consequence, stated in the paper too: the
development DER of a trained model is training-set error. Only the
evaluation DER is held out.

Model selection. The paper does not say how the number of epochs was chosen.
By default this script trains for the configured number of epochs and keeps
the last model, which needs no held-out data. If you want a held-out check
(the paper suggests cross-validation over development sessions), pass
--holdout N: N development sessions *and their augmented copies* are kept out
of training and their loss is logged every epoch, and the epoch with the
lowest held-out loss is also saved as model_best.pt.

Output (in --exp_dir):
  model.pt        last epoch: weights + the exact config used
  model_best.pt   only with --holdout
  train_log.csv   loss per epoch (and held-out loss)
  config_used.yaml

Usage:
  python scripts/04_train.py --cache_dirs data/work/cache/dev data/work/cache/dev_aug --weighting dynamic --exp_dir exp/dynamic_seed42
  python scripts/04_train.py --cache_dirs data/work/cache/dev data/work/cache/dev_aug --weighting static  --exp_dir exp/static_seed42
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import _common
from ctxdiar.data import PairChunkDataset, collate
from ctxdiar.features import load_session
from ctxdiar.model import ContextAdaptiveDiarizer, count_parameters


def masked_bce(probs, y, mask):
    """Eq. (9): mean over valid steps t and both output units n."""
    loss = torch.nn.functional.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), y, reduction="none")
    m = mask.unsqueeze(-1).expand_as(loss)
    return (loss * m).sum() / m.sum().clamp(min=1)


def run_epoch(model, loader, device, optim=None, sched=None, grad_clip=None):
    train = optim is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for emb, protos, y, mask in loader:
            emb, protos, y, mask = emb.to(device), protos.to(device), y.to(device), mask.to(device)
            probs, _ = model(emb, protos)
            loss = masked_bce(probs, y, mask)
            if train:
                optim.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                sched.step()          # cosine annealing is stepped per batch, over the whole run
            total += loss.item() * len(emb)
            n += len(emb)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dirs", nargs="+", required=True, help="feature caches from step 3 (oracle VAD)")
    ap.add_argument("--weighting", choices=["dynamic", "static"], required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--config", default=str(_common.ROOT / "configs/model.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="overrides training.seed; use 3+ seeds for Table 9")
    ap.add_argument("--epochs", type=int, default=None, help="overrides training.epochs")
    ap.add_argument("--holdout", type=int, default=0, help="development sessions kept out for a held-out loss")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = _common.load_yaml(a.config)
    tcfg = cfg["training"]
    if a.seed is not None:
        tcfg["seed"] = a.seed
    if a.epochs is not None:
        tcfg["epochs"] = a.epochs
    _common.set_seed(tcfg["seed"])
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp = Path(a.exp_dir)
    exp.mkdir(parents=True, exist_ok=True)

    files = sorted(p for d in a.cache_dirs for p in Path(d).glob("*.npz"))
    if not files:
        raise SystemExit("no cached sessions found; run step 3 first")
    sessions = [load_session(p) for p in files]
    print(f"{len(sessions)} training sessions, {sum(s.num_steps for s in sessions) * 0.25 / 3600:.1f} h of speech steps")

    held = []
    if a.holdout:
        originals = sorted(s.uniq_id for s in sessions if "_aug" not in s.uniq_id)
        rng = np.random.default_rng(tcfg["seed"])
        held_ids = set(rng.choice(originals, size=a.holdout, replace=False))
        is_held = lambda uid: uid in held_ids or any(uid.startswith(h + "_aug") for h in held_ids)  # noqa: E731
        held = [s for s in sessions if is_held(s.uniq_id)]
        sessions = [s for s in sessions if not is_held(s.uniq_id)]
        print(f"held out: {sorted(held_ids)} ({len(held)} files incl. augmented copies)")

    train_ds = PairChunkDataset(sessions, tcfg["chunk_steps"], tcfg["prototype_source"],
                                impostor_prob=tcfg["impostor_prob"], seed=tcfg["seed"])
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate,
                        num_workers=tcfg["num_workers"], drop_last=False)
    held_loader = None
    if held:
        held_ds = PairChunkDataset(held, tcfg["chunk_steps"], tcfg["prototype_source"], swap_prob=0.0, impostor_prob=0.0)
        held_loader = DataLoader(held_ds, batch_size=tcfg["batch_size"], collate_fn=collate)

    model = ContextAdaptiveDiarizer(weighting=a.weighting, **cfg["model"]).to(device)
    print(f"{a.weighting} model: {count_parameters(model):,} trainable parameters; {len(train_ds):,} training items per epoch")

    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tcfg["epochs"] * len(loader))

    cfg_used = {"weighting": a.weighting, **cfg, "cache_dirs": a.cache_dirs, "holdout": a.holdout}
    (exp / "config_used.yaml").write_text(yaml.safe_dump(cfg_used, sort_keys=False))

    def save(name, epoch):
        torch.save({"state_dict": model.state_dict(), "weighting": a.weighting, "model_cfg": cfg["model"],
                    "chunk_steps": tcfg["chunk_steps"], "seed": tcfg["seed"], "epoch": epoch}, exp / name)

    best = float("inf")
    with open(exp / "train_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_bce", "heldout_bce", "lr", "seconds"])
        for ep in range(1, tcfg["epochs"] + 1):
            t0 = time.time()
            tr = run_epoch(model, loader, device, optim, sched, tcfg["grad_clip"])
            hl = run_epoch(model, held_loader, device) if held_loader else float("nan")
            log.writerow([ep, f"{tr:.5f}", f"{hl:.5f}", f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
            f.flush()
            print(f"epoch {ep:3d}  train BCE {tr:.4f}  held-out BCE {hl:.4f}  ({time.time() - t0:.0f} s)")
            if held_loader and hl < best:
                best = hl
                save("model_best.pt", ep)
    save("model.pt", tcfg["epochs"])
    print(f"saved {exp / 'model.pt'}")


if __name__ == "__main__":
    main()
