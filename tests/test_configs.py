"""The YAML configs carry the paper's Table 2 values exactly."""

from pathlib import Path

import yaml

from ctxdiar import paper
from ctxdiar.scales import SHIFTS, STATIC_WEIGHTS, WINDOWS

ROOT = Path(__file__).resolve().parents[1]
H = paper.HPARAMS


def load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_scales_module_matches_table2():
    assert WINDOWS == H["windows_s"] and SHIFTS == H["shifts_s"] and STATIC_WEIGHTS == H["static_weights"]


def test_frontend_config():
    f = load("frontend.yaml")
    assert f["sample_rate"] == H["sample_rate"]
    assert tuple(f["segmentation"]["windows_s"]) == H["windows_s"]
    assert tuple(f["segmentation"]["shifts_s"]) == H["shifts_s"]
    assert tuple(f["clustering"]["static_weights"]) == H["static_weights"]
    assert f["embeddings"]["dim"] == H["emb_dim"] and f["embeddings"]["model"] == "titanet_large"
    assert f["vad"]["min_off"] == H["vad_overlap_tolerance_s"]


def test_model_config():
    m = load("model.yaml")
    md, tr, inf = m["model"], m["training"], m["inference"]
    assert md["num_scales"] == H["num_scales"] and md["emb_dim"] == H["emb_dim"]
    assert md["channels"] == H["cnn_channels"] and md["conv_repeats"] == H["conv_repeats"]
    assert md["kernel"] == H["kernel"] and md["context"] == H["context_half_width"]
    assert md["lstm_layers"] == H["lstm_layers"] and md["dropout"] == H["dropout"]
    assert tr["loss"] == H["loss"] and tr["optimizer"] == H["optimizer"] and tr["scheduler"] == H["scheduler"]
    assert tr["lr"] == H["lr"] and inf["threshold"] == H["threshold"]


def test_experiment_config():
    e = load("experiment.yaml")
    assert e["scoring"]["collar"] == 0.0
    assert e["augmentation"]["snr_db"] == [10, 20] and e["augmentation"]["speed"] == [0.9, 1.1]
    assert e["significance"]["bootstrap_replicates"] == 10000
