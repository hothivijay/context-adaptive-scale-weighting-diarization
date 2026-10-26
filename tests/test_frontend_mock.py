"""The NeMo wrappers in frontend.py, run against stand-in models.

NeMo and its checkpoints are not needed for the tests, so a tiny fake
`nemo.collections.asr.models` module is installed that mimics the two calls
frontend.py makes (NeMo 2.4.0 API):
    EncDecFrameClassificationModel(input_signal, input_signal_length) -> logits [B, frames, 2]
    EncDecSpeakerLabelModel.forward(input_signal, input_signal_length) -> (logits, embs [B, 192])
This checks the chunking, padding and batching logic around them.
"""

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _FakeVAD(torch.nn.Module):
    """Speech probability = 1 where the 20 ms frame is loud."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_signal, input_signal_length):
        x = input_signal[:, : (input_signal.shape[1] // 320) * 320].reshape(input_signal.shape[0], -1, 320)
        loud = (x.abs().mean(-1) > 0.05).float()
        return torch.stack([10 * (1 - loud), 10 * loud], dim=-1)


class _FakeTitaNet(torch.nn.Module):
    """Embedding = mean of the valid samples, repeated 192 times; it lets the
    test see whether the length input was respected."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_signal, input_signal_length):
        mask = torch.arange(input_signal.shape[1])[None, :] < input_signal_length[:, None]
        m = (input_signal * mask).sum(1) / input_signal_length
        return None, m[:, None].repeat(1, 192)


@pytest.fixture
def fake_nemo(monkeypatch):
    mod = types.ModuleType("nemo.collections.asr.models")
    mod.EncDecFrameClassificationModel = type("F", (), {"from_pretrained": staticmethod(lambda name: _FakeVAD())})
    mod.EncDecSpeakerLabelModel = type("S", (), {"from_pretrained": staticmethod(lambda name: _FakeTitaNet())})
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", mod)
    import importlib

    import ctxdiar.frontend as fe
    return importlib.reload(fe)


def test_frame_vad_chunks_and_frame_length(fake_nemo):
    sr = 16000
    audio = np.zeros(int(130 * sr), dtype=np.float32)           # 130 s: three 60 s chunks
    audio[10 * sr: 20 * sr] = 0.5
    audio[100 * sr: 101 * sr] = 0.5
    vad = fake_nemo.FrameVAD("any", "cpu")
    probs, frame_s = vad.speech_probs(audio, sr, chunk_s=60)
    assert frame_s == pytest.approx(0.02)
    assert len(probs) == pytest.approx(130 / 0.02, abs=2)
    from ctxdiar.vad import postprocess
    reg = postprocess(probs, frame_s, pad_onset=0, pad_offset=0)
    assert reg == [(10.0, 20.0), (100.0, 101.0)]


def test_titanet_batches_respect_lengths(fake_nemo):
    sr = 16000
    audio = np.concatenate([np.full(sr, 1.0), np.full(sr, 3.0)]).astype(np.float32)
    tn = fake_nemo.TitaNet("titanet_large", "cpu")
    segs = np.array([[0.0, 1.0], [1.0, 2.0], [0.5, 1.5], [1.75, 2.0]])
    e = tn.embed(audio, segs, sr, batch_size=3)                 # two batches, uneven lengths
    assert e.shape == (4, 192)
    assert e[:, 0] == pytest.approx([1.0, 3.0, 2.0, 3.0], abs=1e-4)
