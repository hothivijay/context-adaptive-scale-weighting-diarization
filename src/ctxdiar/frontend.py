"""The frozen, pre-trained front end: MarbleNet VAD and TitaNet-Large (NVIDIA NeMo).

The paper initialises both from pre-trained weights and never updates them
(section "Implementation"). They are loaded here through NeMo's
`from_pretrained`, which downloads the checkpoints once (NGC / Hugging Face)
and caches them.

NeMo is imported inside the functions, not at the top of the file, so that
the rest of the package (and the unit tests, and the toy demo) works on a
machine without NeMo or a GPU.

Model names
-----------
* VAD: the paper says "MarbleNet [37]" without naming a checkpoint. The default
  is NVIDIA's multilingual *frame-level* MarbleNet, which gives one speech
  probability every 20 ms and handles many languages, matching the
  multilingual DISPLACE audio. Any NeMo frame-VAD checkpoint name can be given
  in configs/frontend.yaml.
* Embeddings: `titanet_large`, 192-dimensional, as in the paper.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _device(device: str | None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


class FrameVAD:
    """MarbleNet frame VAD: audio -> speech probability per frame."""

    def __init__(self, model_name: str = "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0", device: str | None = None):
        from nemo.collections.asr.models import EncDecFrameClassificationModel

        self.device = _device(device)
        self.model = EncDecFrameClassificationModel.from_pretrained(model_name).to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def speech_probs(self, audio: np.ndarray, sr: int = 16000, chunk_s: float = 60.0):
        """Returns (probs [F], frame_s). Long recordings are processed in 60 s
        pieces so a one-hour session fits in GPU memory; the frame length is
        read from the model output rather than assumed."""
        import torch

        probs: List[np.ndarray] = []
        hop = int(chunk_s * sr)
        frame_s = None
        for i in range(0, len(audio), hop):
            piece = audio[i: i + hop]
            if len(piece) < sr // 10:          # a sliver at the very end: treat as non-speech
                probs.append(np.zeros(max(1, int(round(len(piece) / sr / (frame_s or 0.02)))), dtype=np.float32))
                continue
            x = torch.from_numpy(piece).float().unsqueeze(0).to(self.device)
            n = torch.tensor([x.shape[1]], device=self.device)
            with torch.no_grad():
                logits = self.model(input_signal=x, input_signal_length=n)    # [1, frames, 2]
            p = torch.softmax(logits, dim=-1)[0, :, 1].float().cpu().numpy()
            frame_s = frame_s or (len(piece) / sr) / len(p)
            probs.append(p)
        return np.concatenate(probs), (frame_s or 0.02)


class TitaNet:
    """TitaNet-Large: one 192-d embedding per window."""

    def __init__(self, model_name: str = "titanet_large", device: str | None = None):
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self.device = _device(device)
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name).to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def embed(self, audio: np.ndarray, segments: np.ndarray, sr: int = 16000, batch_size: int = 64) -> np.ndarray:
        """segments [N x 2] in seconds -> embeddings [N x 192].

        Windows of one scale have (almost) the same length, so batches pad very
        little. The model's length input tells it where the real audio ends.
        """
        import torch

        out = []
        for i in range(0, len(segments), batch_size):
            seg = segments[i: i + batch_size]
            pieces = [audio[int(a * sr): max(int(b * sr), int(a * sr) + 1)] for a, b in seg]
            L = max(len(p) for p in pieces)
            x = torch.zeros(len(pieces), L)
            for j, p in enumerate(pieces):
                x[j, : len(p)] = torch.from_numpy(p)
            n = torch.tensor([len(p) for p in pieces])
            with torch.no_grad():
                _, emb = self.model.forward(input_signal=x.to(self.device), input_signal_length=n.to(self.device))
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192), dtype=np.float32)
