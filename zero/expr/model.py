"""GestureTCN — the Phase E N3 model (docs/NEURAL_GESTURES_PLAN.md).

A small CAUSAL temporal convolution network: audio features in
(zero/expr/features.py, 20 Hz), 12 closure/wrist targets out
(zero/expr/retarget.py order). Causal because the sidecar serves it
incrementally on audio-so-far; ~1 M parameters because the output space
is 12 DoF of texture, not a body.

torch imports live inside functions: the Pi and the mock sidecar never
pay for them.
"""
from __future__ import annotations

import numpy as np

from zero.expr.features import FEAT_DIM, FRAME_HZ
from zero.expr.retarget import TARGET_DIM

_CHANNELS = 96
_LAYERS = 6          # dilations 1..32 -> ~3.2 s receptive field at 20 Hz


def build_model():
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self, ch: int, dilation: int):
            super().__init__()
            self.pad = 2 * dilation
            self.conv = nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation)
            self.norm = nn.GroupNorm(8, ch)
            self.act = nn.GELU()

        def forward(self, x):
            import torch.nn.functional as F

            y = F.pad(x, (self.pad, 0))          # causal: left-pad only
            return x + self.act(self.norm(self.conv(y)))

    class GestureTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(FEAT_DIM, _CHANNELS, 1)
            self.blocks = nn.ModuleList(
                [_Block(_CHANNELS, 2 ** i) for i in range(_LAYERS)])
            self.out = nn.Conv1d(_CHANNELS, TARGET_DIM, 1)

        def forward(self, feats):                # (B, T, FEAT) -> (B, T, 12)
            import torch

            x = self.inp(feats.transpose(1, 2))
            for b in self.blocks:
                x = b(x)
            y = self.out(x).transpose(1, 2)
            closures = torch.sigmoid(y[..., :10])
            wrists = torch.tanh(y[..., 10:])     # normalized ±1 (=±45 deg)
            return torch.cat([closures, wrists], dim=-1)

    return GestureTCN()


def save_checkpoint(model, path: str, meta: dict | None = None) -> None:
    import torch

    torch.save({"state": model.state_dict(),
                "feat_dim": FEAT_DIM, "target_dim": TARGET_DIM,
                "meta": meta or {}}, path)


def load_checkpoint(path: str):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["feat_dim"] == FEAT_DIM, "checkpoint/feature mismatch"
    m = build_model()
    m.load_state_dict(ck["state"])
    m.eval()
    return m


class TCNServeModel:
    """The sidecar's GestureModel interface around a trained checkpoint.

    Emits the same frame dicts as EnergyMockModel, in the closure scale
    the Pi's blend expects (texture band, capped) — the raw model output
    is attenuated into [0, cap] so a young, imperfect model can never
    command a full fist as 'texture'."""

    FRAME_HZ = FRAME_HZ
    TEXTURE_CAP = 0.35
    WRIST_CAP_DEG = 6.0

    def __init__(self, checkpoint: str, device: str = "cpu"):
        import torch

        self._torch = torch
        self._model = load_checkpoint(checkpoint).to(device)
        self._device = device

    def frames(self, audio: np.ndarray, sr: int) -> list[dict]:
        from zero.expr.features import extract

        feats = extract(audio, sr)
        if len(feats) == 0:
            return []
        with self._torch.no_grad():
            x = self._torch.from_numpy(feats[None]).to(self._device)
            y = self._model(x)[0].cpu().numpy()
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        out = []
        for t in range(len(y)):
            # both hands' closures averaged into the (symmetric) texture
            # frame the client consumes; per-side wrists kept
            cl = {f: float(np.clip(
                0.5 * (y[t, k] + y[t, 5 + k]) * self.TEXTURE_CAP,
                0.0, self.TEXTURE_CAP)) for k, f in enumerate(fingers)}
            out.append({
                "t": round(t / self.FRAME_HZ, 3),
                "closure": cl,
                "wrist_deg": {
                    "left": float(np.clip(y[t, 10] * 45.0,
                                          -self.WRIST_CAP_DEG,
                                          self.WRIST_CAP_DEG)),
                    "right": float(np.clip(y[t, 11] * 45.0,
                                           -self.WRIST_CAP_DEG,
                                           self.WRIST_CAP_DEG))},
            })
        return out
