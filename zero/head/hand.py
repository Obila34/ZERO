"""Hand/body-pose source for DIRECT head teleoperation.

Runs a YOLO-pose model (ONNX, local, on the Pi's onnxruntime) over camera frames,
pulls out a horizontal control signal — the hand (wrist) X, or the head (nose) X —
smooths it with a 1€ (One-Euro) filter, and exposes a normalised value in [-1, 1].

The head maps that DIRECTLY to a pan angle (open-loop teleoperation, like an FPV
head-tracker): hand centre → 0°, hand full-left → -limit. No visual servo, so it's
snappy and can't wind up. Detection is ~8 fps on the Pi; the head controller runs
at 25 Hz and interpolates between updates, so motion stays smooth.

Same model, two control signals: keypoint='wrist' (hand control now) or 'head'
(nose — the head-motion control version) — swappable at construction.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10


class OneEuro:
    """1€ filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass whose cutoff
    adapts to speed — heavy smoothing when the signal is slow (kills jitter),
    light when fast (low lag). The standard for real-time interactive input."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3,
                 dcutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x = None
        self._dx = 0.0
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float | None = None) -> float:
        t = time.monotonic() if t is None else t
        if self._x is None:
            self._x, self._t = x, t
            return x
        dt = max(1e-3, t - self._t)
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None


class HandPoseSource:
    """Turns camera frames into a smoothed horizontal control value in [-1, 1].

    update(frame) runs synchronously (~116 ms on the Pi). start(eyes) runs it on a
    background thread so the control loop reads the latest value without blocking.
    """

    def __init__(self, model_path, *, keypoint: str = "wrist", imgsz: int = 320,
                 conf: float = 0.25, kp_conf: float = 0.15,
                 min_cutoff: float = 1.0, beta: float = 0.3,
                 mirror: bool = True, gain: float = 2.2):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self._sz = int(imgsz)
        self._conf = float(conf)
        self._kp_conf = float(kp_conf)   # keypoints are noisier than the person box
        self._keypoint = keypoint
        self._mirror = bool(mirror)   # webcam is not mirrored; flip so 'your left' = head left
        self._gain = float(gain)      # amplify: a natural hand sweep should reach full range
        self._filt = OneEuro(min_cutoff=min_cutoff, beta=beta)
        self._x = 0.0
        self._conf_last = 0.0
        self._lock = threading.Lock()
        self._eyes = None
        self._thread = None
        self._stop = threading.Event()

    # ── inference ────────────────────────────────────────────────────────────
    def _preprocess(self, frame_rgb):
        import cv2

        img = cv2.resize(frame_rgb, (self._sz, self._sz))
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]     # 1,3,H,W
        return np.ascontiguousarray(x)

    def _signal(self, kps: np.ndarray) -> float | None:
        """Body-relative horizontal control signal in shoulder-width units, or None.

        The offset of the control point from the SHOULDER MIDLINE, divided by the
        shoulder width. This is symmetric and independent of where the person
        stands, how they are framed, or how far away they are (a wider-appearing
        person has proportionally wider offsets) — unlike raw frame position,
        whose zero drifts with stance. Standard vision-teleop practice.
        """
        ls, rs = kps[L_SHOULDER], kps[R_SHOULDER]
        if ls[2] < self._kp_conf or rs[2] < self._kp_conf:
            return None                          # need both shoulders for a midline
        mid = 0.5 * (ls[0] + rs[0])
        width = abs(ls[0] - rs[0])
        if width < 1e-3:
            return None
        if self._keypoint == "head":
            k = kps[NOSE]
            if k[2] < self._kp_conf:
                return None
            cx = k[0]
        else:
            # track the RAISED hand — the wrist higher in the frame (smaller y).
            # By height, not confidence, so it never flip-flops to the resting hand.
            lw, rw = kps[L_WRIST], kps[R_WRIST]
            cand = [w for w in (lw, rw) if w[2] >= self._kp_conf]
            if not cand:
                return None
            cx = min(cand, key=lambda w: w[1])[0]
        return float((cx - mid) / width)         # shoulder-width units, signed

    def update(self, frame_rgb, now: float | None = None):
        """Run pose on one frame. Returns (x_norm in [-1,1], person_conf). Holds the
        last value (conf 0) when no confident hand/head is found."""
        if frame_rgb is None:
            return self.value
        try:
            out = self._sess.run(None, {self._in: self._preprocess(frame_rgb)})[0]
        except Exception:
            return self.value
        p = out[0].T                    # (N, 56)
        conf = p[:, 4]
        best = int(np.argmax(conf))
        if conf[best] < self._conf:
            with self._lock:
                self._conf_last = 0.0
            return self.value
        kps = p[best, 5:5 + 17 * 3].reshape(17, 3)
        sig = self._signal(kps)
        if sig is None:
            with self._lock:
                self._conf_last = 0.0
            return self.value
        # sig is in shoulder-width units; gain sets how many shoulder-widths of
        # hand travel reach the full range (gain 0.7 -> ~1.4 widths = full swing).
        hx = max(-1.0, min(1.0, sig * self._gain))            # clamp to -1..1
        if self._mirror:
            hx = -hx
        hx = self._filt(hx, now)
        with self._lock:
            self._x = hx
            self._conf_last = float(conf[best])
        return hx, self._conf_last

    @property
    def value(self):
        with self._lock:
            return self._x, self._conf_last

    # ── threaded mode (for HeadSystem) ───────────────────────────────────────
    def start(self, eyes) -> None:
        self._eyes = eyes
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="hand-pose", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self._eyes.current_frame() if self._eyes is not None else None
            if frame is None:
                self._stop.wait(0.05)
                continue
            self.update(frame)
            # pose is the pace-setter (~8 fps); no extra sleep needed
