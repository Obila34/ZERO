"""Hand/body-pose source for DIRECT head teleoperation.

Runs a YOLO-pose model (ONNX, local, on the Pi's onnxruntime) over camera frames
and produces a BODY-RELATIVE horizontal control signal: the offset of the raised
wrist (or the nose) from the shoulder midline, divided by shoulder width. That is
symmetric and independent of where the person stands / how they are framed / how
far away they are - unlike raw frame position, whose zero drifts with stance. A 1
euro filter smooths it; the head maps it straight to a pan angle (open-loop, can't
wind up). Standard vision-teleop practice (shoulder->wrist / offset-relative-to-body).

Robustness (learned the hard way): the nano pose model invents low-confidence,
degenerate "people" (all keypoints piled at a frame edge, ~6 px shoulder width) on
empty/ambiguous scenes. Dividing by that tiny width explodes the signal to the
rails. So we REJECT detections whose shoulders are implausibly close, demand a
solid person confidence, and read the RAW camera frame (not the detection loop's
scene snapshot, which stalls when remote perception retries on backoff).

keypoint='wrist' (hand control) or 'head' (nose - head-motion control) - swappable.
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
    """1 euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass whose cutoff
    adapts to speed - heavy smoothing when the signal is slow (kills jitter),
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
    """Turns camera frames into a smoothed, body-relative control value in [-1, 1]."""

    def __init__(self, model_path, *, keypoint: str = "wrist", imgsz: int = 320,
                 conf: float = 0.5, kp_conf: float = 0.3,
                 min_shoulder_frac: float = 0.10, deadzone: float = 0.0,
                 min_cutoff: float = 1.5, beta: float = 0.5,
                 mirror: bool = True, gain: float = 1.3):
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self._sz = int(imgsz)
        self._conf = float(conf)          # min person-box confidence
        self._kp_conf = float(kp_conf)    # min per-keypoint confidence
        # Reject degenerate detections: real shoulders span a decent fraction of
        # the frame; a ~6 px width is the model hallucinating on an empty scene.
        self._min_shoulder_px = float(min_shoulder_frac) * self._sz
        self._deadzone = float(deadzone)   # |signal| below this -> hold centre (kills jitter)
        self._keypoint = keypoint
        self._mirror = bool(mirror)
        self._gain = float(gain)
        self._filt = OneEuro(min_cutoff=min_cutoff, beta=beta)
        self._x = 0.0
        self._conf_last = 0.0
        self._lock = threading.Lock()
        self._eyes = None
        self._thread = None
        self._stop = threading.Event()

    # -- inference -------------------------------------------------------------
    def _preprocess(self, frame_rgb):
        """Letterbox to a square (preserve aspect - a plain resize squashes
        640x480 by 1.33x and degrades pose). Keypoints come back in this padded
        space; the body-relative signal uses them relatively, so the pad cancels."""
        import cv2

        h, w = frame_rgb.shape[:2]
        s = self._sz / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resized = cv2.resize(frame_rgb, (nw, nh))
        canvas = np.full((self._sz, self._sz, 3), 114, np.uint8)   # YOLO gray pad
        ox, oy = (self._sz - nw) // 2, (self._sz - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        x = canvas.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]     # 1,3,H,W
        return np.ascontiguousarray(x)

    def _best_person(self, out):
        p = out[0].T                    # (N, 56)
        conf = p[:, 4]
        best = int(np.argmax(conf))
        return float(conf[best]), p[best, 5:5 + 17 * 3].reshape(17, 3)

    def _signal(self, kps: np.ndarray):
        """Body-relative horizontal control signal, or None if unusable.
        hand: raised-wrist offset from the shoulder midline / shoulder width.
        head: nose offset from the EAR midline / ear span (a head-yaw proxy)."""
        if self._keypoint == "head":
            return self._head_yaw(kps)
        return self._hand_offset(kps)

    def _head_yaw(self, kps: np.ndarray):
        """Head YAW from face geometry: the nose swings across the ears as you
        turn your head. Offset of the nose from the ear midline, normalised by ear
        span (head width) - distance/position independent, and it isolates yaw
        from where the head sits in the frame. Falls back to the eyes when an ear
        is hidden; works up close when the shoulders are cropped out."""
        nose = kps[NOSE]
        if nose[2] < self._kp_conf:
            return None
        for a, b, floor in ((L_EAR, R_EAR, 0.04), (L_EYE, R_EYE, 0.02)):
            ka, kb = kps[a], kps[b]
            if ka[2] >= self._kp_conf and kb[2] >= self._kp_conf:
                span = abs(ka[0] - kb[0])
                if span >= floor * self._sz:
                    return float((nose[0] - 0.5 * (ka[0] + kb[0])) / span)
        return None

    def _hand_offset(self, kps: np.ndarray):
        """Raised-wrist offset from the shoulder midline, in shoulder-width units."""
        ls, rs = kps[L_SHOULDER], kps[R_SHOULDER]
        if ls[2] < self._kp_conf or rs[2] < self._kp_conf:
            return None                          # need both shoulders for a midline
        mid = 0.5 * (ls[0] + rs[0])
        width = abs(ls[0] - rs[0])
        if width < self._min_shoulder_px:        # degenerate / hallucinated
            return None
        # track the RAISED hand - the wrist higher in the frame (smaller y), by
        # height not confidence, so it never flip-flops to the resting hand.
        lw, rw = kps[L_WRIST], kps[R_WRIST]
        cand = [w for w in (lw, rw) if w[2] >= self._kp_conf]
        if not cand:
            return None
        cx = min(cand, key=lambda w: w[1])[0]
        return float((cx - mid) / width)         # shoulder-width units, signed

    def update(self, frame_rgb, now: float | None = None):
        """Run pose on one frame. Returns (x_norm in [-1,1], person_conf). Holds
        the last value (conf 0) when there is no usable detection."""
        if frame_rgb is None:
            return self.value
        try:
            out = self._sess.run(None, {self._in: self._preprocess(frame_rgb)})[0]
        except Exception:
            return self.value
        person, kps = self._best_person(out)
        sig = self._signal(kps) if person >= self._conf else None
        if sig is None:
            with self._lock:
                self._conf_last = 0.0
            return self.value
        if abs(sig) < self._deadzone:
            sig = 0.0                       # near-centre jitter -> stay put
        hx = max(-1.0, min(1.0, sig * self._gain))
        if self._mirror:
            hx = -hx
        hx = self._filt(hx, now)
        with self._lock:
            self._x = hx
            self._conf_last = person
        return hx, self._conf_last

    def debug(self, frame_rgb) -> dict:
        """Diagnostic: raw detection facts for one frame (no filtering / no hold).
        Used by scripts/hand_diag.py to VERIFY the pose sees a real hand before any
        motion is enabled."""
        if frame_rgb is None:
            return {"frame": None}
        out = self._sess.run(None, {self._in: self._preprocess(frame_rgb)})[0]
        person, kps = self._best_person(out)
        ls, rs = kps[L_SHOULDER], kps[R_SHOULDER]
        lw, rw = kps[L_WRIST], kps[R_WRIST]
        w = lw if lw[1] < rw[1] else rw
        return {
            "person": person, "sz": self._sz,
            "Lsh": (float(ls[0]), float(ls[1]), float(ls[2])),
            "Rsh": (float(rs[0]), float(rs[1]), float(rs[2])),
            "wrist": (float(w[0]), float(w[1]), float(w[2])),
            "width": float(abs(ls[0] - rs[0])),
            "min_width": self._min_shoulder_px,
            "signal": self._signal(kps),
        }

    @property
    def value(self):
        with self._lock:
            return self._x, self._conf_last

    # -- threaded mode (for HeadSystem) ----------------------------------------
    def start(self, eyes) -> None:
        self._eyes = eyes
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="hand-pose", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _frame(self):
        """Freshest camera frame - raw feed, NOT the detection-loop snapshot."""
        e = self._eyes
        if e is None:
            return None
        getter = getattr(e, "raw_frame", None) or getattr(e, "current_frame", None)
        return getter() if getter is not None else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self._frame()
            if frame is None:
                self._stop.wait(0.03)
                continue
            self.update(frame)
            # pose is the pace-setter (~8 fps); no extra sleep needed
