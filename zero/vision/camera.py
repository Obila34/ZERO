"""Threaded camera capture for the always-on eyes.

``CameraStream`` opens the USB camera once and pulls frames in a background
thread, always keeping only the *latest* frame. Readers never block on the driver
and never process a stale backlog — they take whatever the most recent grab
produced. Frames are converted BGR->RGB and downscaled to the configured stream
resolution so YOLO11n keeps up on the Pi 5 CPU.

cv2/numpy are imported lazily so a box without a camera (text mode, the GPU node)
can import the package without pulling in OpenCV.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from zero.utils.logging import get_logger

log = get_logger("vision.camera")


class CameraStream:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 request_fps: int = 30, mjpg: bool = True):
        self._index = int(index)
        self._width = int(width)
        self._height = int(height)
        self._request_fps = int(request_fps)
        self._mjpg = bool(mjpg)

        self._capture = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._frame = None           # latest RGB frame (np.ndarray)
        self._frame_id = 0
        self._last_read_id = -1

    def start(self) -> "CameraStream":
        if self._thread is not None:
            return self  # already started
        import cv2

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._index, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self._index}. Check the USB "
                f"connection and that nothing else holds /dev/video{self._index}."
            )
        # MJPG FIRST: USB webcams (e.g. the Logitech BRIO) usually can't sustain
        # raw YUYV at 640x480/30 over USB bandwidth, so read() silently returns
        # nothing. MJPG is compressed and negotiates reliably. Set the fourcc
        # before the resolution.
        if self._mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._request_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # shallow buffer = freshest frame

        # Warm up: cameras commonly drop the first frames while exposure/format
        # settle. Read a few before declaring the stream healthy, so we can warn
        # loudly (with the actual format) if nothing ever arrives.
        warm = None
        for _ in range(30):
            ok, warm = cap.read()
            if ok and warm is not None:
                break
            time.sleep(0.03)
        if warm is None:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            cc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
            log.warning("camera %d opened but produced NO frame in warmup "
                        "(fourcc=%r). Try a different vision.camera.index, or set "
                        "vision.camera.mjpg: false.", self._index, cc)
        self._capture = cap

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="CameraStream",
                                        daemon=True)
        self._thread.start()
        log.info("camera %d open @ %dx%d (mjpg=%s, warmup=%s)", self._index,
                 self._width, self._height, self._mjpg,
                 "ok" if warm is not None else "no-frame")
        return self

    def _loop(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, frame_bgr = self._capture.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)  # transient grab failure: back off and retry
                continue
            h, w = frame_bgr.shape[:2]
            if (w, h) != (self._width, self._height):
                frame_bgr = cv2.resize(frame_bgr, (self._width, self._height),
                                       interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Return the latest RGB frame (a copy), or None if none yet."""
        with self._lock:
            if self._frame is None:
                return None
            self._last_read_id = self._frame_id
            return self._frame.copy()

    def read_new(self, timeout: float = 1.0):
        """Block up to ``timeout`` seconds for a frame newer than the last read."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_id != self._last_read_id:
                    self._last_read_id = self._frame_id
                    return self._frame.copy()
            time.sleep(0.002)
        return self.read()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
