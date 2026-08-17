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
                 request_fps: int = 30, mjpg: bool = True,
                 device: str = "", prefer: str = ""):
        # A USB camera that browns out re-enumerates under a NEW node: the BRIO
        # went /dev/video0 -> /dev/video1 mid-session on 2026-08-17 and the run
        # died with ENODEV because the index was pinned. `device` is an explicit
        # path (a /dev/v4l/by-id/... symlink survives re-enumeration), `prefer`
        # is a case-insensitive substring of the camera's name to hunt for when
        # the configured node isn't there. Index stays the last resort.
        self._device = str(device or "")
        self._prefer = str(prefer or "")
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
        # The camera is opened INSIDE the reader thread (see _loop). Many V4L2
        # setups only deliver frames to the thread that opened the device — opening
        # here on the main thread and read()-ing on the worker returns nothing.
        self._stop.clear()
        self._open_error: Optional[Exception] = None
        self._opened = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CameraStream",
                                        daemon=True)
        self._thread.start()
        self._opened.wait(timeout=6.0)  # block until the thread opens the device
        if self._open_error is not None:
            raise self._open_error
        return self

    def _resolve_target(self):
        """What to hand VideoCapture: an explicit path, a by-id symlink for the
        preferred camera, or the plain index. Re-evaluated on every open, so a
        re-enumerated camera is picked up on reconnect instead of failing."""
        import glob
        import os

        if self._device and os.path.exists(self._device):
            return self._device
        if self._prefer:
            want = self._prefer.lower().replace(" ", "_")
            for link in sorted(glob.glob("/dev/v4l/by-id/*video-index0")):
                if want in os.path.basename(link).lower():
                    log.info("camera: matched %r -> %s", self._prefer, link)
                    return link
        if self._device:
            log.warning("camera: %s is gone — falling back to index %d",
                        self._device, self._index)
        return self._index

    def _open(self):
        import cv2

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        target = self._resolve_target()
        cap = cv2.VideoCapture(target, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {target!r}. Check the USB connection "
                f"and that nothing else holds it."
            )
        # MJPG first: USB webcams (e.g. the BRIO) often can't sustain raw YUYV at
        # 640x480/30, so read() returns nothing. MJPG is compressed and reliable.
        if self._mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._request_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # shallow buffer = freshest frame
        return cap

    def _loop(self) -> None:
        import cv2

        try:
            cap = self._open()
        except Exception as e:  # surface the open failure to start()
            self._open_error = e
            self._opened.set()
            return
        self._capture = cap
        self._opened.set()
        log.info("camera %d open @ %dx%d (mjpg=%s)", self._index, self._width,
                 self._height, self._mjpg)

        fails = 0
        got_any = False
        while not self._stop.is_set():
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                fails += 1
                # Re-open to self-heal, in two situations: no frames shortly
                # after opening (a previous run left the device busy — V4L2
                # recovers slowly after a hard kill), OR frames STOPPED after
                # working, which is what a USB brown-out looks like. The second
                # case used to be fatal: the node re-enumerated, this loop spun
                # on a dead handle forever, and the run went blind (2026-08-17).
                retry_at = ((100, 300, 700, 1500) if not got_any
                            else (60, 200, 600, 1400, 3000))
                if fails in retry_at:
                    log.warning("camera: %d failed grabs (%s) — re-opening...",
                                fails, "never delivered" if not got_any
                                else "stopped delivering")
                    try:
                        cap.release()
                        time.sleep(0.3)
                        cap = self._open()      # re-resolves the device node
                        self._capture = cap
                        log.info("camera re-opened")
                    except Exception as e:
                        log.warning("camera re-open failed: %s", e)
                        time.sleep(0.5)
                time.sleep(0.005)  # transient grab failure: back off and retry
                continue
            fails = 0
            got_any = True
            h, w = frame_bgr.shape[:2]
            if (w, h) != (self._width, self._height):
                frame_bgr = cv2.resize(frame_bgr, (self._width, self._height),
                                       interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Return the latest RGB frame (a copy), or None if none yet.

        NOTE: this ADVANCES the shared read cursor, so read_new() will block
        until the next grab. Extra consumers should use peek() instead.
        """
        with self._lock:
            if self._frame is None:
                return None
            self._last_read_id = self._frame_id
            return self._frame.copy()

    def peek(self):
        """The latest RGB frame WITHOUT touching the read cursor.

        For side consumers (face tracking, teleop) that must not starve the
        detection loop: read() marks the frame consumed, so a second reader
        calling it in a tight loop left read_new() waiting out its whole
        timeout every pass — perception ran at a crawl (2026-08-17).
        """
        with self._lock:
            return None if self._frame is None else self._frame.copy()

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
