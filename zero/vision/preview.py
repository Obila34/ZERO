"""Live preview of what the eyes see — a desktop window OR a browser stream.

Two sinks, one interface (``show(frame_rgb, detections)`` / ``close()`` / ``ok``):

* ``WindowPreview`` — an OpenCV ``imshow`` window. Great on a dev desktop or a Pi
  with a monitor attached; useless on a headless Pi (no display).
* ``WebPreview`` — an MJPEG stream over HTTP (stdlib only, no extra deps). Open
  ``http://<pi-ip>:<port>/`` in any browser on the network. This is the one that
  works on a **headless** Pi reached over SSH/LAN.

``build_preview(mode=...)`` picks one. ``mode="auto"`` (the default) uses a window
when a display is present and the web stream otherwise — so it "just works" on a
laptop and on a headless Pi without config changes.

Drawing (``annotate``) uses only cv2 imgproc, which the headless opencv build has
too — only ``imshow`` needs the GUI, which is why the web sink runs anywhere.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional

from zero.utils.logging import get_logger

log = get_logger("vision.preview")


def annotate(frame_rgb, detections, scale: float = 1.0, attention=None):
    """Return a BGR image with detection boxes, labels and an object count
    drawn. ``attention`` is the digital-gaze window (x, y, w, h): drawn as an
    orange box plus a picture-in-picture inset (top-right) of what it sees."""
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        x, y, w, h = (int(v) for v in det.bbox)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 220, 0), 2)
        tag = f"{det.label} {det.confidence:.2f}"
        if det.color:
            tag = f"{det.color} {tag}"
        cv2.putText(bgr, tag, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"{len(detections)} objects", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if attention is not None:
        _draw_attention(cv2, bgr, attention)
    if scale != 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST)
    return bgr


_ORANGE = (0, 150, 255)  # BGR


def _draw_attention(cv2, bgr, window) -> None:
    """Orange gaze box + a picture-in-picture inset (top-right) of the crop."""
    H, W = bgr.shape[:2]
    x, y, w, h = (int(v) for v in window)
    x, y = max(0, min(W - 1, x)), max(0, min(H - 1, y))
    w, h = max(1, min(W - x, w)), max(1, min(H - y, h))
    # Copy the crop BEFORE drawing the box, so the inset shows clean pixels
    # (and survives the window overlapping its own inset region).
    crop = bgr[y:y + h, x:x + w].copy()
    cv2.rectangle(bgr, (x, y), (x + w, y + h), _ORANGE, 2)
    cv2.putText(bgr, "attention", (x + 4, min(H - 6, y + h - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ORANGE, 1, cv2.LINE_AA)
    if crop.size == 0:
        return
    iw = max(1, int(W * 0.28))
    ih = max(1, int(iw * h / w))
    if ih > int(H * 0.45):  # keep tall windows from dominating the view
        ih = int(H * 0.45)
        iw = max(1, int(ih * w / h))
    inset = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)
    x0, y0 = W - iw - 8, 8
    bgr[y0:y0 + ih, x0:x0 + iw] = inset
    cv2.rectangle(bgr, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), _ORANGE, 1)


class WindowPreview:
    """OpenCV ``imshow`` window. Disables itself on any HighGUI error."""

    def __init__(self, title: str, scale: float = 1.0):
        self._title = title
        self._scale = scale
        self.ok = True

    def show(self, frame_rgb, detections, attention=None) -> None:
        try:
            import cv2

            cv2.imshow(self._title,
                       annotate(frame_rgb, detections, self._scale, attention))
            # Pump the GUI event loop; 'q' closes the preview (not ZERO itself).
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self.ok = False
                cv2.destroyWindow(self._title)
        except Exception as e:  # headless box / opencv-python-headless build
            self.ok = False
            log.warning("window preview disabled (no display?): %s", e)

    def close(self) -> None:
        try:
            import cv2

            cv2.destroyWindow(self._title)
        except Exception:
            pass


def _make_handler():
    from http.server import BaseHTTPRequestHandler

    _PAGE = (
        b"<!doctype html><html><head><title>ZERO - what the camera sees</title>"
        b"<style>body{margin:0;background:#111;display:flex;justify-content:center;"
        b"align-items:center;height:100vh}img{max-width:100%;max-height:100vh}"
        b"</style></head><body><img src='/stream'></body></html>"
    )

    class Handler(BaseHTTPRequestHandler):
        # BaseHTTPRequestHandler logs every request to stderr; silence it.
        def log_message(self, *args) -> None:  # noqa: D401
            return

        def do_GET(self) -> None:
            pub: "WebPreview" = self.server.publisher  # type: ignore[attr-defined]
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_PAGE)))
                self.end_headers()
                self.wfile.write(_PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            last_id = -1
            try:
                while not pub.closed:
                    last_id, jpg = pub.wait_for_frame(last_id, timeout=1.0)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed — normal

    return Handler


class WebPreview:
    """MJPEG-over-HTTP stream — viewable in a browser, works headless."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8008,
                 scale: float = 1.0, quality: int = 70):
        from http.server import ThreadingHTTPServer

        self._scale = scale
        self._quality = int(quality)
        self._latest: Optional[bytes] = None
        self._id = 0
        self._cond = threading.Condition()
        self.closed = False
        self.ok = False

        try:
            self._server = ThreadingHTTPServer((host, port), _make_handler())
        except OSError as e:
            log.warning("web preview disabled — could not bind %s:%d (%s)",
                        host, port, e)
            return
        self._server.daemon_threads = True
        self._server.publisher = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="EyesWebPreview", daemon=True)
        self._thread.start()
        self.ok = True
        shown = host if host not in ("0.0.0.0", "") else "<pi-ip>"
        log.info("camera preview streaming at http://%s:%d/  (open in a browser)",
                 shown, port)

    def wait_for_frame(self, last_id: int, timeout: float):
        """Block until a frame newer than ``last_id`` is available (or timeout)."""
        with self._cond:
            if self._id == last_id:
                self._cond.wait(timeout)
            return self._id, self._latest

    def show(self, frame_rgb, detections, attention=None) -> None:
        if not self.ok:
            return
        try:
            import cv2

            bgr = annotate(frame_rgb, detections, self._scale, attention)
            ok, buf = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                return
            with self._cond:
                self._latest = buf.tobytes()
                self._id += 1
                self._cond.notify_all()
        except Exception as e:
            log.debug("web preview frame failed: %s", e)

    def close(self) -> None:
        self.closed = True
        with self._cond:
            self._cond.notify_all()
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass


def _has_display() -> bool:
    """True if a local GUI is likely available (so a window can open)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # Windows / macOS have a native display


def build_preview(mode: str, *, title: str, scale: float,
                  host: str, port: int, jpeg_quality: int):
    """Construct the preview sink named by ``mode`` ('auto' | 'window' | 'web')."""
    mode = (mode or "auto").lower()
    if mode == "auto":
        mode = "window" if _has_display() else "web"
        log.info("preview mode 'auto' -> %s", mode)
    if mode == "web":
        return WebPreview(host=host, port=port, scale=scale, quality=jpeg_quality)
    return WindowPreview(title=title, scale=scale)
