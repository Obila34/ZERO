"""Gateway transports — how a resolved joint pose actually leaves the Pi.

The MotionBus decides WHAT to post each tick; a transport is only the wire.
Two implementations:

  HttpTransport — the AF-1 gateway on the Arm Pi. Verified surface (audit
      2026-08-24): /api/joint_cmd (single joint), /api/pose_cmd (batch),
      /api/stop, /api/calibration (stored per-joint zero offsets). Batch is
      used ONLY for joints the firmware itself batches (the PCA hand joints —
      its own fingerspell path uses pose_cmd for exactly that set); everything
      else goes joint-by-joint for exact parity with the proven head/arm
      drivers. Whether pose_cmd applies stepper offsets is UNVERIFIED — do
      not widen the batch set until that is checked on real metal (Phase 5).

  NullTransport — records every post and always succeeds. The default, and
      what every test runs against; the whole motion stack can be exercised
      with zero risk to hardware.

Stdlib only. A transport never raises into the bus tick: failures return
False and are logged at debug.
"""
from __future__ import annotations

import json
import math
import urllib.request

from zero.utils.logging import get_logger

log = get_logger("motion.transport")


class NullTransport:
    """Moves nothing; remembers what would have been posted."""

    moves_hardware = False

    def __init__(self):
        self.posted: dict[str, float] = {}   # last value per joint
        self.posts: list[dict[str, float]] = []  # every batch, in order
        self.stops = 0

    def post_joint(self, name: str, deg: float) -> bool:
        self.posted[name] = float(deg)
        self.posts.append({name: float(deg)})
        return True

    def post_pose(self, pose: dict[str, float]) -> bool:
        self.posted.update({k: float(v) for k, v in pose.items()})
        self.posts.append(dict(pose))
        return True

    def stop(self) -> bool:
        self.stops += 1
        return True

    def fetch_offsets(self) -> dict[str, float] | None:
        return {}

    def close(self) -> None:
        pass


class HttpTransport:
    """POSTs to the AF-1 gateway. MOVES THE ROBOT."""

    moves_hardware = True

    def __init__(self, *, base_url: str, timeout_s: float = 0.7):
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout_s)

    def _post(self, path: str, body: bytes) -> bool:
        req = urllib.request.Request(
            f"{self._base}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=self._timeout).close()
            return True
        except Exception as e:      # never raise into the bus tick
            log.debug("gateway post %s failed: %s", path, e)
            return False

    def post_joint(self, name: str, deg: float) -> bool:
        body = json.dumps({"name": name, "angle_deg": deg,
                           "angle_rad": deg * math.pi / 180.0}).encode()
        return self._post("/api/joint_cmd", body)

    def post_pose(self, pose: dict[str, float]) -> bool:
        body = json.dumps({"joints": pose}).encode()
        return self._post("/api/pose_cmd", body)

    def stop(self) -> bool:
        return self._post("/api/stop", b"")

    def fetch_offsets(self) -> dict[str, float] | None:
        """The gateway's stored zero offsets (stepper calibration). None on
        failure — the caller must treat the offsets as unknown, not zero."""
        try:
            with urllib.request.urlopen(f"{self._base}/api/calibration",
                                        timeout=self._timeout) as r:
                return {k: float(v)
                        for k, v in json.loads(r.read().decode()).items()}
        except Exception as e:
            log.warning("gateway offsets unavailable (%s)", e)
            return None

    def close(self) -> None:
        pass
