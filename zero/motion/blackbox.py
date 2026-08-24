"""Joint-angle black box — every angle the robot is commanded, in SQLite.

Same philosophy as the crash black-box (fcc4518): evidence should survive
whatever happens next. The MotionBus writes every ACKNOWLEDGED post here
(what the gateway accepted, not what a producer wished), throttled so a
40 Hz gaze stream doesn't drown the file, and scripts/joint_snapshot.py can
pull the gateway's own telemetry into the same table for a ground-truth
row set (source='telemetry' vs source=<track>).

Schema:
    joint_angles(ts REAL, joint TEXT, angle_deg REAL, source TEXT)
    ts = unix seconds; angle_deg = EFFECTIVE degrees (offset-free, the
    same frame config envelopes use); source = winning track ('gaze',
    'gesture', 'sign', ...) or 'telemetry' for gateway-read rows.

Failures never propagate: a broken disk must not stop the robot moving.
"""
from __future__ import annotations

import sqlite3
import threading
import time

from zero.utils.logging import get_logger

log = get_logger("motion.blackbox")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS joint_angles (
    ts        REAL NOT NULL,
    joint     TEXT NOT NULL,
    angle_deg REAL NOT NULL,
    source    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_joint_ts ON joint_angles (joint, ts);
"""

# Throttle: a row is worth writing when the joint actually went somewhere.
_MIN_DELTA_DEG = 0.2      # below this it's deadband noise
_MIN_INTERVAL_S = 0.2     # per joint, unless the move is big
_BIG_DELTA_DEG = 5.0      # a big hop is always recorded


class JointAngleLog:
    def __init__(self, db_path: str = "zero_joints.sqlite"):
        self._path = str(db_path)
        self._lock = threading.Lock()
        # Dedupe per (joint, source): bus tracks and the telemetry sampler
        # measure in slightly different frames (head_nod especially), and a
        # shared key would make them re-log each other's values forever.
        self._last: dict[tuple[str, str], tuple[float, float]] = {}
        self._conn: sqlite3.Connection | None = None
        try:
            self._conn = sqlite3.connect(self._path,
                                         check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except Exception as e:
            log.warning("joint black box unavailable (%s) — not recording", e)
            self._conn = None

    def log(self, joint: str, angle_deg: float, source: str) -> None:
        """One acknowledged command. Throttled; never raises."""
        if self._conn is None:
            return
        now = time.time()
        prev = self._last.get((joint, source))
        if prev is not None:
            dt, dd = now - prev[0], abs(angle_deg - prev[1])
            if dd < _MIN_DELTA_DEG:
                return
            if dt < _MIN_INTERVAL_S and dd < _BIG_DELTA_DEG:
                return
        self._last[(joint, source)] = (now, float(angle_deg))
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO joint_angles VALUES (?,?,?,?)",
                    (now, joint, float(angle_deg), source))
                self._conn.commit()
        except Exception as e:
            log.debug("black box write failed: %s", e)

    def snapshot(self, angles: dict[str, float], source: str) -> int:
        """Bulk-record a full pose (e.g. gateway telemetry). Unthrottled —
        a snapshot is deliberate. Returns rows written."""
        if self._conn is None:
            return 0
        now = time.time()
        rows = [(now, j, float(a), source) for j, a in sorted(angles.items())]
        try:
            with self._lock:
                self._conn.executemany(
                    "INSERT INTO joint_angles VALUES (?,?,?,?)", rows)
                self._conn.commit()
            return len(rows)
        except Exception as e:
            log.debug("black box snapshot failed: %s", e)
            return 0

    def last_angles(self) -> dict[str, tuple[float, float, str]]:
        """Latest recorded row per joint: {joint: (angle_deg, ts, source)}."""
        if self._conn is None:
            return {}
        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT joint, angle_deg, ts, source FROM joint_angles "
                    "WHERE (joint, ts) IN (SELECT joint, MAX(ts) "
                    "FROM joint_angles GROUP BY joint)")
                return {j: (a, t, s) for j, a, t, s in cur.fetchall()}
        except Exception as e:
            log.debug("black box read failed: %s", e)
            return {}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def gateway_effective(tel: dict, offsets: dict) -> dict[str, float]:
    """Gateway telemetry + stored offsets -> {joint: effective degrees},
    EXCLUDING boot-default rows: a joint echoing raw 0.0 with the restart
    cluster's timestamp was never commanded, and raw+offset for it would be
    a fictitious angle (a bicep "at -108" that is hanging at rest). Shared
    by scripts/joint_snapshot.py and the TelemetrySampler so the two can
    never drift apart on this rule."""
    from collections import Counter

    stamps = Counter(round(float(v.get("timestamp", 0)))
                     for v in tel.values()
                     if isinstance(v, dict)
                     and float(v.get("angle_deg", 1.0)) == 0.0)
    boot = stamps.most_common(1)[0][0] if stamps else None
    if boot is not None and stamps[boot] < 3:
        boot = None
    out: dict[str, float] = {}
    for j, v in tel.items():
        if j == "null" or not isinstance(v, dict):
            continue
        raw = float(v.get("angle_deg", 0.0))
        if (raw == 0.0 and boot is not None
                and round(float(v.get("timestamp", 0))) == boot):
            continue
        out[j] = raw + float(offsets.get(j, 0.0))
    return out


class TelemetrySampler:
    """Background poll of the gateway's own telemetry into the black box.

    This is what catches motion ZERO did not command — someone driving the
    robot from the AF-1 cockpit never touches the bus, but their commands
    still echo in telemetry. Change-only (log()'s per-source dedupe), so a
    still robot costs nothing but the poll itself. Never raises; a dead
    gateway just means quiet samples until it returns."""

    def __init__(self, box: JointAngleLog, base_url: str, *,
                 period_s: float = 60.0, timeout_s: float = 5.0):
        self._box = box
        self._base = base_url.rstrip("/")
        self._period = max(5.0, float(period_s))
        self._timeout = float(timeout_s)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run,
                                        name="telemetry-sampler", daemon=True)
        self._thread.start()

    def _fetch(self, path: str):
        import json
        import urllib.request

        with urllib.request.urlopen(f"{self._base}{path}",
                                    timeout=self._timeout) as r:
            return json.loads(r.read().decode())

    def _run(self) -> None:
        while not self._stop_evt.wait(self._period):
            try:
                tel = self._fetch("/api/telemetry")
                off = {k: float(v)
                       for k, v in self._fetch("/api/calibration").items()}
            except Exception as e:
                log.debug("telemetry sample skipped: %s", e)
                continue
            for j, eff in gateway_effective(tel, off).items():
                self._box.log(j, eff, "telemetry")

    def stop(self) -> None:
        self._stop_evt.set()
