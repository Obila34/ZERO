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
        self._last: dict[str, tuple[float, float]] = {}   # joint -> (ts, deg)
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
        prev = self._last.get(joint)
        if prev is not None:
            dt, dd = now - prev[0], abs(angle_deg - prev[1])
            if dd < _MIN_DELTA_DEG:
                return
            if dt < _MIN_INTERVAL_S and dd < _BIG_DELTA_DEG:
                return
        self._last[joint] = (now, float(angle_deg))
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
