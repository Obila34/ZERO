#!/usr/bin/env python3
"""Fleet monitor — one page that answers "is ZERO's fleet healthy?"

Polls every service ZERO depends on, across all machines, and serves a
plain status dashboard + JSON. Born from a week of silent failures: a
disabled vLLM unit ate the brain after a reboot, a lab experiment
OOM-killed vision and whisper, the gateway 200-acked commands with dead
serial ports. Every one of those is a row here now.

    GET /            html dashboard (auto-refresh)
    GET /status.json machine-readable state
State changes are appended to fleet_monitor.log with timestamps — the
post-mortem trail the journal never gave us.

Stdlib only. Runs on zerolabs1 (tailnet-reachable from any machine):
    python server/fleet_monitor.py --port 8501
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# name, kind, target, note
CHECKS = [
    ("brain gemma4-vllm (zl1)", "http",
     "http://127.0.0.1:8001/v1/models", "LLM — every reply"),
    ("ears kyutai-stt (zl2)", "tcp", "100.100.95.12:8090",
     "live speech recognition"),
    ("voice kyutai-tts (zl2)", "tcp", "100.100.95.12:8091",
     "ZERO's voice"),
    ("hands gesture-v4 (zl2)", "http", "http://100.100.95.12:8200/health",
     "neural co-speech gestures"),
    ("sign-sense (zl1)", "http", "http://127.0.0.1:8210/health",
     "fingerspell reading"),
    ("vision (zl1)", "http", "http://127.0.0.1:8000/health",
     "depth + scene facts"),
    ("memory embedder (zl1)", "http", "http://127.0.0.1:11434/api/tags",
     "semantic recall"),
    ("AF-1 gateway (arm pi)", "http",
     "http://100.67.233.65:5000/api/telemetry", "every joint command"),
    ("head pi (ssh)", "tcp", "100.106.44.56:22", "the robot itself"),
]

GPU_HOSTS = [("zl0", "maxwell@100.95.210.94"),
             ("zl1", None),
             ("zl2", "maxwell@100.100.95.12")]

_state: dict = {"checks": {}, "gpus": {}, "updated": 0.0}
_lock = threading.Lock()
_LOG = "fleet_monitor.log"


def _http_ok(url: str) -> bool:
    try:
        with _OPENER.open(url, timeout=4) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500          # 404/405 = process is up
    except Exception:
        return False


def _tcp_ok(target: str) -> bool:
    host, port = target.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=4):
            return True
    except Exception:
        return False


def _gpu(host_ssh):
    q = ("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
         "--format=csv,noheader,nounits")
    try:
        if host_ssh:
            out = subprocess.run(["ssh", "-o", "ConnectTimeout=4",
                                  "-o", "BatchMode=yes", host_ssh, q],
                                 capture_output=True, text=True,
                                 timeout=8).stdout
        else:
            out = subprocess.run(q.split(), capture_output=True, text=True,
                                 timeout=8).stdout
        used, total, util = [int(x) for x in out.strip().split(", ")]
        return {"used_mb": used, "total_mb": total, "util_pct": util}
    except Exception:
        return None


def _log_change(name: str, up: bool) -> None:
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{'UP  ' if up else 'DOWN'} {name}\n")
    try:
        with open(_LOG, "a") as f:
            f.write(line)
    except OSError:
        pass


def _poll_loop(interval: float) -> None:
    while True:
        checks = {}
        for name, kind, target, note in CHECKS:
            up = _http_ok(target) if kind == "http" else _tcp_ok(target)
            with _lock:
                prev = _state["checks"].get(name)
            if prev is None or prev["up"] != up:
                _log_change(name, up)
                since = time.time()
            else:
                since = prev["since"]
            checks[name] = {"up": up, "note": note, "since": since}
        gpus = {label: _gpu(ssh) for label, ssh in GPU_HOSTS}
        with _lock:
            _state["checks"] = checks
            _state["gpus"] = gpus
            _state["updated"] = time.time()
        time.sleep(interval)


def _render() -> str:
    with _lock:
        s = json.loads(json.dumps(_state))
    rows = []
    for name, c in s["checks"].items():
        age = int(time.time() - c["since"])
        span = (f"{age//3600}h{(age%3600)//60:02d}m" if age >= 3600
                else f"{age//60}m{age%60:02d}s")
        cls = "up" if c["up"] else "down"
        rows.append(f"<tr class={cls}><td>{'&#9679;'}</td><td>{name}</td>"
                    f"<td>{c['note']}</td><td>{span}</td></tr>")
    gpu_rows = []
    for label, g in s["gpus"].items():
        if g is None:
            gpu_rows.append(f"<tr class=down><td>&#9679;</td>"
                            f"<td>GPU {label}</td><td>unreachable</td>"
                            f"<td></td></tr>")
        else:
            pct = 100 * g["used_mb"] // max(1, g["total_mb"])
            cls = "down" if pct > 95 else ("warn" if pct > 85 else "up")
            gpu_rows.append(
                f"<tr class={cls}><td>&#9679;</td><td>GPU {label}</td>"
                f"<td>{g['used_mb']}/{g['total_mb']} MB ({pct}%), "
                f"util {g['util_pct']}%</td><td></td></tr>")
    n_down = sum(1 for c in s["checks"].values() if not c["up"])
    title = ("ALL SYSTEMS UP" if n_down == 0
             else f"{n_down} DOWN")
    upd = time.strftime("%H:%M:%S", time.localtime(s["updated"]))
    return f"""<!doctype html><html><head>
<meta http-equiv=refresh content=10><meta charset=utf-8>
<title>ZERO fleet</title><style>
body{{font-family:system-ui,sans-serif;background:#111;color:#ddd;
margin:2em auto;max-width:44em;padding:0 1em}}
h1{{font-size:1.2em}} .ok h1{{color:#5c5}} .bad h1{{color:#e55}}
table{{width:100%;border-collapse:collapse}}
td{{padding:.45em .5em;border-bottom:1px solid #222}}
tr.up td:first-child{{color:#5c5}} tr.down td:first-child{{color:#e55}}
tr.warn td:first-child{{color:#da3}}
tr.down td{{color:#faa}} small{{color:#777}}</style></head>
<body class={'ok' if n_down == 0 else 'bad'}>
<h1>ZERO fleet &mdash; {title}</h1>
<small>updated {upd}, refreshes every 10 s &middot; state changes logged
to fleet_monitor.log</small>
<table>{''.join(rows)}{''.join(gpu_rows)}</table>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/status.json":
            with _lock:
                body = json.dumps(_state).encode()
            ctype = "application/json"
        else:
            body = _render().encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--interval", type=float, default=30.0)
    a = ap.parse_args()
    threading.Thread(target=_poll_loop, args=(a.interval,),
                     daemon=True).start()
    print(f"fleet monitor on :{a.port} (poll every {a.interval:.0f}s)")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
