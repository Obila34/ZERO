"""HttpGatewayDriver against a real in-process fake gateway — the driver was
shipped with zero tests (audit). Covers: per-axis posting, the nod servo
mapping (sign/offset/clamp), retry-until-delivered on failure (audit H1),
link-health surfacing (audit C2), and bounded-hop walking so hardware never
receives a large jump after an outage (audit H2)."""
import json
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from zero.head.driver import HttpGatewayDriver


class FakeGateway:
    def __init__(self):
        self.posts = []          # (path, body-dict) in arrival order
        self.fail = False
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else {}
                if outer.fail:
                    self.send_error(500)
                    return
                outer.posts.append((self.path, body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):   # keep pytest output clean
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def joint_angles(self, joint):
        return [b["angle_deg"] for p, b in self.posts
                if p == "/api/joint_cmd" and b["name"] == joint]

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _driver(gw, **kw):
    kw.setdefault("max_hz", 200.0)
    kw.setdefault("deadband_deg", 0.1)
    kw.setdefault("timeout_s", 1.0)
    kw.setdefault("max_jump_deg", 0.0)     # off unless the test wants it
    return HttpGatewayDriver(base_url=f"http://127.0.0.1:{gw.port}",
                             pan_joint="head_tilt_joint",
                             tilt_joint="head_nod_joint", **kw)


def _wait(cond, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_drive_nod_false_never_posts_the_nod():
    gw = FakeGateway()
    d = _driver(gw, drive_nod=False)
    try:
        d.send(10.0, 5.0)
        assert _wait(lambda: gw.joint_angles("head_tilt_joint"))
        time.sleep(0.1)
        assert gw.joint_angles("head_nod_joint") == []
    finally:
        d.close()
        gw.close()


def test_nod_mapping_sign_offset_clamp():
    gw = FakeGateway()
    d = _driver(gw, drive_nod=True, nod_sign=-1.0, nod_offset_deg=-40.0,
                nod_min_deg=-40.0, nod_max_deg=0.0)
    try:
        # tilt +10 (up) with sign -1 and offset -40 -> angle -50, clamped -40
        d.send(0.0, 10.0)
        assert _wait(lambda: gw.joint_angles("head_nod_joint"))
        assert gw.joint_angles("head_nod_joint")[-1] == -40.0
        # tilt -10 -> -10*-1 - 40 = -30: inside the window, no clamp
        d.send(0.0, -10.0)
        assert _wait(lambda: -30.0 in gw.joint_angles("head_nod_joint"))
    finally:
        d.close()
        gw.close()


def test_failed_post_is_retried_and_health_recovers():
    gw = FakeGateway()
    gw.fail = True
    d = _driver(gw, drive_nod=False)
    try:
        d.send(15.0, 0.0)
        # posts fail -> unhealthy after 3 consecutive failures
        assert _wait(lambda: not d.healthy, timeout=5.0)
        assert gw.joint_angles("head_tilt_joint") == []
        # link comes back -> the SAME command must still be delivered (H1)
        gw.fail = False
        assert _wait(lambda: 15.0 in gw.joint_angles("head_tilt_joint"),
                     timeout=5.0)
        assert _wait(lambda: d.healthy)
    finally:
        d.close()
        gw.close()


def test_bounded_hops_walk_to_a_far_target():
    gw = FakeGateway()
    d = _driver(gw, drive_nod=False, max_jump_deg=5.0)
    try:
        d.send(0.0, 0.0)                      # establish the acked pose
        assert _wait(lambda: 0.0 in gw.joint_angles("head_tilt_joint"))
        d.send(20.0, 0.0)
        assert _wait(lambda: 20.0 in gw.joint_angles("head_tilt_joint"),
                     timeout=5.0)
        angles = gw.joint_angles("head_tilt_joint")
        deltas = [abs(b - a) for a, b in zip(angles, angles[1:])]
        assert max(deltas) <= 5.0 + 1e-6      # no single hardware jump > max_jump
        assert angles[-1] == 20.0
    finally:
        d.close()
        gw.close()
