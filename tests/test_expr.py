"""Living Hands: tap isolation, prosody-timed beats, semantic shapes,
priority subordination — all against NullTransport (nothing moves)."""
import json
import time

import numpy as np

from zero.arms import hands
from zero.expr.schedule import HandScheduler
from zero.expr.semantics import analyze
from zero.expr.tap import SpeechTap
from zero.motion.bus import BusJoint, MotionBus
from zero.motion.transport import NullTransport

SR = 24000


class FakeCfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


_BUSES = []


def teardown_module():
    for b in _BUSES:
        b.close()
    _BUSES.clear()


def _bus():
    t = NullTransport()
    bus = MotionBus(t, rate_hz=500.0)
    _BUSES.append(bus)
    for name, s in hands.hand_joint_specs().items():
        bus.register(BusJoint(name, min_deg=s["min"], max_deg=s["max"],
                              home_deg=s["home"], batch=True))
    return bus, t


def _sched(bus, over=None):
    # playout_delay 0: tests call on_playout directly with no output
    # prebuffer, so the anchor IS the audible time here.
    cfg = FakeCfg({"expression.hands.rate_hz": 100.0,
                   "expression.hands.latency_ms": 0.0,
                   "expression.hands.playout_delay_ms": 0.0,
                   "expression.hands.beat.min_gap_s": 0.1,
                   "expression.hands.idle_release_s": 0.4})
    cfg.update(over or {})
    return HandScheduler(cfg, bus)


def _stress_sentence(accent_at=0.8, dur=1.4):
    """Synthetic sentence: weak syllables with one clear stress burst."""
    x = np.zeros(int(dur * SR), dtype=np.float32)

    def place(t, f0, amp, blen):
        n = int(blen * SR)
        tt = np.arange(n) / SR
        f = f0 * (1 + 0.3 * np.sin(np.pi * tt / blen))
        sig = np.hanning(n) * amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
        i = int(t * SR)
        x[i:i + n] += sig[:len(x) - i]

    place(0.10, 140, 0.15, 0.15)
    place(0.45, 140, 0.15, 0.15)
    place(accent_at - 0.09, 180, 0.5, 0.18)      # centered on accent_at
    place(1.15, 140, 0.15, 0.15)
    return x + 0.003 * np.random.randn(len(x)).astype(np.float32)


# ── the tap: total isolation from the speech path ───────────────────────────

def test_tap_is_noop_unattached_and_swallows_listener_errors():
    tap = SpeechTap()
    tap.audio(0, "hi", np.zeros(10), SR)      # no listener: no-op, no error
    tap.playout(0, 10)

    class Bomb:
        def on_audio(self, *a):
            raise RuntimeError("boom")

        def on_playout(self, *a):
            raise RuntimeError("boom")

    tap.attach(Bomb())
    tap.audio(0, "hi", np.zeros(10), SR)      # swallowed — speech path safe
    tap.playout(0, 10)
    tap.detach()
    assert not tap.attached


# ── semantics: precision over recall ────────────────────────────────────────

def test_semantic_triggers_and_ordinary_speech_silence():
    g = analyze("there are three reasons this matters")
    assert g.kind == "count"
    # count of three: first three fingers extended, rest closed
    assert g.closure["index"] == 0.0 and g.closure["ring"] == 0.0
    assert g.closure["pinky"] == 1.0
    assert analyze("that dog is absolutely enormous").kind == "aperture"
    assert analyze("maybe, I think it could work").kind == "palm_up"
    assert analyze("No, that is wrong").kind == "negation"
    assert analyze("it grows over time you know").kind == "sweep"
    # ordinary sentences: NOTHING — over-gesturing reads as nervous
    for s in ("the weather is nice today",
              "I put the cup on the table",
              "she did not want to leave early",   # mid-sentence negation
              "let's talk about your day"):
        assert analyze(s) is None, s


# ── end-to-end: beat apex lands on the accent ───────────────────────────────

def test_beat_apex_lands_on_the_accent():
    bus, t = _bus()
    sched = _sched(bus)
    accent_at = 0.8
    audio = _stress_sentence(accent_at=accent_at)
    # synthesis side: whole sentence available ahead of playout (fast GPU)
    sched.on_audio(0, "well THAT is something", audio, SR)
    # playback side: 50 ms pieces at real-time pace
    hop = int(0.05 * SR)
    t0 = time.monotonic()
    trace = []                                   # (t, left index angle)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
        v = t.posted.get("left_indexp1_joint")
        if v is not None:
            trace.append((time.monotonic() - t0, v))
    sched.stop()
    assert trace, "beats never reached the bus"
    # apex = deepest excursion from open (90 deg)
    t_apex, v_apex = max(trace, key=lambda p: abs(p[1] - 90.0))
    assert abs(v_apex - 90.0) > 1.5, "no visible beat"
    # anchor==t0 (first playout at loop start); latency_ms=0 in this cfg
    err = abs(t_apex - accent_at)
    assert err < 0.20, f"apex missed accent by {err*1000:.0f} ms"


def test_existing_behaviors_always_outrank_the_layer():
    bus, t = _bus()
    sched = _sched(bus)
    audio = _stress_sentence()
    sched.on_audio(0, "hello there", audio, SR)
    hop = int(0.05 * SR)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        if i == hop * 6:
            # a sign starts mid-speech: it must own the hands instantly
            bus.write("sign", {"left_indexp1_joint": 0.0})
        time.sleep(0.05)
    time.sleep(0.1)
    assert t.posted["left_indexp1_joint"] == 0.0   # sign's value stands
    assert bus.owner("left_indexp1_joint") == "sign"
    sched.stop()


def test_quiet_release_frees_the_hands():
    bus, t = _bus()
    sched = _sched(bus)
    audio = _stress_sentence()
    sched.on_audio(0, "okay", audio, SR)
    for i in range(6):
        sched.on_playout(0, int(0.05 * SR))
        time.sleep(0.05)
    # speech ends; within idle_release the claim must clear
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        if bus.owner("left_indexp1_joint") is None:
            break
        time.sleep(0.05)
    assert bus.owner("left_indexp1_joint") is None
    # ...and the hands were left at their open rest
    assert abs(t.posted["left_indexp1_joint"] - 90.0) < 3.0
    sched.stop()


def test_estop_silences_the_layer():
    bus, t = _bus()
    sched = _sched(bus)
    bus.estop()
    sched.on_audio(0, "hello", _stress_sentence(), SR)
    for _ in range(4):
        sched.on_playout(0, int(0.05 * SR))
        time.sleep(0.05)
    assert t.posted == {}                     # nothing reached the wire
    sched.stop()


def test_occupied_hand_is_left_alone():
    class FakeArms:
        _hand_state = {"left": "holding", "right": "free"}

    bus, t = _bus()
    cfg = FakeCfg({"expression.hands.rate_hz": 100.0,
                   "expression.hands.latency_ms": 0.0,
                   "expression.hands.playout_delay_ms": 0.0,
                   "expression.hands.beat.min_gap_s": 0.1,
                   "expression.hands.idle_release_s": 0.4})
    sched = HandScheduler(cfg, bus, arms_provider=lambda: FakeArms())
    audio = _stress_sentence()
    sched.on_audio(0, "hi", audio, SR)
    hop = int(0.05 * SR)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    assert not any(k.startswith("left_") for k in t.posted), t.posted
    assert any(k.startswith("right_") for k in t.posted)
    sched.stop()


def test_flat_speech_produces_no_beats():
    bus, t = _bus()
    sched = _sched(bus)
    n = int(1.2 * SR)
    tt = np.arange(n) / SR
    flat = (0.2 * np.sin(2 * np.pi * 140 * tt)).astype(np.float32)
    sched.on_audio(0, "monotone sentence", flat, SR)
    for i in range(0, n, int(0.05 * SR)):
        sched.on_playout(0, int(0.05 * SR))
        time.sleep(0.05)
    sched.stop()
    # writes may occur (base pose while speaking) but no beat excursion
    idx = [p["left_indexp1_joint"] for p in t.posts
           if "left_indexp1_joint" in p]
    assert all(abs(v - 90.0) < 1.0 for v in idx), "phantom beat on monotone"


def test_build_gates_and_bit_identical_off_state():
    from zero.expr.system import build_expr
    from zero.expr.tap import TAP

    assert build_expr(FakeCfg({})) is None                       # default off
    assert build_expr(FakeCfg({"expression.hands.enabled": True})) is None
    assert not TAP.attached      # nothing attached: speech-path taps no-op


# ── the Pi incident of 2026-08-25: analysis cost must stay collapsed ────────

def test_idle_polling_is_free_and_long_sentences_stay_correct():
    """The first deploy pegged three Pi cores: find_accents ran over each
    sentence's whole growing 24 kHz buffer at the 25 Hz render tick. Polls
    with no new audio must now be length-compares, and a sentence longer
    than the analysis window must still report late accents in sentence
    coordinates (window-trim offset)."""
    from zero.expr.prosody import RollingProsody

    # long "sentence": 9 s with a clear stress at 8.0 s (past the 6 s window)
    x = np.zeros(int(9.0 * SR), dtype=np.float32)

    def place(t, f0, amp, blen):
        n = int(blen * SR)
        tt = np.arange(n) / SR
        f = f0 * (1 + 0.3 * np.sin(np.pi * tt / blen))
        sig = np.hanning(n) * amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
        i = int(t * SR)
        x[i:i + n] += sig[:len(x) - i]

    for t0 in (3.6, 4.4, 5.2, 6.0, 6.8, 7.4):
        place(t0, 140, 0.15, 0.15)
    place(7.91, 180, 0.5, 0.18)                     # accent center 8.0
    x += 0.003 * np.random.randn(len(x)).astype(np.float32)

    rp = RollingProsody(SR)
    seen = []
    for i in range(0, len(x), int(0.3 * SR)):
        rp.feed(x[i:i + int(0.3 * SR)])
        seen += rp.poll()
    time.sleep(0.6)
    seen += rp.poll()
    assert any(abs(t - 8.0) < 0.15 for t in seen), \
        f"late accent lost to window trim: {seen}"

    # completed sentence: further polling is a no-op, and FAST
    t0 = time.perf_counter()
    for _ in range(500):
        rp.poll()
    assert (time.perf_counter() - t0) < 0.01, "idle polls must be free"


def test_gestures_chain_across_sentences_via_engagement():
    """Phase D: between two gestural sentences the hands hold a warm
    ready-posture (engagement) instead of collapsing to dead rest — and
    still park fully once speech truly ends."""
    bus, t = _bus()
    sched = _sched(bus, {"expression.hands.engage_decay_s": 3.0,
                         "expression.hands.engage_closure": 0.10})
    hop = int(0.05 * SR)
    # sentence 0 with a clear accent -> a beat fires -> engagement rises
    a0 = _stress_sentence()
    sched.on_audio(0, "well THAT is something", a0, SR)
    for i in range(0, len(a0), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    # short inter-sentence gap (0.3 s): still "speaking" window
    time.sleep(0.3)
    mid_gap = t.posted.get("left_indexp1_joint", 90.0)
    # posture held: engaged closure keeps the index visibly off full-open
    assert mid_gap < 89.0, f"hands collapsed to rest between sentences ({mid_gap})"
    # sentence 1 plays; then real silence -> full park at open rest
    a1 = _stress_sentence()
    sched.on_audio(1, "and THIS one too", a1, SR)
    for i in range(0, len(a1), hop):
        sched.on_playout(1, hop)
        time.sleep(0.05)
    end = time.monotonic() + 6.0
    while time.monotonic() < end and bus.owner("left_indexp1_joint") == "idle":
        time.sleep(0.05)
    time.sleep(0.1)
    assert abs(t.posted["left_indexp1_joint"] - 90.0) < 2.0, "did not park"
    sched.stop()


def test_simultaneously_detected_accents_all_become_beats():
    """Hardware probe regression (2026-08-26): analysis returning a whole
    sentence's accents in one poll must yield a beat for EACH — the old
    spawn-time rate limiter dropped all but the first."""
    bus, t = _bus()
    sched = _sched(bus, {"expression.hands.beat.min_gap_s": 0.3})
    # one long sentence, three well-separated accents, audio fully pre-fed
    x = np.zeros(int(3.2 * SR), dtype=np.float32)

    def place(tt0, f0, amp, blen):
        n = int(blen * SR)
        ts = np.arange(n) / SR
        f = f0 * (1 + 0.3 * np.sin(np.pi * ts / blen))
        sig = np.hanning(n) * amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
        i = int(tt0 * SR)
        x[i:i + n] += sig[:len(x) - i]

    for tt in (0.2, 0.5, 1.3, 2.4):
        place(tt, 140, 0.15, 0.14)
    for a in (0.8, 1.7, 2.7):
        place(a - 0.09, 185, 0.5, 0.18)
    x += 0.003 * np.random.randn(len(x)).astype(np.float32)
    sched.on_audio(0, "one TWO three FOUR five SIX", x, SR)
    hop = int(0.05 * SR)
    for i in range(0, len(x), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    time.sleep(0.5)
    n_apex = len(sched._apex_log)
    sched.stop()
    assert n_apex >= 3, f"only {n_apex} of 3 accents became beats"


# ── Phase E: neural texture — N0 scaffolding (docs/NEURAL_GESTURES_PLAN.md) ─

def test_neural_mock_texture_drives_hands_and_semantic_composites():
    from zero.expr.neural import NeuralGestureClient

    bus, t = _bus()
    ncfg = FakeCfg({"expression.hands.neural.url": "mock"})
    client = NeuralGestureClient(ncfg)
    cfg = FakeCfg({"expression.hands.rate_hz": 100.0,
                   "expression.hands.latency_ms": 0.0,
                   "expression.hands.playout_delay_ms": 0.0,
                   "expression.hands.beat.min_gap_s": 0.1,
                   "expression.hands.idle_release_s": 0.4})
    sched = HandScheduler(cfg, bus, neural=client)
    # loud sentence with a semantic trigger ("three reasons" -> count)
    audio = _stress_sentence()
    sched.on_audio(0, "there are three reasons for that", audio, SR)
    time.sleep(0.5)                      # mock inference happens off-thread
    hop = int(0.05 * SR)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    time.sleep(0.3)
    sched.stop()
    client.stop()
    # neural texture reached the wire: index closure varies with energy
    idx = [p["left_indexp1_joint"] for p in t.posts
           if "left_indexp1_joint" in p]
    assert idx and min(idx) < 89.0, "neural texture never moved the hands"
    # the semantic COUNT still composited: counting THREE extends
    # index+middle+ring and CLOSES pinky+thumb — the pinky's curl is the
    # signature that meaning stayed rule-based under the neural texture
    pinky = [p["right_pinkyp1_joint"] for p in t.posts
             if "right_pinkyp1_joint" in p]
    assert pinky and min(pinky) < 30.0, "semantic count lost under neural"


def test_neural_dead_sidecar_falls_back_to_procedural_seamlessly():
    from zero.expr.neural import NeuralGestureClient

    bus, t = _bus()
    ncfg = FakeCfg({"expression.hands.neural.url": "http://127.0.0.1:1",
                    "expression.hands.neural.timeout_s": 0.1})
    client = NeuralGestureClient(ncfg)
    sched = _sched(bus)
    sched._neural = client
    audio = _stress_sentence()
    sched.on_audio(0, "well THAT is something", audio, SR)
    hop = int(0.05 * SR)
    for i in range(0, len(audio), hop):
        sched.on_playout(0, hop)
        time.sleep(0.05)
    sched.stop()
    client.stop()
    assert not client.healthy            # it noticed the sidecar is dead
    # ...and the procedural beat still fired regardless
    idx = [p["left_indexp1_joint"] for p in t.posts
           if "left_indexp1_joint" in p]
    assert idx and min(idx) < 89.0, "procedural fallback failed"


def test_neural_off_is_bit_identical():
    from zero.expr.neural import build_neural

    assert build_neural(FakeCfg({})) is None


def test_gesture_sidecar_server_roundtrip():
    import subprocess
    import sys as _sys
    import urllib.request as _rq

    proc = subprocess.Popen(
        [_sys.executable, "server/gesture_server.py", "--model", "mock",
         "--port", "18200"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5.0
        ok = False
        while time.monotonic() < deadline:
            try:
                with _rq.urlopen("http://127.0.0.1:18200/health",
                                 timeout=0.5) as r:
                    ok = json.loads(r.read())["ok"]
                    break
            except Exception:
                time.sleep(0.1)
        assert ok, "sidecar never came up"
        # a real client session against the real server
        from zero.expr.neural import NeuralGestureClient
        client = NeuralGestureClient(FakeCfg({
            "expression.hands.neural.url": "http://127.0.0.1:18200",
            "expression.hands.neural.timeout_s": 2.0}))
        client.feed(0, _stress_sentence(), SR)
        deadline = time.monotonic() + 4.0
        fr = None
        while time.monotonic() < deadline and fr is None:
            fr = client.frames_for(0, 0.5)
            time.sleep(0.05)
        client.stop()
        assert fr is not None and "closure" in fr, "no frames from sidecar"
        assert 0.0 <= max(fr["closure"].values()) <= 0.35
    finally:
        proc.terminate()
        proc.wait(timeout=3)
