"""GazeScheduler — social gaze turn-taking. Pure logic, deterministic clock."""
from zero.head.social import GazeScheduler


def _face_ratio(sched, state, *, secs=600.0, dt=0.04, t0=1000.0):
    """Fraction of ticks spent on-face over a long simulated run in one state."""
    sched.set_state(state, t0)
    on = total = 0
    t = t0
    while t < t0 + secs:
        t += dt
        if sched.tick(t).on_face:
            on += 1
        total += 1
    return on / total


def test_idle_looks_at_partner_no_offset():
    s = GazeScheduler()
    s.set_state("idle", 100.0)
    b = s.tick(101.0)
    assert b.on_face and b.offset_deg == (0.0, 0.0)


def test_thinking_looks_up_and_away_then_returns():
    s = GazeScheduler(up_deg=10.0, think_hold_s=3.5)
    s.set_state("thinking", 100.0)
    mid = s.tick(101.0)               # within the hold
    assert not mid.on_face
    assert mid.offset_deg[1] > 0.0    # up
    assert mid.offset_deg[0] == 0.0   # not sideways
    back = s.tick(100.0 + 4.0)        # after the hold
    assert back.on_face


def test_listening_and_speaking_hit_argyle_asymmetry():
    s = GazeScheduler()
    listen = _face_ratio(s, "listening")
    speak = _face_ratio(s, "speaking")
    # ~71% listening, ~41% speaking (tolerance for episode quantisation/jitter)
    assert 0.66 <= listen <= 0.76, listen
    assert 0.36 <= speak <= 0.46, speak
    # and the asymmetry itself: less eye contact while speaking
    assert speak < listen - 0.15


def test_turn_opens_on_the_face():
    s = GazeScheduler()
    s.set_state("speaking", 100.0)
    assert s.tick(100.1).on_face      # first beat of a turn is on-face


def test_mutual_gaze_ceiling_forces_an_aversion():
    # face fraction ~1.0 would otherwise never look away; the ceiling must.
    s = GazeScheduler(max_mutual_s=5.0, rhythm={"listening": (1.0, 0.999)})
    s.set_state("listening", 100.0)
    averted = any(not s.tick(100.0 + i * 0.1).on_face for i in range(1, 70))
    assert averted   # within ~7 s, the 5 s ceiling forced a look-away


def test_sentence_end_pulls_gaze_back_to_yield_turn():
    s = GazeScheduler()
    s.set_state("speaking", 100.0)
    # force an aversion, then a sentence end should cancel it (look back)
    s._avert_until = 200.0
    s.notice_sentence_end(150.0)
    assert s.tick(150.1).on_face
