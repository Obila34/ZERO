"""HandScheduler — the Living Hands timeline, from taps to the idle track.

Consumes the SpeechTap's two signals and turns them into hand motion:

  on_audio(idx, sentence, piece)  — synthesis side, ahead of playout. The
      sentence's text is analyzed once (semantic gesture, if any); its audio
      accumulates in a RollingProsody whose accents become pending BEATS at
      sentence-relative times.
  on_playout(idx, n_samples)      — playback side. The first piece of a
      sentence anchors its wall-clock start; each piece advances the cursor.
      Pending events become absolutely-timed envelopes: apex at
      anchor + t_accent − latency, preparation starting prep_lead earlier.

A 25 Hz motion tick renders the sum of (base open-rest pose) + (micro-motion
floor) + (active gesture envelopes, raised-cosine weighted) into per-joint
degrees via the hand model, and writes the bus IDLE track — priority 0,
outranked by sign, commands, gestures and gaze by the bus's fixed
arithmetic. That subordination is the whole non-interference guarantee.

Timing degradation (streaming TTS): an accent whose lead time has already
passed when it becomes schedulable fires immediately — the literature says
lags are what perception punishes, and a ~0 ms apex is inside tolerance.
Every apex is recorded through the bus black box (source 'idle'), so
apex-vs-accent error is measurable after the fact, not guessed.

Barge-in/stop needs no extra hook: when playout ticks cease mid-sentence,
pending events for the dead sentence are dropped and the hands ease back to
the floor, then the idle claim is released after idle_release_s.
"""
from __future__ import annotations

import math
import threading
import time

from zero.arms import hands
from zero.expr.floor import MicroMotion
from zero.expr.prosody import RollingProsody
from zero.expr.semantics import analyze
from zero.utils.logging import get_logger

log = get_logger("expr.schedule")  # noqa: F401 — kept for field debugging


def _bump(x: float) -> float:
    """Smooth 0→1→0 over x in [0,1] (sin² window) — the attack/release
    shape of every envelope: no velocity step at either end."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return math.sin(math.pi * x) ** 2


class _Envelope:
    """One gesture event: weight rises to peak at apex, falls to 0 after."""

    __slots__ = ("t_apex", "attack", "decay", "closure", "wrist", "sides",
                 "wrist_swing", "peak")

    def __init__(self, t_apex: float, attack: float, decay: float,
                 closure: dict, wrist: str | None, sides,
                 wrist_swing: float = 0.0):
        self.t_apex = t_apex
        self.attack = max(0.05, attack)
        self.decay = max(0.1, decay)
        self.closure = closure          # (finger -> target closure) or deltas
        self.wrist = wrist              # symbolic orientation or None
        self.sides = tuple(sides)
        self.wrist_swing = wrist_swing  # deg, for dynamic kinds (negation)
        self.peak = 1.0                 # scaled down by turnaround()

    def weight(self, now: float) -> float:
        dt = now - self.t_apex
        if dt < -self.attack or dt > self.decay:
            return 0.0
        if dt <= 0.0:
            return self.peak * _bump(0.5 * (1.0 + dt / self.attack))
        return self.peak * _bump(0.5 * (1.0 + dt / self.decay))

    def turnaround(self, now: float) -> None:
        """Speech died under this gesture: decay out from the CURRENT
        weight. Re-timing the apex alone made a barely-started prep SNAP
        to the full shape (weight jumped to peak at dt=0)."""
        if self.t_apex <= now:
            return                       # already decaying
        self.peak = self.weight(now)
        self.t_apex = now

    def done(self, now: float) -> bool:
        return now - self.t_apex > self.decay or self.peak <= 0.001


class _Sentence:
    __slots__ = ("text", "prosody", "anchor", "cursor", "semantic",
                 "accents_pending", "semantic_scheduled")

    def __init__(self, text: str, sr: int):
        self.text = text
        self.prosody = RollingProsody(sr)
        self.anchor: float | None = None      # wall clock at first playout
        self.cursor = 0                       # samples played
        self.semantic = analyze(text)
        self.accents_pending: list[float] = []
        self.semantic_scheduled = False


class HandScheduler:
    def __init__(self, cfg, bus, *, arms_provider=None, room_provider=None):
        self._bus = bus
        self._arms = arms_provider or (lambda: None)
        self._room = room_provider or (lambda: None)
        g = lambda k, d: cfg.get(k, d)   # noqa: E731
        self._rate = float(g("expression.hands.rate_hz", 25.0))
        self._latency = float(g("expression.hands.latency_ms", 60.0)) / 1000.0
        # The playout tap fires when a piece is PULLED into the output
        # pipeline, not when it becomes audible: the player prebuffers
        # (~300 ms, audio.prebuffer_ms) and the sink adds its own latency.
        # Without this term every apex led the audible syllable by a third
        # of a second (audit expr #3). Tune from black-box-vs-ear on the
        # robot; BT sinks need more.
        self._playout_delay = float(
            g("expression.hands.playout_delay_ms", 320.0)) / 1000.0
        self._prep = float(g("expression.hands.prep_lead_ms", 250.0)) / 1000.0
        self._beat_amp = float(g("expression.hands.beat.amp", 0.09))
        self._beat_wrist = float(g("expression.hands.beat.wrist_deg", 2.0))
        self._beat_gap = float(g("expression.hands.beat.min_gap_s", 0.35))
        self._sem_on = bool(g("expression.hands.semantic.enabled", True))
        self._sem_gap = float(g("expression.hands.semantic.min_gap_s", 4.0))
        self._floor_on = bool(g("expression.hands.floor.enabled", False))
        self._floor = MicroMotion(
            closure_amp=float(g("expression.hands.floor.amp", 0.03)),
            wrist_amp_deg=float(g("expression.hands.floor.wrist_deg", 1.5)))
        self._quiet_rms = float(g("expression.hands.floor.quiet_rms", 150.0))
        self._idle_release = float(g("expression.hands.idle_release_s", 1.5))
        self._sr = 24000                     # updated from the first tap
        self._sentences: dict[int, _Sentence] = {}
        self._cur_idx: int | None = None
        self._envs: list[_Envelope] = []
        self._last_beat_t = 0.0
        self._last_sem_t = 0.0
        self._last_playout = 0.0
        self._active = False                 # currently holding an idle claim
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._apex_log: list[float] = []   # planned apex times
        self._thread = threading.Thread(target=self._run, name="living-hands",
                                        daemon=True)
        self._thread.start()

    # ── SpeechTap listener interface ─────────────────────────────────────────
    def on_audio(self, idx: int, sentence: str, piece, sr: int) -> None:
        self._sr = int(sr) or self._sr
        with self._lock:
            st = self._sentences.get(idx)
            if st is None:
                st = self._sentences[idx] = _Sentence(sentence, self._sr)
                # Bounded store, pruned relative to the sentence AT THE
                # SPEAKER — pruning on the synthesis index deleted the
                # currently-playing sentence whenever short sentences let
                # TTS run more than 4 ahead (audit expr #5).
                floor_idx = (self._cur_idx if self._cur_idx is not None
                             else idx) - 1
                for k in [k for k in self._sentences if k < floor_idx]:
                    self._sentences.pop(k, None)
            st.prosody.feed(piece)

    def on_playout(self, idx: int, n_samples: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_playout = now
            st = self._sentences.get(idx)
            if st is None:
                st = self._sentences[idx] = _Sentence("", self._sr)
            if st.anchor is None:
                st.anchor = now
                self._cur_idx = idx
                # a NEW sentence at the speaker: pending events of any PRIOR
                # sentence that never played are dead. Future sentences
                # (synthesized ahead, queued to play next) keep theirs —
                # clearing k != idx killed the beats of every sentence after
                # the first whenever TTS outran playback (audit expr #2).
                for k, other in self._sentences.items():
                    if k < idx and other.anchor is None:
                        other.accents_pending.clear()
            st.cursor += int(n_samples)

    # ── event scheduling (runs inside the tick) ──────────────────────────────
    def _schedule_from(self, st: _Sentence, now: float) -> None:
        # (prosody polling happens lock-free in the tick before this call)
        if st.anchor is None:
            return
        remaining = []
        for t_rel in st.accents_pending:
            t_apex = (st.anchor + self._playout_delay + t_rel
                      - self._latency)
            if t_apex < now - 0.15:
                continue                      # too stale even to degrade
            if t_apex - now > 10.0:
                remaining.append(t_rel)       # far future: keep pending
                continue
            self._spawn_for(st, t_rel, max(t_apex, now + 0.02), now)
        st.accents_pending = remaining

    def _spawn_for(self, st: _Sentence, t_rel: float, t_apex: float,
                   now: float) -> None:
        sem = st.semantic
        est_dur = max(st.prosody.duration_s, 0.5)
        if (self._sem_on and sem is not None and not st.semantic_scheduled
                and now - self._last_sem_t >= self._sem_gap):
            # land the semantic shape on the accent nearest its word position
            frac = sem.word_index / max(1, sem.total_words)
            if abs(t_rel - frac * est_dur) < max(0.6, 0.25 * est_dur):
                st.semantic_scheduled = True
                self._last_sem_t = now
                self._envs.append(_Envelope(
                    t_apex, self._prep, sem.hold_s,
                    dict(sem.closure), sem.wrist, sem.sides,
                    wrist_swing=(25.0 if sem.kind == "negation" else 0.0)))
                self._apex_log.append(t_apex)
                return
        if now - self._last_beat_t < self._beat_gap:
            return
        self._last_beat_t = now
        # A stale accent still RAMPS over ~0.1 s from now — an attack
        # window already mostly elapsed made the first render jump near
        # peak in one tick (audit expr #4).
        if t_apex - now < 0.1:
            t_apex = now + 0.1
        self._envs.append(_Envelope(
            t_apex, min(self._prep, max(0.1, t_apex - now)), 0.22,
            {"index": self._beat_amp, "middle": self._beat_amp * 0.8},
            None, ("left", "right"),
            wrist_swing=self._beat_wrist))
        if len(self._apex_log) > 200:      # bounded (audit expr #9)
            del self._apex_log[:100]
        self._apex_log.append(t_apex)

    # ── rendering ────────────────────────────────────────────────────────────
    def _free_sides(self) -> tuple[str, ...]:
        arms = self._arms()
        state = getattr(arms, "_hand_state", None) if arms is not None else None
        if not isinstance(state, dict):
            return hands.SIDES
        return tuple(s for s in hands.SIDES if state.get(s, "free") == "free")

    def _floor_scale(self) -> float:
        room = self._room()
        if room is None:
            return 1.0
        try:
            rms = float(room.floor())
        except Exception:
            return 1.0
        # quiet room -> servos audible on the mic -> shrink toward zero
        return max(0.0, min(1.0, rms / max(1.0, self._quiet_rms)))

    def _render(self, now: float) -> dict[str, float]:
        sides = self._free_sides()
        if not sides:
            return {}
        speaking = now - self._last_playout < 0.5
        floor_scale = self._floor_scale() if self._floor_on else 0.0
        cl_off = (self._floor.closure_offsets(now, floor_scale)
                  if floor_scale > 0 else {})
        wr_off = (self._floor.wrist_offsets(now, floor_scale)
                  if floor_scale > 0 else {})
        pose: dict[str, float] = {}
        for side in sides:
            closure = {f: cl_off.get((side, f), 0.0) for f in hands.FINGERS}
            wrist_deg = hands.wrist_deg(side, hands.REST_ORIENT) \
                + wr_off.get(side, 0.0)
            wrist_w = 0.0
            for env in self._envs:
                if side not in env.sides:
                    continue
                w = env.weight(now)
                if w <= 0.0:
                    continue
                for f, target in env.closure.items():
                    base = closure[f]
                    closure[f] = base + (target - base) * w
                if env.wrist is not None and w > wrist_w:
                    wrist_w = w
                    tgt = hands.wrist_deg(side, env.wrist)
                    rest = hands.wrist_deg(side, hands.REST_ORIENT)
                    wrist_deg = rest + (tgt - rest) * w + wr_off.get(side, 0.0)
                if env.wrist_swing:
                    # one out-and-back flick across the envelope
                    phase = (now - (env.t_apex - env.attack)) / \
                        (env.attack + env.decay)
                    wrist_deg += env.wrist_swing * math.sin(
                        math.pi * max(0.0, min(1.0, phase))) * \
                        (1 if side == "right" else -1)
            for f in hands.FINGERS:
                c = max(0.0, min(1.0, closure[f]))
                pose[hands.joint_name(side, f)] = hands.finger_deg(side, f, c)
            pose[hands.wrist_name(side)] = wrist_deg
        if not speaking and floor_scale <= 0 and not self._envs:
            return {}
        return pose

    # ── the tick ─────────────────────────────────────────────────────────────
    def _run(self) -> None:
        dt = 1.0 / max(5.0, self._rate)
        while not self._stop_evt.wait(dt):
            if self._bus.estopped:
                # Standing idle targets would be RE-POSTED by the bus the
                # moment resume() runs — a minutes-old mid-gesture pose
                # replayed onto the hands (audit expr #7). Drop everything.
                if self._active or self._envs:
                    with self._lock:
                        self._envs.clear()
                        for st in self._sentences.values():
                            st.accents_pending.clear()
                    self._bus.release("idle")
                    self._active = False
                continue
            now = time.monotonic()
            # ANALYSIS RUNS LOCK-FREE. The lock below is shared with the TTS
            # producer (on_audio) and the playback writer (on_playout);
            # holding it across a 30-100 ms find_accents on the Pi would
            # block the playback thread mid-word — the audio-underrun
            # incident class in a second disguise (audit 2026-08-25 expr #1).
            # RollingProsody.poll is safe without the scheduler lock: feed()
            # REPLACES its buffer reference, poll() snapshots it once.
            with self._lock:
                snapshot = list(self._sentences.values())
                speaking = now - self._last_playout < 0.5
            fresh: list[tuple] = []
            if speaking:
                for st in snapshot:
                    for t_rel in st.prosody.poll():
                        fresh.append((st, t_rel))
            with self._lock:
                speaking = now - self._last_playout < 0.5
                if not speaking:
                    # Playout ceased (sentence over, or barge-in): pending
                    # accents are dropped; live envelopes TURN AROUND from
                    # their CURRENT weight and decay out — never a snap to
                    # the full shape for words never spoken (audit expr #4).
                    for st in self._sentences.values():
                        st.accents_pending.clear()
                    for e in self._envs:
                        e.turnaround(now)
                else:
                    for st, t_rel in fresh:
                        st.accents_pending.append(t_rel)
                    for st in self._sentences.values():
                        self._schedule_from(st, now)
                self._envs = [e for e in self._envs if not e.done(now)]
                has_events = bool(self._envs)
            if has_events or speaking or (
                    self._floor_on and self._active):
                pose = self._render(now)
                if pose:
                    self._bus.write("idle", pose)
                    self._active = True
                    continue
            if self._active and now - self._last_playout > self._idle_release:
                # Quiet: park at open rest, CONFIRM the bus posted it, then
                # release. Write-then-release in the same breath is a race —
                # the async bus tick can find the claim already gone and the
                # park never reaches the wire (the hands would hold whatever
                # mid-decay pose the last render left).
                base: dict[str, float] = {}
                for side in self._free_sides():
                    base.update(hands.open_hand_pose(side))
                if base:
                    self._bus.write("idle", base)
                    probe = next(iter(base))
                    deadline = time.monotonic() + 0.25
                    while (time.monotonic() < deadline
                           and not self._stop_evt.is_set()):
                        if abs(self._bus.last.get(probe, 1e9)
                               - base[probe]) < 1.0:
                            break
                        time.sleep(0.01)
                if time.monotonic() - self._last_playout > self._idle_release:
                    # speech did NOT resume during the confirm wait
                    self._bus.release("idle")
                    self._active = False

    # ── lifecycle / introspection ────────────────────────────────────────────
    def apex_stats(self) -> dict:
        n = len(self._apex_log)
        return {"scheduled_apexes": n}

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)       # an in-flight tick must not re-write
        try:                          # the claim after this release
            self._bus.release("idle")
        except Exception:
            pass
