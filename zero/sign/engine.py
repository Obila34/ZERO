"""SignEngine — plays letters, spelled words and lexicon signs on the bus.

Runs on the MotionBus "sign" track, which outranks gesture and gaze: while a
word is being spelled a speech beat cannot corrupt a letter, and a lexicon
sign may later claim the head for a non-manual marker without fighting face
tracking. Every transition is a minimum-jerk ease at its own speed cap —
hand servos are quick, so signing is snappier than gesture playback, but
never a step command.

Honesty rules, same as the arm layer:
  * a letter outside the alphabet, or a sign not in the lexicon, is refused
    out loud — never approximated silently (the lexicon ships empty until a
    KSL signer fills it; see zero/sign/lexicon.py);
  * approximate letters (R, V, P, Q, Z — hardware limits) are still signed,
    because the engine SPEAKS each letter as it signs it, which keeps a
    spelled word unambiguous; describe()/capabilities() answer honestly
    when someone asks;
  * e-stop freezes signing like everything else on the bus.

Speech-sign sync: spell() returns the exact sentence the voice should say
("Spelling PETER: P - E - T - E - R.") and paces the hands at letter_s per
letter — the cadence of a spoken hyphenated readout — so the letter lands as
it is heard, the same synchrony rule the gesture layer follows for strokes.
"""
from __future__ import annotations

import threading
import time

from zero.arms import hands
from zero.arms.driver import STEPPER_JOINTS
from zero.motion.profile import PEAK, min_jerk
from zero.sign.handshapes import HANDSHAPES, capabilities, describe
from zero.sign.lexicon import load_lexicon
from zero.utils.logging import get_logger

log = get_logger("sign.engine")


def sign_prompt_block(engine: "SignEngine | None") -> str:
    """The system-prompt section teaching the sign capability — GENERATED
    from the engine's real state, never hand-written, so the model can't be
    taught a sign the robot doesn't have (the first build hardcoded a block
    that drifted from the hardware within a week)."""
    if engine is None:
        return ""
    caps = engine.capabilities()
    lex = engine.sign_names()
    lex_line = (
        "You know these KSL signs: " + ", ".join(lex) + ". "
        if lex else
        "You don't know full KSL signs yet (your sign vocabulary is being "
        "built with a KSL signer) — offer to fingerspell instead. ")
    return (
        "Sign language: you have real hands with fingers and wrists and can "
        "fingerspell in Kenyan Sign Language (KSL) using the one-handed "
        "manual alphabet on both hands. Use the arms tool for ANY request "
        "to sign, spell a word in sign, or show a letter. All 26 letters "
        "are available; " +
        ", ".join(caps["approx"]) + " are approximate because of your "
        "hands' mechanics — if asked about those, say so honestly. "
        + lex_line +
        "When you spell, the tool returns the exact sentence to say — it is "
        "paced to your hands, so never reword it.")


class SignEngine:
    def __init__(self, cfg, bus):
        self._bus = bus
        self._side = str(cfg.get("sign.side", "both")).lower()
        self._letter_s = float(cfg.get("sign.letter_s", 0.85))
        self._move_s = float(cfg.get("sign.transition_s", 0.25))
        self._hold_s = float(cfg.get("sign.single_hold_s", 3.0))
        self._max_dps = float(cfg.get("sign.max_speed_dps", 360.0))
        self._rate = float(cfg.get("sign.rate_hz", 30.0))
        self._lexicon = load_lexicon(cfg.get("sign.lexicon_path",
                                             "data/sign_lexicon.yaml"))
        # Signing stance (Phase 5, in/out live): arms rise into a forward
        # KSL stance around the letters, at a STEPPER-safe speed. Targets
        # are effective degrees with motor direction baked in (config).
        self._stance_on = bool(cfg.get("sign.stance.enabled", True))
        self._stance_move_s = float(cfg.get("sign.stance.move_s", 1.2))
        self._stance_dps = float(cfg.get("sign.stance_speed_dps", 90.0))
        self._stance_joints = {
            side: {str(j): float(v) for j, v in (targets or {}).items()}
            for side, targets in (cfg.get("sign.stance.joints") or {}).items()}
        # Belief: last commanded angle per hand joint, seeded from the bus so
        # the first ease starts from wherever the hands actually are.
        self._pose: dict[str, float] = {}
        self._gen = 0
        self._lock = threading.Lock()
        self._player: threading.Thread | None = None

    # ── vocabulary ───────────────────────────────────────────────────────────
    def knows_letter(self, ch: str) -> bool:
        return ch.upper() in HANDSHAPES

    def knows_sign(self, gloss: str) -> bool:
        return gloss.lower() in self._lexicon

    def sign_names(self) -> list[str]:
        return sorted(self._lexicon)

    def describe_letter(self, ch: str) -> str:
        return describe(ch)

    def capabilities(self) -> dict[str, list[str]]:
        return capabilities()

    # ── the spoken/tool surface ──────────────────────────────────────────────
    def letter(self, ch: str, side: str | None = None) -> str | None:
        """Sign one letter and hold it briefly. Returns the sentence to
        speak, or None when refused."""
        ch = ch.upper().strip()
        if ch not in HANDSHAPES or self._bus.estopped:
            return None
        self._start([ch], side or self._side, hold_last_s=self._hold_s)
        sides = self._sides(side or self._side)
        where = "both hands" if len(sides) == 2 else f"my {sides[0]} hand"
        return f"Signing the letter {ch} on {where}."

    def spell(self, word: str, side: str | None = None) -> str | None:
        """Fingerspell a word. Returns the letter-readout sentence the voice
        should speak in step with the hands, or None when refused."""
        letters = [c for c in word.upper() if c.isalpha()]
        if not letters or self._bus.estopped:
            return None
        unknown = [c for c in letters if c not in HANDSHAPES]
        if unknown:                       # cannot happen for A-Z, but honest
            return None
        self._start(letters, side or self._side)
        readout = " - ".join(letters)
        return f"Spelling {''.join(letters)}: {readout}."

    def sign(self, gloss: str) -> str | None:
        """Play a lexicon sign. None when the lexicon doesn't have it — the
        caller says so out loud rather than inventing a movement."""
        entry = self._lexicon.get(gloss.lower().strip())
        if entry is None or self._bus.estopped:
            return None
        self._start_segments(entry["segments"])
        return f"Signing {gloss}."

    def rest(self) -> None:
        """Ease to open hands and release the sign track."""
        with self._lock:
            self._gen += 1
            gen = self._gen
        self._player = threading.Thread(
            target=self._play_rest, args=(gen,), name="sign-rest", daemon=True)
        self._player.start()

    def stop(self) -> None:
        with self._lock:
            self._gen += 1               # kill any running playback
        self._bus.release("sign")

    def status(self) -> dict:
        caps = capabilities()
        return {"letters_exact": len(caps["exact"]),
                "letters_approx": caps["approx"],
                "lexicon": self.sign_names(),
                "side": self._side}

    # ── playback ─────────────────────────────────────────────────────────────
    def _sides(self, side: str) -> tuple[str, ...]:
        return ("left", "right") if side in ("both", "all") else (side,)

    def _letter_pose(self, ch: str, sides) -> dict[str, float]:
        ent = HANDSHAPES[ch]
        pose: dict[str, float] = {}
        for s in sides:
            pose.update(hands.hand_pose(s, ent["closure"], ent["orient"]))
        return pose

    def _seed_pose(self, joints) -> None:
        last = self._bus.last
        for j in joints:
            if j not in self._pose:
                spec = self._bus.spec(j)
                self._pose[j] = last.get(
                    j, spec.home_deg if spec is not None else 0.0)

    def _stance_pose(self, sides) -> dict[str, float]:
        """Stance targets for the signing side(s), FILTERED to joints the
        bus actually has — signing degrades to hands-only when the arm
        steppers aren't registered (arms.driver: null), silently and
        safely."""
        if not self._stance_on:
            return {}
        pose: dict[str, float] = {}
        for s in sides:
            for j, v in self._stance_joints.get(s, {}).items():
                if self._bus.spec(j) is not None:
                    pose[j] = v
        return pose

    def _start(self, letters: list[str], side: str,
               hold_last_s: float | None = None) -> None:
        sides = self._sides(side)
        frames: list[tuple[dict[str, float], float, float]] = []
        stance = self._stance_pose(sides)
        if stance:
            # Arms rise BEFORE the first handshape — preparation precedes
            # the stroke, same rule the gesture layer follows.
            frames.append((dict(stance), self._stance_move_s, 0.15))
        for ch in letters:
            pose = self._letter_pose(ch, sides)
            dwell = max(0.0, self._letter_s - self._move_s)
            frames.append((pose, self._move_s, dwell))
            for orient, secs in HANDSHAPES[ch].get("motion", []):
                sweep = dict(pose)
                for s in sides:
                    sweep[hands.wrist_name(s)] = hands.wrist_deg(s, orient)
                frames.append((sweep, secs, 0.0))
        if hold_last_s is not None and frames:
            pose, mv, _dw = frames[-1]
            frames[-1] = (pose, mv, hold_last_s)
        self._launch(frames, finish_open=True, sides=sides,
                     lower=sorted(stance))

    def _start_segments(self, segments: list[dict]) -> None:
        frames: list[tuple[dict[str, float], float, float]] = []
        touched_sides: set[str] = set()
        for seg in segments:
            pose: dict[str, float] = {}
            for s, spec in (seg.get("hands") or {}).items():
                touched_sides.add(s)
                shape = spec.get("shape")
                closure = (HANDSHAPES[str(shape).upper()]["closure"]
                           if shape is not None else spec.get("closure") or {})
                orient = spec.get("orient")
                if orient is None and shape is not None:
                    orient = HANDSHAPES[str(shape).upper()]["orient"]
                pose.update(hands.hand_pose(s, closure, orient))
            # Arm targets ride along; joints not registered on the bus (the
            # steppers, until Phase 5) are dropped by write() with a log.
            pose.update({str(j): float(v)
                         for j, v in (seg.get("arm") or {}).items()})
            frames.append((pose, float(seg.get("move_s", 0.4)),
                           float(seg.get("hold_s", 0.2))))
        self._launch(frames, finish_open=True,
                     sides=tuple(touched_sides) or ("left", "right"))

    def _launch(self, frames, *, finish_open: bool, sides,
                lower: list | None = None) -> None:
        with self._lock:
            self._gen += 1
            gen = self._gen
        self._player = threading.Thread(
            target=self._play, args=(frames, gen, finish_open, sides,
                                     lower or []),
            name="sign-play", daemon=True)
        self._player.start()

    def _play(self, frames, gen: int, finish_open: bool, sides,
              lower: list) -> None:
        dt = 1.0 / max(1.0, self._rate)
        for pose, move_s, hold_s in frames:
            if not self._ease_to(pose, move_s, gen, dt):
                return                    # preempted or e-stopped
            t_end = time.monotonic() + hold_s
            while time.monotonic() < t_end:
                with self._lock:
                    if gen != self._gen:
                        return
                if self._bus.estopped:
                    return
                time.sleep(dt)
        if finish_open:
            open_pose: dict[str, float] = {}
            for s in sides:
                open_pose.update(hands.open_hand_pose(s))
            # ...and the stance comes DOWN with the hands: every raised
            # joint eases back to its bus home (rest) in the same frame.
            for j in lower:
                spec = self._bus.spec(j)
                if spec is not None:
                    open_pose[j] = spec.home_deg
            if not self._ease_to(
                    open_pose, self._stance_move_s if lower else 0.4,
                    gen, dt):
                return
        with self._lock:
            if gen == self._gen:          # only the newest playback releases
                self._bus.release("sign")

    def _play_rest(self, gen: int) -> None:
        dt = 1.0 / max(1.0, self._rate)
        open_pose: dict[str, float] = {}
        for s in ("left", "right"):
            open_pose.update(hands.open_hand_pose(s))
        if self._ease_to(open_pose, 0.5, gen, dt):
            with self._lock:
                if gen == self._gen:
                    self._bus.release("sign")

    def _ease_to(self, target: dict[str, float], dur: float, gen: int,
                 dt: float) -> bool:
        """Minimum-jerk from the current belief to target, writing the sign
        track each step. False when preempted or e-stopped."""
        self._seed_pose(target)
        start = {j: self._pose[j] for j in target if j in self._pose}
        goal = {j: v for j, v in target.items() if j in start}
        if not goal:
            # Nothing registered (e.g. an arm-only segment before Phase 5) —
            # skip the segment rather than stall the whole sign.
            return True
        # Per-joint speed caps: hand servos are quick, but a stance move
        # rides 160:1 geared steppers — those stretch the segment to their
        # own (much slower) ceiling. The frame finishes late rather than
        # whipping an arm.
        dur = max(0.05, float(dur))
        for j in goal:
            cap = (self._stance_dps if j in STEPPER_JOINTS
                   else self._max_dps)
            dur = max(dur, PEAK * abs(goal[j] - start[j]) / max(1e-6, cap))
        t0 = time.monotonic()
        while True:
            with self._lock:
                if gen != self._gen:
                    return False
            if self._bus.estopped:
                return False
            tau = min(1.0, (time.monotonic() - t0) / dur)
            ease = min_jerk(tau)
            step = {j: start[j] + (goal[j] - start[j]) * ease for j in goal}
            self._pose.update(step)
            self._bus.write("sign", step)
            if tau >= 1.0:
                return True
            time.sleep(dt)
