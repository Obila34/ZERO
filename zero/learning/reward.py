"""Reward tagging — the dopamine signal, computed from what already happens.

Every conversation turn becomes an episode tagged with a scalar reward in
[-1, 1], assembled from three free signals:

  * affect       — the speaker's valence×confidence (zero/perception/affect.py)
                   while saying it: tone IS feedback.
  * explicit     — the NEXT utterance's verdict ("no, that's wrong" / "thanks,
                   perfect") retro-tags the previous reply. Strongest signal.
  * interaction  — a barge-in on the reply is negative (they cut ZERO off);
                   the conversation continuing is mildly positive (engagement).

Consumers:
  * EpisodeStore   — the tag rides on the turn episode; nightly consolidation
                     weights memory importance and training data with it.
  * InteractionPolicy — proactive outcomes feed ``record_outcome`` so
                     unsuccessful proactive kinds cool down (bandit-style).

Everything here is regex + arithmetic: ~µs per turn, safe on the hot path.
"""
from __future__ import annotations

import re
import time

from zero.utils.logging import get_logger

log = get_logger("learning.reward")

# Explicit verdicts about the PREVIOUS reply. Deliberately narrow: generic
# negativity ("this weather is awful") is affect's job, not a verdict on ZERO.
_NEG_RE = re.compile(
    r"^(no+[,.! ]|nope\b|wrong\b|that'?s (wrong|not right|not it)\b|"
    r"stop\b|shut up\b|not what i (said|asked|meant)\b|you'?re not listening\b|"
    r"i didn'?t (say|ask)\b|be quiet\b|nevermind\b|never mind\b)", re.I)
_POS_RE = re.compile(
    r"\b(thanks?( you)?|perfect|exactly|that'?s (right|it)|well done|"
    r"good (job|one)|nice one|haha+|lol|love (it|that)|awesome|brilliant)\b",
    re.I)


def score_feedback(text: str) -> float:
    """Explicit verdict in the opening of an utterance: -1, +1, or 0."""
    t = (text or "").strip().lower()
    if not t:
        return 0.0
    if _NEG_RE.search(t):
        return -1.0
    if _POS_RE.search(t):
        return 1.0
    return 0.0


class RewardTagger:
    """Stateful per-session tagger. Call order per turn (from main.py):

      on_user(text)              -> retro-tags the previous episode with any
                                    explicit verdict; resolves proactive outcome
      on_turn(user, reply, ...)  -> writes this turn's episode (provisional
                                    reward from affect/barge-in/engagement)
      on_proactive(kind)         -> a proactive utterance was just spoken; the
                                    next on_user scores it
    """

    _ENGAGE_WINDOW_S = 90.0     # a follow-up within this = engagement

    def __init__(self, episodes, policy=None):
        self._episodes = episodes
        self._policy = policy
        self._last_episode_id: int | None = None
        self._last_turn_ts = 0.0
        self._pending_proactive: str | None = None

    # ── per-utterance hooks ──────────────────────────────────────────────────
    def on_user(self, text: str) -> None:
        """The user spoke: their opening words may be a verdict on what ZERO
        just said (previous turn episode and/or a proactive utterance)."""
        verdict = score_feedback(text)
        if self._pending_proactive is not None:
            # Any reply at all means the proactive attempt landed *somewhat*;
            # an explicit verdict overrides.
            outcome = verdict if verdict != 0.0 else 0.3
            if self._policy is not None:
                try:
                    self._policy.record_outcome(self._pending_proactive, outcome)
                except Exception as e:
                    log.debug("policy outcome failed: %s", e)
            self._pending_proactive = None
        if verdict != 0.0 and self._last_episode_id is not None:
            try:
                self._episodes.tag_reward(self._last_episode_id, verdict)
            except Exception as e:
                log.debug("retro-tag failed: %s", e)
            self._last_episode_id = None      # one verdict per episode

    def on_turn(self, user_text: str, reply: str, *,
                affect=None, barged_in: bool = False,
                person_id: int | None = None) -> int | None:
        """A completed exchange -> one 'turn' episode with provisional reward."""
        now = time.time()
        reward = 0.0
        affect_payload = None
        if affect is not None and getattr(affect, "confidence", 0.0) > 0.15:
            reward += 0.4 * float(affect.valence) * float(affect.confidence)
            affect_payload = {"label": affect.label,
                              "valence": round(float(affect.valence), 3),
                              "arousal": round(float(affect.arousal), 3),
                              "confidence": round(float(affect.confidence), 3)}
        if barged_in:
            reward -= 0.5                      # they cut the reply off
        if self._last_turn_ts and now - self._last_turn_ts <= self._ENGAGE_WINDOW_S:
            reward += 0.1                      # conversation is flowing
        self._last_turn_ts = now
        reward = max(-1.0, min(1.0, reward))
        try:
            eid = self._episodes.add(
                "turn",
                {"user": (user_text or "")[:300], "reply": (reply or "")[:300],
                 "affect": affect_payload, "barged_in": bool(barged_in)},
                reward=reward, person_id=person_id, ts=now)
        except Exception as e:                 # learning must never break a turn
            log.debug("episode write failed: %s", e)
            return None
        self._last_episode_id = eid
        return eid

    def on_proactive(self, kind: str, text: str = "",
                     person_id: int | None = None) -> None:
        """ZERO spoke up on its own; log it and await the human's reaction.
        Silence (no on_user before the next on_proactive/session end) scores
        it slightly negative at resolution time."""
        self._resolve_silent_proactive()
        self._pending_proactive = kind
        try:
            self._episodes.add("proactive", {"pkind": kind, "text": text[:200]},
                               person_id=person_id)
        except Exception as e:
            log.debug("proactive episode write failed: %s", e)

    def end_session(self) -> None:
        """Session over: an unanswered proactive utterance counts against."""
        self._resolve_silent_proactive()
        self._last_episode_id = None
        self._last_turn_ts = 0.0

    def _resolve_silent_proactive(self) -> None:
        if self._pending_proactive is None:
            return
        if self._policy is not None:
            try:
                self._policy.record_outcome(self._pending_proactive, -0.2)
            except Exception as e:
                log.debug("policy outcome failed: %s", e)
        self._pending_proactive = None
