"""Phase 3 learning loop: episodes, reward tagging, surprise, budgets,
consolidation idempotency."""
from __future__ import annotations

import time

import pytest

from zero.learning.episodes import SCHEMA_VERSION, EpisodeStore
from zero.learning.reward import RewardTagger, score_feedback


@pytest.fixture()
def store(tmp_path):
    return EpisodeStore(str(tmp_path / "ep.sqlite"))


# ── episode store ────────────────────────────────────────────────────────────
def test_episode_roundtrip_and_schema_version(store):
    eid = store.add("turn", {"user": "hi", "reply": "hey"}, reward=0.5,
                    person_id=3)
    rows = store.recent("turn")
    assert rows[0]["id"] == eid
    assert rows[0]["v"] == SCHEMA_VERSION
    assert rows[0]["payload"]["user"] == "hi"
    assert rows[0]["reward"] == 0.5


def test_retro_tag_clamps_and_overwrites(store):
    eid = store.add("turn", {}, reward=0.1)
    store.tag_reward(eid, -5.0)
    assert store.recent("turn")[0]["reward"] == -1.0


def test_consolidation_markers_are_idempotent(store):
    ids = [store.add("turn", {}) for _ in range(3)]
    assert len(store.unconsolidated()) == 3
    store.mark_consolidated(ids)
    assert store.unconsolidated() == []
    store.mark_consolidated(ids)          # re-running is harmless
    assert store.unconsolidated() == []


def test_migrations_are_stable_on_reopen(tmp_path):
    path = str(tmp_path / "ep.sqlite")
    EpisodeStore(path).add("scene", {"label": "mug"})
    again = EpisodeStore(path)            # re-running migrations must be no-op
    assert again.count() == 1


# ── reward tagging ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,score", [
    ("no, that's wrong", -1.0),
    ("stop", -1.0),
    ("you're not listening", -1.0),
    ("thanks, perfect", 1.0),
    ("haha nice one", 1.0),
    ("what's the weather tomorrow", 0.0),
    ("this weather is awful", 0.0),       # negativity ≠ a verdict on ZERO
])
def test_score_feedback(text, score):
    assert score_feedback(text) == score


class _Affect:
    def __init__(self, valence, confidence=0.8, arousal=0.5, label="excited"):
        self.valence, self.confidence = valence, confidence
        self.arousal, self.label = arousal, label


class _Policy:
    def __init__(self):
        self.outcomes = []

    def record_outcome(self, kind, reward):
        self.outcomes.append((kind, reward))


def test_turn_reward_composes_affect_and_bargein(store):
    tagger = RewardTagger(store)
    eid = tagger.on_turn("hi", "hello!", affect=_Affect(+1.0), barged_in=False)
    assert store.recent("turn")[0]["id"] == eid
    assert store.recent("turn")[0]["reward"] == pytest.approx(0.32)  # 0.4*1*0.8
    eid2 = tagger.on_turn("go on", "more...", affect=None, barged_in=True)
    r2 = [e for e in store.recent("turn") if e["id"] == eid2][0]["reward"]
    assert r2 == pytest.approx(-0.4)      # -0.5 barge-in + 0.1 engagement


def test_explicit_verdict_retro_tags_previous_turn(store):
    tagger = RewardTagger(store)
    eid = tagger.on_turn("what's 2+2", "five!", affect=None)
    tagger.on_user("no, that's wrong")
    assert [e for e in store.recent("turn")
            if e["id"] == eid][0]["reward"] == -1.0


def test_proactive_outcome_reaches_policy(store):
    policy = _Policy()
    tagger = RewardTagger(store, policy=policy)
    tagger.on_proactive("greet", "morning!")
    tagger.on_user("thanks, good morning to you")
    assert policy.outcomes == [("greet", 1.0)]
    tagger.on_proactive("remark", "nice weather")
    tagger.end_session()                  # silence -> slightly negative
    assert policy.outcomes[-1] == ("remark", -0.2)


def test_policy_cooldown_scaling():
    from zero.proactive.policy import InteractionPolicy

    p = InteractionPolicy(quiet_hours=None)
    assert p.cooldown_scale("greet") == 1.0
    for _ in range(6):
        p.record_outcome("greet", -1.0)
    assert p.cooldown_scale("greet") > 2.0     # falling flat -> back off
    for _ in range(12):
        p.record_outcome("greet", 1.0)
    assert p.cooldown_scale("greet") < 0.7     # landing -> lean in


# ── surprise ─────────────────────────────────────────────────────────────────
def test_surprise_decays_with_repetition_and_persists(tmp_path):
    from zero.world.surprise import SurprisePredictor

    path = tmp_path / "stats.json"
    pred = SurprisePredictor(str(path))
    first = pred.observe("mug", "appeared")
    for _ in range(50):
        pred.observe("mug", "appeared")
    later = pred.surprise("mug", "appeared")
    assert later < first                        # routine stops surprising
    assert pred.surprise("snake", "appeared") > later   # novelty stays high
    pred.save()
    again = SurprisePredictor(str(path))        # sense of normal persists
    assert again.surprise("mug", "appeared") == pytest.approx(later, abs=0.2)


def test_surprise_gate_routes_events(tmp_path):
    from zero.world.state import WorldEvent, WorldState
    from zero.world.surprise import SurpriseGate, SurprisePredictor

    class _Narrator:
        pokes = 0

        def poke(self):
            self.pokes += 1

    world = WorldState()
    store = EpisodeStore(str(tmp_path / "ep.sqlite"))
    narrator = _Narrator()
    gate = SurpriseGate(world, SurprisePredictor(None), episodes=store,
                        narrator=narrator, remember_bits=0.5, narrate_bits=0.5)
    gate.start()
    try:
        world.update_objects([], [WorldEvent("appeared", "snake", time.time())])
        t0 = time.time()
        while gate.scored < 1 and time.time() - t0 < 3:
            time.sleep(0.01)
    finally:
        gate.stop()
    assert gate.scored == 1
    assert narrator.pokes == 1
    scenes = store.recent("scene")
    assert scenes and scenes[0]["payload"]["label"] == "snake"
    assert scenes[0]["surprise"] > 0


# ── budgets ──────────────────────────────────────────────────────────────────
def test_duty_budget_caps_and_recovers():
    from zero.world.budget import DutyBudget

    b = DutyBudget(max_duty=0.5, window_s=10.0)
    t = 100.0
    assert b.allowed(t)
    b.record(6.0, t)                      # 6s of work in a 10s window = 60%
    assert not b.allowed(t + 1)
    assert b.rejections == 1
    assert b.allowed(t + 11)              # window slid past the spend


def test_rate_budget_caps_per_minute():
    from zero.world.budget import RateBudget

    b = RateBudget(max_per_min=3)
    t = 100.0
    assert all(b.allowed(t + i) for i in range(3))
    assert not b.allowed(t + 4)
    assert b.allowed(t + 61)              # a minute later


# ── consolidation (distill + idempotency, real stores on tmp paths) ──────────
def test_consolidation_distill_is_idempotent(tmp_path, monkeypatch):
    import scripts.consolidate as cons

    class _Cfg:
        def __init__(self, tmp):
            self._tmp = tmp

        def get(self, k, d=None):
            return {"memory.enabled": True,
                    "memory.embeddings.backend": "hash"}.get(k, d)

        def resolve_path(self, k, d=None):
            return self._tmp / (d or k.replace(".", "_"))

    cfg = _Cfg(tmp_path)
    store = EpisodeStore(str(tmp_path / "zero_episodes.sqlite"))
    store.add("turn", {"user": "hi", "reply": "hey"}, reward=0.8, person_id=1)
    store.add("scene", {"event": "appeared", "label": "snake"}, surprise=9.0)

    # build_memory reads many keys; stub it with a recording fake instead.
    class _Mem:
        def __init__(self):
            self.episodes = []

        def add_episode(self, summary, person_id=None, importance=4.0):
            self.episodes.append((summary, person_id, importance))

    mem = _Mem()
    monkeypatch.setattr("zero.factory.build_memory", lambda c: mem)
    out1 = cons.step_distill(cfg, dry=False)
    assert "2 memories" in out1
    assert any("went well" in s for s, _, _ in mem.episodes)
    assert any("unusual" in s for s, _, _ in mem.episodes)
    high = [imp for s, _, imp in mem.episodes if "unusual" in s][0]
    assert high > 6.0                      # surprise raised importance
    out2 = cons.step_distill(cfg, dry=False)   # second run: nothing left
    assert "no unconsolidated" in out2
