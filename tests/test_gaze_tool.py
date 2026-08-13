"""GazeTool + HeadSystem command application. Headless, NullDriver, no threads."""
import numpy as np

from zero.head.system import HeadSystem
from zero.tools.base import ToolContext
from zero.tools.gaze import GazeTool


class FakeCfg:
    def __init__(self, over=None):
        self._o = over or {}

    def get(self, k, d=None):
        return self._o.get(k, d)


class FakeEyes:
    def __init__(self):
        self.suppressed = 0

    def attention(self):
        return None

    def current_frame(self):
        return None

    def suppress_changes(self, duration_s=0.6, until=None):
        self.suppressed += 1

    def resettle(self):
        pass


def _head():
    return HeadSystem(FakeCfg(), eyes=FakeEyes())


def _ctx(head):
    return ToolContext(extras={"head": head})


def test_tool_missing_head_is_graceful():
    assert "can't move" in GazeTool().run({"target": "left"}, ToolContext()).lower()


def test_tool_direction_sets_command_target():
    h = _head()
    msg = GazeTool().run({"target": "left", "degrees": 30}, _ctx(h))
    assert "left" in msg.lower()
    assert h._cmd_target == (-30.0, 0.0)      # egocentric, absolute from home


def test_tool_center():
    h = _head()
    GazeTool().run({"target": "center"}, _ctx(h))
    assert h._cmd_target == (0.0, 0.0)


def test_tool_text_fast_path():
    h = _head()
    msg = GazeTool().run({"text": "look up"}, _ctx(h))
    assert h._cmd_target == (0.0, 25.0)       # default 25° up
    assert "up" in msg.lower()


def test_then_answer_settles_and_prompts_model():
    h = _head()
    h.start()          # controller thread runs so the neck actually reaches target
    try:
        out = GazeTool().run({"target": "right", "then_answer": True}, _ctx(h))
    finally:
        h.stop()
    assert "describe" in out.lower()          # model is told to look then answer


def test_directed_command_holds_until_released():
    import time
    h = _head()
    h.look_direction("pan", -20.0)
    assert h._cmd_target == (-20.0, 0.0)      # override is set
    assert h._cmd_hold is True                # directed commands HOLD (turn + stay)
    time.sleep(0.01)
    h._source_tick(time.monotonic())          # does NOT expire — stays put
    assert h._cmd_target == (-20.0, 0.0)
    h.resume_tracking()                       # 'follow me' releases the hold
    assert h._cmd_target is None
    assert h._cmd_hold is False


def test_hold_and_track_intents():
    h = _head()
    assert h.apply_command({"kind": "hold"}).lower().startswith("okay")
    assert h._cmd_hold is True                # 'freeze' holds where it is
    assert h.apply_command({"kind": "track"}).lower().startswith("okay")
    assert h._cmd_target is None and h._cmd_hold is False
