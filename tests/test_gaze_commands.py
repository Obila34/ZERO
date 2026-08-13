"""Fast lexical gaze-command parser — high precision, no false fires."""
from zero.head.commands import DEFAULT_DEG, parse_gaze_command as P


def test_bare_directions():
    assert P("look left") == {"kind": "direction", "axis": "pan", "sign": -1.0,
                              "degrees": DEFAULT_DEG}
    assert P("turn to your right")["axis"] == "pan"
    assert P("turn to your right")["sign"] == 1.0
    assert P("look up")["axis"] == "tilt" and P("look up")["sign"] == 1.0
    assert P("glance down")["sign"] == -1.0


def test_explicit_degrees():
    r = P("turn left 30 degrees")
    assert r["axis"] == "pan" and r["sign"] == -1.0 and r["degrees"] == 30.0
    assert P("look 15° up")["degrees"] == 15.0


def test_full_range_commands():
    from zero.head.commands import FULL_DEG
    for phrase in ["turn head to its complete left", "look all the way left",
                   "turn completely left", "look as far left as you can"]:
        r = P(phrase)
        assert r["axis"] == "pan" and r["sign"] == -1.0, phrase
        assert r["degrees"] == FULL_DEG, phrase          # clamps to the ±limit
    assert P("look all the way right")["sign"] == 1.0
    assert P("look all the way right")["degrees"] == FULL_DEG


def test_center_and_forward():
    for phrase in ["look forward", "face forward", "look straight ahead",
                   "turn back to center", "look centre", "face front"]:
        assert P(phrase) == {"kind": "center"}, phrase


def test_look_at_me():
    assert P("look at me") == {"kind": "person", "name": "me"}
    assert P("turn back to me")["name"] == "me"


def test_look_at_name():
    assert P("look at Sam") == {"kind": "person", "name": "sam"}
    assert P("turn to look at Greg")["name"] == "greg"


def test_non_commands_do_not_fire():
    for phrase in ["look, I think we should go", "I looked everywhere",
                   "what's on the table over there?", "how are you today",
                   "let me look that up", "look at that!", "you look great"]:
        assert P(phrase) is None, phrase


def test_ambiguous_targets_are_not_names():
    # 'look at it/that/there' are not people — leave them for the LLM
    assert P("look at it") is None
    assert P("look at that") is None
    assert P("look at there") is None


def test_hold_still_fires_on_short_imperatives():
    for phrase in ["stop", "freeze", "hold on", "hold still", "stay put",
                   "don't move", "okay stop"]:
        assert P(phrase) == {"kind": "hold"}, phrase


def test_hold_fires_with_head_motion_context():
    for phrase in ["stop moving your head please", "please hold your head still",
                   "stop turning your head around now"]:
        assert P(phrase) == {"kind": "hold"}, phrase


def test_stop_buried_in_a_long_sentence_does_not_hijack():
    # audit M1: these used to return a head-hold and eat the whole turn
    for phrase in ["hold on, what's the time right now?",
                   "we should stop for lunch before the meeting",
                   "can you stop the timer please",
                   "stop joking around with everyone",
                   "stop the music for a second"]:
        assert P(phrase) is None, phrase


def test_turn_it_up_is_not_a_gaze_command():
    # audit M1: "it" was a filler word, so volume talk moved the head
    assert P("turn it up") is None
    assert P("turn it down") is None
    assert P("turn it up a bit please") is None
    # but real gaze phrasings still parse
    assert P("turn your head up")["axis"] == "tilt"
    assert P("look up")["sign"] == 1.0
