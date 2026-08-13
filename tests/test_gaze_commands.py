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
