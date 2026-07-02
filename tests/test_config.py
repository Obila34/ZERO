from zero.config import Config, _deep_merge


def test_deep_merge_nested_override():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
    # The base must not be mutated.
    assert base["a"]["b"] == 1


def test_deep_merge_replaces_non_dicts():
    base = {"a": {"b": 1}}
    override = {"a": "flat"}
    assert _deep_merge(base, override) == {"a": "flat"}


def test_dotted_get():
    cfg = Config({"tts": {"engine": "piper", "orpheus": {"voice": "tara"}}})
    assert cfg.get("tts.engine") == "piper"
    assert cfg.get("tts.orpheus.voice") == "tara"
    assert cfg.get("tts.missing", "default") == "default"
    assert cfg.get("nope.nope") is None


def test_get_through_non_dict_returns_default():
    cfg = Config({"a": "scalar"})
    assert cfg.get("a.b", 42) == 42
