from zero.llm.persona import CUES, SYSTEM_TEMPLATE, build_system_prompt


def test_build_system_prompt_returns_template():
    prompt = build_system_prompt()
    assert prompt == SYSTEM_TEMPLATE
    assert "Zero" in prompt


def test_prompt_only_advertises_known_cues():
    # Any cue tag mentioned in the prompt must exist in the shared vocabulary,
    # otherwise the TTS layer can't translate what the LLM emits.
    import re

    for tag in re.findall(r"\[[a-z]+\]", SYSTEM_TEMPLATE):
        assert tag in CUES, f"prompt mentions {tag} but it's not in CUES"
