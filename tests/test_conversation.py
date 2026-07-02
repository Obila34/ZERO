from zero.conversation import Conversation


def _fill_turns(convo: Conversation, n: int) -> None:
    for i in range(n):
        convo.add_user(f"user {i}")
        convo.add_assistant(f"assistant {i}")


def test_messages_prepends_system_and_memory():
    convo = Conversation("SYSTEM", history_turns=3, trim_at_turns=8)
    convo.set_memory("MEMORY BLOCK")
    convo.add_user("hi")
    msgs = convo.messages()
    assert msgs[0]["role"] == "system"
    assert "SYSTEM" in msgs[0]["content"]
    assert "MEMORY BLOCK" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_append_only_until_trigger():
    # The cache-friendliness invariant: below the trigger, nothing is dropped.
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    _fill_turns(convo, 8)  # exactly at the trigger (16 messages)
    assert len(convo._history) == 16


def test_trims_back_to_target_past_trigger():
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    _fill_turns(convo, 9)  # crosses the trigger
    assert len(convo._history) <= 3 * 2


def test_trimmed_history_starts_with_user_turn():
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    _fill_turns(convo, 20)
    assert convo._history[0]["role"] == "user"


def test_reset_clears_history_keeps_system():
    convo = Conversation("s")
    convo.add_user("hi")
    convo.reset()
    assert convo._history == []
    assert convo.messages()[0]["role"] == "system"


def test_transcript_format():
    convo = Conversation("s")
    convo.add_user("hello")
    convo.add_assistant("hey there")
    assert convo.transcript() == "user: hello\nassistant: hey there"
