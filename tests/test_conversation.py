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


# ── in-session compaction (rolling summary) ──────────────────────────────────
def test_trim_holds_dropped_turns_as_pending():
    # Trimming no longer discards old turns — they're held for summarisation, so
    # nothing is lost from context before the summary lands.
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    _fill_turns(convo, 12)
    snap = convo.pending_snapshot()
    assert snap is not None
    prev, pending = snap
    assert prev == "" and pending and pending[0]["role"] == "user"
    # The very first turn is still present in the rendered context (via pending).
    assert any(m.get("content") == "user 0" for m in convo.messages())


def test_compaction_keeps_context_bounded():
    # Draining pending each turn (as the main loop does) keeps the prompt small
    # no matter how long the session runs — this is what stops the GPU filling.
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    convo.set_memory("MEM")
    bound = 1 + convo.trim_at_messages + convo.target_messages  # sys + live + transient
    for i in range(40):
        convo.add_user(f"u{i}")
        convo.add_assistant(f"a{i}")
        snap = convo.pending_snapshot()
        if snap is not None:
            convo.apply_summary("ROLLING SUMMARY", snap[1])
        assert len(convo.messages()) <= bound
    sys_msg = convo.messages()[0]["content"]
    assert "ROLLING SUMMARY" in sys_msg and "MEM" in sys_msg
    assert convo.pending_snapshot() is None


def test_reset_clears_summary_and_pending():
    convo = Conversation("s", history_turns=3, trim_at_turns=8)
    _fill_turns(convo, 20)
    convo.apply_summary("S", convo.pending_snapshot()[1])
    convo.reset()
    assert convo._history == []
    assert convo.pending_snapshot() is None
    assert "S" not in convo.messages()[0]["content"]
