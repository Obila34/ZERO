"""The reasoning-leak guard on the OpenAI-compatible engine.

A live venue session SPOKE "thought Heythere!" — the old prefix regex required
the capital letter immediately after the marker, so the spaced form slipped
through to the voice. These pin the widened behaviour, and the words that must
survive untouched.
"""
from zero.llm.openai_engine import _strip_reasoning_prefix


class TestReasoningPrefixStrip:
    def test_welded_marker_stripped(self):
        assert _strip_reasoning_prefix("thoughtIt's looking warm.") == \
            "It's looking warm."

    def test_spaced_marker_stripped(self):
        # The exact live regression: a space between marker and reply.
        assert _strip_reasoning_prefix("thought Heythere!") == "Heythere!"
        assert _strip_reasoning_prefix("thought Hey there!") == "Hey there!"

    def test_colon_form_stripped(self):
        assert _strip_reasoning_prefix("thinking: Right, so.") == "Right, so."

    def test_bare_marker_chunk_dropped(self):
        # The capital can arrive in the NEXT stream chunk — a chunk that is
        # nothing but the marker is dropped entirely.
        assert _strip_reasoning_prefix("thought") == ""
        assert _strip_reasoning_prefix(" thinking ") == ""

    def test_real_words_survive(self):
        for text in ("thoughtful people agree.",
                     "thinking about it, yes.",
                     "thought about you yesterday.",
                     "Thought you'd never ask!",   # capitalised = a real reply
                     "Hey there!"):
            assert _strip_reasoning_prefix(text) == text
