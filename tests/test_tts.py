import numpy as np

from zero.tts.base import resample_linear
from zero.tts.fallback import _plain_text
from zero.tts.orchestrator import split_sentences, split_stream
from zero.tts.remote_engine import _to_orpheus


class TestSplitSentences:
    def test_basic(self):
        assert split_sentences("Hello there. How are you?") == \
            ["Hello there.", "How are you?"]

    def test_trailing_fragment_kept(self):
        parts = split_sentences("Done. And then")
        assert parts == ["Done.", "And then"]

    def test_empty(self):
        assert split_sentences("") == []

    def test_newline_ends_sentence(self):
        assert split_sentences("line one\nline two") == ["line one", "line two"]

    def test_ellipsis(self):
        assert split_sentences("Well… maybe.") == ["Well…", "maybe."]


class TestSplitStream:
    """The incremental splitter behind _speak_streaming. The load-bearing
    property is the RAW remainder: the old pattern (keep split_sentences'
    stripped tail as the buffer) ate the space between LLM chunks, and live
    replies were spoken as 'I'mstill' / 'Heythere'."""

    def test_complete_sentences_match_split_sentences(self):
        complete, rest = split_stream("Hello there. How are you?")
        assert complete == ["Hello there.", "How are you?"]
        assert rest == ""

    def test_remainder_keeps_its_whitespace(self):
        complete, rest = split_stream("Hey there! I'm ")
        assert complete == ["Hey there!"]
        assert rest == " I'm "          # NOT stripped — the join depends on it

    def test_chunk_join_does_not_lose_the_space(self):
        # Exactly the venue failure: the joining space sat at the END of a
        # chunk, and the stripped buffer turned "I'm still" into "I'mstill".
        _, buf = split_stream("Hey there! I'm ")
        buf += "still here."
        complete, buf = split_stream(buf)
        assert complete == ["I'm still here."]
        assert buf == ""

    def test_incomplete_text_stays_buffered(self):
        complete, rest = split_stream("And then")
        assert complete == []
        assert rest == "And then"

    def test_eager_first_emits_a_leading_clause(self):
        text = "Well, that's a really good question, and here's the thing"
        complete, rest = split_stream(text, eager_first=True)
        assert complete == ["Well, that's a really good question,"]
        assert rest.lstrip() == "and here's the thing"

    def test_eager_first_does_not_chop_short_openers(self):
        complete, rest = split_stream("Yes, sure", eager_first=True)
        assert complete == []
        assert rest == "Yes, sure"

    def test_eager_only_applies_when_nothing_is_complete(self):
        complete, rest = split_stream("Okay. And then, we go on",
                                      eager_first=True)
        assert complete == ["Okay."]
        assert "And then" in rest


class TestOrpheusCueTranslation:
    def test_mapped_cues(self):
        assert _to_orpheus("[laughs] hello") == "<laugh> hello"
        assert _to_orpheus("oh [sighs] fine") == "oh <sigh> fine"

    def test_unmapped_cue_stripped(self):
        assert _to_orpheus("[whistles] hi") == "hi"

    def test_emphasis_stripped(self):
        assert _to_orpheus("that is *really* good") == "that is really good"


class TestPlainText:
    def test_strips_cues_and_emphasis(self):
        assert _plain_text("[laughs] that's *really* funny [pause]") == \
            "that's really funny"


class TestResample:
    def test_identity(self):
        x = np.ones(100, dtype=np.float32)
        assert resample_linear(x, 22050, 22050) is x

    def test_length_scales(self):
        x = np.zeros(22050, dtype=np.float32)
        y = resample_linear(x, 22050, 24000)
        assert len(y) == 24000
        assert y.dtype == np.float32

    def test_empty(self):
        x = np.zeros(0, dtype=np.float32)
        assert len(resample_linear(x, 22050, 24000)) == 0
