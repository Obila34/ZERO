import numpy as np

from zero.tts.base import resample_linear
from zero.tts.fallback import _plain_text
from zero.tts.orchestrator import split_sentences
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
