from zero.main import is_stop_phrase, is_visual_question


class TestStopPhrase:
    def test_plain_goodbye(self):
        assert is_stop_phrase("Goodbye")
        assert is_stop_phrase("goodbye zero.")
        assert is_stop_phrase("Okay, that's all!")
        assert is_stop_phrase("go to sleep")

    def test_mention_inside_long_sentence_is_not_stop(self):
        assert not is_stop_phrase("tell me about the movie goodbye lenin please")
        assert not is_stop_phrase("he said that's all nonsense and walked away")

    def test_word_boundary(self):
        # "bye" inside another word must not match.
        assert not is_stop_phrase("goodbyes are hard")

    def test_normal_speech_is_not_stop(self):
        assert not is_stop_phrase("what's the weather like today")


class TestVisualQuestion:
    def test_clearly_visual(self):
        assert is_visual_question("what do you see right now")
        assert is_visual_question("look at this")
        assert is_visual_question("what color is my shirt")
        assert is_visual_question("what am I holding")
        assert is_visual_question("how many people are here")
        assert is_visual_question("can you see me")
        assert is_visual_question("how do I look today")
        assert is_visual_question("have you seen my keys")

    def test_ordinary_speech_is_not_visual(self):
        # These all contain demonstratives/common words that used to false-trigger.
        assert not is_visual_question("what do you think about that")
        assert not is_visual_question("there are many ways to do it")
        assert not is_visual_question("who was the first president")
        assert not is_visual_question("tell me a story")

    def test_polysemous_words_are_not_visual(self):
        # The bare words "see", "look", "watch", "picture" are mostly non-visual
        # in speech — only their visual phrase forms should trigger.
        assert not is_visual_question("I see what you mean")
        assert not is_visual_question("I'm looking forward to it")
        assert not is_visual_question("I watched a movie yesterday")
        assert not is_visual_question("picture this scenario for a second")
        assert not is_visual_question("it looks like rain tomorrow")


class TestVisibleLabelMatching:
    def test_naming_a_detected_object_is_visual(self):
        assert is_visual_question("hand me the cup", {"cup"})
        assert is_visual_question("whose laptop is that", {"laptop", "chair"})

    def test_plural_of_detected_object(self):
        assert is_visual_question("pass me those cups", {"cup"})

    def test_head_noun_of_multiword_label(self):
        # COCO labels like "cell phone" should match the way people say them.
        assert is_visual_question("is my phone over on the desk", {"cell phone"})

    def test_undetected_object_is_not_visual(self):
        assert not is_visual_question("hand me the cup", {"laptop"})
        assert not is_visual_question("hand me the cup", set())

    def test_generic_person_label_is_ignored(self):
        # "person" is detected almost always and the word is common in abstract
        # speech — it must not route ordinary chat to the visual path.
        assert not is_visual_question("he's a really nice person", {"person"})

    def test_no_substring_false_match(self):
        # "cup" must not match inside "cupboard talk" etc. (word boundaries).
        assert not is_visual_question("I love cupcakes", {"cup"})
