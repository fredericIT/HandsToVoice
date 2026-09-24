import unittest

from src.speech_recognizer import fuzzy_match

VOCAB = {
    "mama": "mama",
    "abana": "abana",
    "yego": "yego",
    "ni": "ni",
    "oya": "oya",
    "marume": "marume",
    "kuwa_kabiri": "kuwa kabiri",
    "amazina_yange": "amazina yange",
    "uyu_munsi": "uyu munsi",
}


class FuzzyMatchTest(unittest.TestCase):
    def test_isolated_word(self):
        self.assertEqual(fuzzy_match("mama", VOCAB), ("mama", 1.0))

    def test_word_inside_sentence(self):
        self.assertEqual(fuzzy_match("ndashaka kuvuga na mama ubu", VOCAB)[0], "mama")

    def test_two_word_phrase_inside_sentence(self):
        self.assertEqual(fuzzy_match("uyu mwaka ni kuwa kabiri nk uko", VOCAB)[0], "kuwa_kabiri")

    def test_small_transcription_error_still_matches(self):
        self.assertEqual(fuzzy_match("abanna", VOCAB)[0], "abana")

    def test_short_word_matches_when_spoken_alone(self):
        self.assertEqual(fuzzy_match("ni", VOCAB), ("ni", 1.0))
        self.assertEqual(fuzzy_match("oya", VOCAB), ("oya", 1.0))

    def test_short_word_ignored_inside_sentence(self):
        # "ni" ("is") appears in ordinary sentences; it must not pop up a sign.
        self.assertIsNone(fuzzy_match("iki ni ikintu kidasanzwe", VOCAB))

    def test_unrelated_word_with_similar_letters_rejected(self):
        # "amakuru" vs "marume" scores ~0.62 — above the single-word
        # threshold, so it must be rejected by the in-sentence threshold.
        self.assertIsNone(fuzzy_match("iki ni ikintu kidasanzwe cy amakuru", VOCAB))

    def test_sentence_without_vocabulary_word(self):
        self.assertIsNone(fuzzy_match("twese turi hano turishimye none", VOCAB))

    def test_empty_text(self):
        self.assertIsNone(fuzzy_match("", VOCAB))
        self.assertIsNone(fuzzy_match("   ", VOCAB))


if __name__ == "__main__":
    unittest.main()
