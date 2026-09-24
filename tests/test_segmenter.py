import unittest

import numpy as np

from src.speech_listener import Segmenter

SR = 16000
RNG = np.random.default_rng(0)


def noise(seconds, rms=30):
    return (RNG.normal(0, rms, int(SR * seconds))).astype(np.int16)


def voice(seconds, amplitude=3000):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def run(*chunks):
    seg = Segmenter()
    out = []
    for chunk in chunks:
        out += seg.feed(chunk)
    return out


class SegmenterTest(unittest.TestCase):
    def test_room_noise_alone_gives_nothing(self):
        self.assertEqual(run(noise(3)), [])

    def test_one_word_gives_one_utterance(self):
        utts = run(noise(1), voice(0.6), noise(1))
        self.assertEqual(len(utts), 1)
        self.assertAlmostEqual(len(utts[0]) / SR, 0.6, delta=0.3)

    def test_two_words_with_pause_give_two_utterances(self):
        utts = run(noise(1), voice(0.5), noise(0.6), voice(0.5), noise(1))
        self.assertEqual(len(utts), 2)

    def test_short_click_is_ignored(self):
        self.assertEqual(run(noise(1), voice(0.08), noise(1)), [])

    def test_quiet_voice_is_detected(self):
        # Softer speaker: well above room noise but below the old 350 floor.
        utts = run(noise(1), voice(0.6, amplitude=300), noise(1))
        self.assertEqual(len(utts), 1)

    def test_nonstop_speech_is_capped(self):
        seg = Segmenter()
        utts = run(noise(1), voice(10), noise(1))
        self.assertGreaterEqual(len(utts), 2)
        for u in utts:
            self.assertLessEqual(len(u), seg.max_frames * seg.frame_len + seg.frame_len)

    def test_output_is_float_in_range(self):
        utt = run(noise(1), voice(0.6), noise(1))[0]
        self.assertEqual(utt.dtype, np.float32)
        self.assertLessEqual(np.abs(utt).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
