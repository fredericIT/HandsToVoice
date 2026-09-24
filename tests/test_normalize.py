import unittest

import numpy as np

from src.normalize import FEATURE_LENGTH, normalize_landmarks

RNG = np.random.default_rng(0)


def hand(offset=(0.5, 0.5), scale=0.1):
    pts = RNG.uniform(-1, 1, (21, 3)).astype(np.float32) * scale
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    return pts.reshape(-1)


class NormalizeLandmarksTest(unittest.TestCase):
    def test_output_length(self):
        self.assertEqual(normalize_landmarks(hand()).shape, (FEATURE_LENGTH,))

    def test_wrist_is_origin(self):
        out = normalize_landmarks(hand())
        np.testing.assert_allclose(out[:3], 0.0, atol=1e-6)

    def test_position_in_frame_does_not_matter(self):
        h = hand()
        moved = h.reshape(21, 3).copy()
        moved[:, :2] += 0.2
        np.testing.assert_allclose(normalize_landmarks(h)[:63],
                                   normalize_landmarks(moved.reshape(-1))[:63], atol=1e-5)

    def test_distance_from_camera_does_not_matter(self):
        h = hand().reshape(21, 3)
        far = h.copy()
        far -= far[0]
        far *= 0.5
        far += h[0]
        np.testing.assert_allclose(normalize_landmarks(h.reshape(-1))[:63],
                                   normalize_landmarks(far.reshape(-1))[:63], atol=1e-5)

    def test_no_hand_gives_zeros(self):
        out = normalize_landmarks(np.zeros(63, dtype=np.float32))
        self.assertEqual(out.shape, (FEATURE_LENGTH,))
        self.assertFalse(out.any())

    def test_face_offset(self):
        h = hand(offset=(0.6, 0.7))
        wrist = h.reshape(21, 3)[0]
        out = normalize_landmarks(h, face_ref=(0.5, 0.3, 0.2))
        self.assertAlmostEqual(out[63], (wrist[0] - 0.5) / 0.2, places=5)
        self.assertAlmostEqual(out[64], (wrist[1] - 0.3) / 0.2, places=5)

    def test_no_face_gives_zero_offset(self):
        out = normalize_landmarks(hand())
        self.assertEqual(out[63], 0.0)
        self.assertEqual(out[64], 0.0)


if __name__ == "__main__":
    unittest.main()
