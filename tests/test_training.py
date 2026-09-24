import unittest

import numpy as np

from src import training
from src.training import FEATURE_LENGTH, HAND_FEATURE_LENGTH, SEQUENCE_LENGTH


def dataset(per_class=(10, 7, 12)):
    rng = np.random.default_rng(0)
    X = [rng.normal(0, 1, (SEQUENCE_LENGTH, FEATURE_LENGTH)).astype(np.float32)
         for n in per_class for _ in range(n)]
    y = np.array([c for c, n in enumerate(per_class) for _ in range(n)])
    return np.array(X), y


class SplitTest(unittest.TestCase):
    def test_train_and_val_do_not_overlap(self):
        _, y = dataset()
        tr, val = training.split_per_class(y, np.random.default_rng(1))
        self.assertEqual(set(tr) & set(val), set())
        self.assertEqual(len(tr) + len(val), len(y))

    def test_every_class_is_validated(self):
        _, y = dataset()
        _, val = training.split_per_class(y, np.random.default_rng(1))
        self.assertEqual(set(y[val]), set(y))

    def test_single_sample_class_stays_in_training(self):
        y = np.array([0, 0, 0, 0, 1])
        tr, val = training.split_per_class(y, np.random.default_rng(1))
        self.assertIn(4, tr)


class AugmentTest(unittest.TestCase):
    def test_shapes_and_labels(self):
        X, y = dataset((3, 3))
        Xa, ya = training.augment(X, y, np.random.default_rng(2))
        self.assertEqual(Xa.shape[1:], (SEQUENCE_LENGTH, FEATURE_LENGTH))
        self.assertEqual(len(Xa), len(ya))
        self.assertEqual(len(Xa) % len(X), 0)
        self.assertEqual(np.bincount(ya).tolist(), [len(Xa) // 2] * 2)

    def test_originals_are_kept(self):
        X, y = dataset((2,))
        Xa, _ = training.augment(X, y, np.random.default_rng(2))
        for seq in X:
            self.assertTrue(any(np.array_equal(seq, a) for a in Xa))

    def test_some_copies_have_face_removed(self):
        X, y = dataset((4,))
        Xa, _ = training.augment(X, y, np.random.default_rng(2))
        no_face = (np.abs(Xa[:, :, HAND_FEATURE_LENGTH:]).sum(axis=(1, 2)) == 0).mean()
        self.assertGreater(no_face, 0.1)
        self.assertLess(no_face, 0.4)


if __name__ == "__main__":
    unittest.main()
