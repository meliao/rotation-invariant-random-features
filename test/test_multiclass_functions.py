import unittest 

import numpy as np

from src.multiclass_functions import _find_positive_class, _least_squares_soln, compute_top_k_accuracy

class TestFindPositiveClass(unittest.TestCase):

    def test_0(self) -> None:
        """
        Starts with 1s and -1s
        """

        labels = np.array([-1, -1, 1, 1, -1])

        expected_0 = np.array([0, 0, 1, 1, 0])

        expected_1 = np.array([1, 1, 0, 0, 1])

        out_0 = _find_positive_class(1, labels)

        out_1 = _find_positive_class(-1, labels)

        self.assertEqual(out_0.dtype, np.int64)
        self.assertEqual(out_1.dtype, np.int64)

        self.assertTrue(np.allclose(expected_0, out_0))
        self.assertTrue(np.allclose(expected_1, out_1))


class TestComputeTopKAccuracy(unittest.TestCase):
    def test_0(self) -> None:
        features = np.zeros((4, 6))
        features[:, :4] = np.eye(4)

        weights_dd = {0: np.array([1, 0, 0, 0, 0, 0]),
                        1: np.array([0, 1, 0, 0, 0, 0]),
                        2: np.array([0, 0, 0.5, 0.5, 0, 0]),}
        true_labels = np.array([0, 1, 2, 2])

        self.assertEqual(1., compute_top_k_accuracy(features, weights_dd, true_labels, k=[4])[0])
        self.assertEqual(1., compute_top_k_accuracy(features, weights_dd, true_labels, k=[1])[0])

        weights_dd = {0: np.array([1, 0.85, 0, 0, 0, 0]),
                        1: np.array([0.75, 0.5, 0, 0, 0, 0]),
                        2: np.array([0, 0, 0.5, 0.5, 0, 0]),}
        true_labels = np.array([0, 1, 2, 2])
        self.assertEqual(0.75, compute_top_k_accuracy(features, weights_dd, true_labels, k=[1])[0])
        self.assertEqual(1., compute_top_k_accuracy(features, weights_dd, true_labels, k=[2])[0])


if __name__ == '__main__':
    unittest.main()