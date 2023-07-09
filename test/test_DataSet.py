import unittest 

import numpy as np
from sympy import N

from src.DataSet import DataSet, ShapeNetEncoding

class TestShapeNetEncoding(unittest.TestCase):
    def setUp(self) -> None:
        coords = []
        self.n_samples = 10
        for i in range(self.n_samples):
            coords.append(np.random.normal(size=(np.random.randint(0, 100), 3)))

        self.coords = coords
        self.max_L = 5
        self.radial_params = np.arange(10, dtype=np.float32) + 1

    def test_0(self) -> None:
        """
        Tests that initialization happens without error.
        """
        x = ShapeNetEncoding(self.coords, self.max_L, self.radial_params)


    def test_multiprocessor_precompute(self) -> None:
        """
        Tests that the pre-computation is the same for multiprocessesing and single threading
        """

        x1 = ShapeNetEncoding(self.coords, self.max_L, self.radial_params, bump_width=0.1)

        x2 = ShapeNetEncoding(self.coords, self.max_L, self.radial_params, bump_width=0.1)

        x1.precompute(n_cores=None, chunksize=2, max_L=self.max_L)

        x2.precompute(n_cores=10, chunksize=2, max_L=self.max_L)

        arr_1 = np.array(x1.precompute_arrays)
        arr_2 = np.array(x2.precompute_arrays)

        self.assertTrue(np.allclose(arr_1, arr_2))

    def test_feature_matrix_ncol(self) -> None:

        x = ShapeNetEncoding(self.coords, self.max_L, self.radial_params, bump_width=0.1)
        for n_features in range(10):
            w = np.empty((n_features, self.max_L + 1, 2 * self.max_L + 1))
            self.assertEqual(x.feature_matrix_ncols(w, True), n_features + 1)

    def test_feature_matrix_row(self) -> None:

        x = ShapeNetEncoding(self.coords, self.max_L, self.radial_params, bump_width=0.1)
        n_features = 10
        w = np.random.normal(size=(n_features, self.max_L + 1, 2 * self.max_L + 1, self.radial_params.shape[0]))

        x.precompute(None, None, self.max_L)

        out = x.feature_matrix_row(0, w, False)
        out_1 = x.feature_matrix_row(0, w, True)

        self.assertIsNotNone(out)
        self.assertIsNotNone(out_1)

        self.assertTrue(np.allclose(out, out_1[:-1]))



if __name__ == "__main__":
    unittest.main()