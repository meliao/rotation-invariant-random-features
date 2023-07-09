import pickle
import unittest
import numpy as np
from scipy import stats
import os

from src.data.DataClasses import DataSample3D
from src.atom_encoding import (CHARGES_LIST_QM7, CHARGES_LIST_QM9, CHARGE_PAIRS_LIST, N_CHARGE_PAIRS, N_CHARGES,
                                FCHLEncoding, 
                                MolleculeDatasetEncoding, 
                                ElementPairsDatasetEncoding)

from utils import check_scalars_close, check_array_equality, check_no_nan_in_array


class TestConstants(unittest.TestCase):
    def test_0(self) -> None:
        self.assertEqual(len(CHARGES_LIST_QM9), 5, CHARGES_LIST_QM9)
        self.assertEqual(len(CHARGES_LIST_QM7), 5, CHARGES_LIST_QM7)
        x = CHARGE_PAIRS_LIST
        self.assertEqual(len(x), 10)
        for i in x:
            self.assertEqual(len(i), 2)

class TestMolleculeDatasetEncoding(unittest.TestCase):
    def setUp(self) -> None:
        charge_lst = [i for i in CHARGES_LIST_QM7] #copy
        charge_lst.append(0)
        self.n_samples = 10
        self.max_n_atoms = 23
        self.charges = np.random.choice(charge_lst, size=(self.n_samples, self.max_n_atoms))
        self.points = np.random.normal(size=(self.n_samples, self.max_n_atoms, 3))

    def test_init(self) -> None:
        x = MolleculeDatasetEncoding(self.points, self.charges)
        self.assertIsInstance(x, MolleculeDatasetEncoding)

    def test_bad_init(self) -> None:
        with self.assertRaises(AssertionError) as context:
            x = MolleculeDatasetEncoding(self.points, self.charges[:4])

    def test_len(self) -> None:
        x = MolleculeDatasetEncoding(self.points, self.charges)
        self.assertEqual(len(x), self.n_samples)
        self.assertEqual(len(x), x.n_samples)


class TestElementPairsDatasetEncoding(unittest.TestCase):
    def setUp(self) -> None:
        charge_lst = [i for i in CHARGES_LIST_QM7]
        charge_lst.append(0)
        self.n_samples = 10
        self.max_n_atoms = 23
        self.charges = np.random.choice(charge_lst, size=(self.n_samples, self.max_n_atoms))
        self.points = np.random.normal(size=(self.n_samples, self.max_n_atoms, 3))

    def test_init(self) -> None:
        x = ElementPairsDatasetEncoding(self.points, self.charges)
        self.assertIsInstance(x, ElementPairsDatasetEncoding)

    def test_iter(self) -> None:
        x = ElementPairsDatasetEncoding(self.points, self.charges)
        for i in range(self.n_samples):
            i_lst = [j for j in x[i]]
            self.assertEqual(len(i_lst), N_CHARGE_PAIRS)
            for ds_j in i_lst:
                self.assertIsInstance(ds_j, DataSample3D)
                bool_is_nan = np.isnan(ds_j.coords).any()
                self.assertFalse(bool_is_nan)

    def test_zero_len_coords(self) -> None:
        dummy_charges = np.zeros((self.n_samples, self.max_n_atoms))
        dummy_coords = np.random.normal(size=(self.n_samples, self.max_n_atoms, 3))
        x = ElementPairsDatasetEncoding(dummy_coords, dummy_charges)

        for i in range(self.n_samples):
            i_lst = [j for j in x[i]]
            self.assertEqual(len(i_lst), N_CHARGE_PAIRS)
            for ds_j in i_lst:
                self.assertIsInstance(ds_j, DataSample3D)
                bool_is_nan = np.isnan(ds_j.coords).any()
                self.assertFalse(bool_is_nan)



    def test_get_ds_lst(self) -> None:
        x = ElementPairsDatasetEncoding(self.points, self.charges)
        y = x.get_list_of_samples()
        self.assertEqual(len(y), len(CHARGE_PAIRS_LIST) * self.n_samples)
        self.assertIsInstance(y[0], DataSample3D)
        self.assertIsInstance(y[-1], DataSample3D)

    def test_check_ds_lst_shape_equivalence(self) -> None:
        x = ElementPairsDatasetEncoding(self.points, self.charges)
        xx = ElementPairsDatasetEncoding(self.points, self.charges)
        y = x.get_list_of_samples()
        yy = xx.get_list_of_samples()
        for i in range(len(y)):
            ds_y = y[i]
            ds_yy = yy[i]
            self.assertTupleEqual(ds_y.coords.shape, ds_yy.coords.shape)


class TestFCHLEncoding(unittest.TestCase):
    def setUp(self) -> None:
        charge_lst = [i for i in CHARGES_LIST_QM7]
        charge_lst.append(0)
        self.n_samples = 10
        self.max_n_atoms = 23
        self.charges = np.random.choice(charge_lst, size=(self.n_samples, self.max_n_atoms))
        self.points = np.random.normal(size=(self.n_samples, self.max_n_atoms, 3))
        self.max_L = 5
        self.n_features = 3
        self.weights = np.random.normal(size=(self.n_features, self.max_L+1, 2*self.max_L+1, 1))
        self.EXPECTED_N_RADIAL_FUNCS = 2

        ser_der = '.tmp_A_arrays_{}'
        x = 0
        while os.path.isdir(ser_der.format(x)):
            for fp in os.listdir(ser_der.format(x)):
                os.remove(os.path.join(ser_der.format(x), fp))
            os.rmdir(ser_der.format(x))
            x += 1

    def tearDown(self) -> None:
        ser_der = '.tmp_A_arrays_{}'
        x = 0
        while os.path.isdir(ser_der.format(x)):
            for fp in os.listdir(ser_der.format(x)):
                os.remove(os.path.join(ser_der.format(x), fp))
            os.rmdir(ser_der.format(x))
            x += 1

    def test_init(self) -> None:
        x = FCHLEncoding(self.points, self.charges)
        self.assertIsInstance(x, FCHLEncoding)

    @unittest.skip
    def test_iter(self) -> None:

        x = FCHLEncoding(self.points, self.charges)
        x.precompute(n_cores=None, chunksize=1, max_L=self.max_L)

        for i in range(self.n_samples):
            features = x.feature_matrix_row(i, self.weights, False)
            self.assertTupleEqual(features.shape, (self.n_features * N_CHARGES * N_CHARGES,))
            check_no_nan_in_array(features)

    @unittest.skip
    def test_get_datasamples(self) -> None:
        """
        Generates a dataset with 1 DataSample that has 1 type of each atom 
        """
        n_points = 1
        points = np.random.normal(size=(n_points, 5, 3))

        charges = np.array([[1, 6, 7, 8, 16]])

        x = FCHLEncoding(points, charges)

        # Try getting the DS of Hydrogen atoms centered at Hydrogen atoms. 
        # This should result in a list of length 1 with an empty DataSample

        ds_lst = x.get_datasamples(0, (1,1))

        self.assertEqual(len(ds_lst), 1)

        self.assertEqual(ds_lst[0].n_points, 0)


        # Try getting the DS of Carbon atoms centered at Hydrogen atoms. 
        # This should result in a list of length 1 with a DataSample with 1 atom.

        ds_lst = x.get_datasamples(0, (1, 6))

        self.assertEqual(len(ds_lst), 1)

        self.assertEqual(ds_lst[0].n_points, 1)

        shifted_C_atom = ds_lst[0].coords_cart
        shifted_C_atom = shifted_C_atom.flatten()

        C_atom = points[0, 1] - points[0, 0]
        C_atom = C_atom.flatten()

        self.assertTupleEqual(shifted_C_atom.shape, C_atom.shape)
        check_array_equality(shifted_C_atom, C_atom)

    @unittest.skip
    def test_counting_rf_evals(self) -> None:
        """
        Generates a dataset with 1 DataSample that has 1 type of each atom
        """
        n_points = 1
        points = np.random.normal(size=(n_points, 5, 3))

        charges = np.array([[1, 6, 7, 8, 16]])

        x = FCHLEncoding(points, charges)
        

        n_features = 1
        max_L = 5
        x.precompute(n_cores=None, chunksize=1, max_L=max_L)

        weights = np.random.normal(size=(n_features, max_L + 1, 2 * max_L + 1, 1))

        # Out should have shape (5 * 5 * 1,) = (25,)
        out = x.feature_matrix_row(0, weights, False)

        self.assertTupleEqual(out.shape, (25,))

    @unittest.skip
    def test_counting_rf_evals_1(self) -> None:

        n_points = 1
        n_deltas = 5
        points = np.random.normal(size=(n_points, n_deltas, 3))

        charges = np.ones((1, n_deltas))

        x = FCHLEncoding(points, charges)

        n_features = 1
        max_L = 5

        weights = np.random.normal(size=(n_features, max_L + 1, 2 * max_L + 1))

        x.max_L = max_L

        ds_lst = x.get_datasamples(0, (1,1))

        self.assertEqual(len(ds_lst), 5)
        for ds_i in ds_lst:
            self.assertEqual(ds_i.n_points, 4)
    
    def test_ds_lst_to_array(self) -> None:

        x = FCHLEncoding(self.points, self.charges)
        x.max_L = self.max_L
        empty_points = [DataSample3D.from_cartesian_coords(np.zeros((0, 3))), ]
        empty_coords = [x.coords for x in empty_points]

        test_arr = x._ds_lst_to_array(empty_coords)

        expected_shape = (1, self.max_L + 1, 2 * self.max_L + 1, self.EXPECTED_N_RADIAL_FUNCS, self.EXPECTED_N_RADIAL_FUNCS)

        self.assertTupleEqual(expected_shape, test_arr.shape)

    def test_ds_lst_to_array_1(self) -> None:
        x = FCHLEncoding(self.points, self.charges)

        x.max_L = self.max_L

        empty_points = []

        test_arr = x._ds_lst_to_array(empty_points)

        expected_shape = (1, 0, 0, self.EXPECTED_N_RADIAL_FUNCS, self.EXPECTED_N_RADIAL_FUNCS)

        self.assertTupleEqual(expected_shape, test_arr.shape)


    @unittest.skip
    def test_precompute_serialization_1(self) -> None:

        n_points = 1
        n_deltas = 5
        points = np.random.normal(size=(n_points, n_deltas, 3))

        charges = np.ones((1, n_deltas))
        max_L = 5


        x = FCHLEncoding(points, charges)
        x.precompute(n_cores=None, chunksize=1, max_L=max_L)

        # Check that the directory exists
        expected_ser_dir = '.tmp_A_arrays_0'
        self.assertTrue(os.path.isdir(expected_ser_dir))


        # Check that it has one file in it
        file_lst = os.listdir(expected_ser_dir)
        self.assertEqual(1, len(file_lst))

        # Check that the arrays can be recovered
        with open(os.path.join(expected_ser_dir, file_lst[0]), 'rb') as f:
            out = pickle.load(f)

        expected = x.precomputed_arrays[0]

        self.assertEqual(len(out), len(expected))

        for i in range(len(out)):
            self.assertTrue(np.allclose(out[i], expected[i]))




    # def test_ds_lst_to_array_2(self) -> None:
    #     x = FCHLEncoding(self.points, self.charges)

    #     x.max_L = self.max_L

    #     empty_points = []

    #     test_arr = x._ds_lst_to_array(empty_points)

    #     expected_shape = (1, 0, 0, 0, 0)

    #     self.assertTupleEqual(expected_shape, test_arr.shape)


    def test_rotational_invariance(self) -> None:
        """
        Tests rotational invariance of _ds_lst_to_array() method
        """

        x = FCHLEncoding(self.points, self.charges)
        x.max_L = self.max_L

        n_samples = 10
        n_deltas = 5

        samples = [DataSample3D.from_cartesian_coords(np.random.normal(size=(n_deltas, 3))) for i in range(n_samples)]
        coords = [x.coords for x in samples]

        rot = stats.special_ortho_group.rvs(3)

        samples_rot = [i.rotate(rot) for i in samples]
        coords_rot = [x.coords for x in samples_rot]

        arr = x._ds_lst_to_array(coords)

        arr_rot = x._ds_lst_to_array(coords_rot)

        check_array_equality(arr, arr_rot, rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    unittest.main()