import unittest
from typing import Tuple
import numpy as np
from scipy import special, stats

from src.kernels import (cartesian_to_spherical_batch, 
                            compute_cos_cos_MC_term, 
                            compute_random_feature, compute_random_feature_no_sin, 
                            compute_single_term, 
                            compute_sum_of_features, 
                            cos_cos_inner_product, eval_polynomial_radial_funcs, 
                            eval_Gaussian_radial_funcs_from_centers, eval_Gaussian_radial_funcs_from_widths,
                            eval_spherical_harmonics, factorial_coeff, numba_gammaln, 
                            precompute_array_for_ds, 
                            rotate_spherical_points_batch, 
                            spherical_harmonic_coeff, 
                            spherical_harmonics_from_coords, 
                            spherical_to_cartesian, 
                            cartesian_to_spherical, 
                            rotate_spherical_points,
                            compute_cos_cos_feature_via_MC,
                            assoc_Legendre_functions, spherical_to_cartesian_batch)

from src.data.DataClasses import DataSample3D

from utils import check_scalars_close, check_array_equality


# Defining relative and absolute tolerances
ATOL = 1e-08
RTOL = 2e-05

class TestCoordinateTransformFunctions(unittest.TestCase):
    def test_0(self) -> None:
        """
        Tests the point at the North pole
        """
        x, y, z = spherical_to_cartesian(1, 0, 0)

        self.assertAlmostEqual(x, 0.)
        self.assertAlmostEqual(y, 0.)
        self.assertAlmostEqual(z, 1.)

        r, phi, theta = cartesian_to_spherical(0, 0, 1)

        self.assertAlmostEqual(r, 1.)
        self.assertAlmostEqual(theta, 0.)
        self.assertAlmostEqual(phi, 0.)

    def test_1(self) -> None:

        cart_point = np.array([1, 0, 0])
        sphe_point = np.array([1, 0, np.pi/2])

        x,y,z = spherical_to_cartesian(sphe_point[0], sphe_point[1], sphe_point[2])

        self.assertTrue(np.allclose([x,y,z], cart_point))

        r, phi, theta = cartesian_to_spherical(cart_point[0], cart_point[1], cart_point[2])

        self.assertTrue(np.allclose([r, phi, theta], sphe_point))

    def test_2(self) -> None:
        """
        Generates random points and tests invertibility
        """

        points = np.random.normal(size=3)

        r, phi, theta = cartesian_to_spherical(points[0], points[1], points[2])

        x, y, z = spherical_to_cartesian(r, phi, theta)

        self.assertTrue(np.allclose([x,y,z], points))

    def test_3(self) -> None:
        """
        Passing back and forth a point at the origin
        """
        r = 0.
        phi = 0.
        theta = 0.

        x, y, z = spherical_to_cartesian(r, phi, theta)

        check_scalars_close(x, 0.)
        check_scalars_close(y, 0.)
        check_scalars_close(z, 0.)

        x = 0.
        y = 0.
        z = 0.

        r, phi, theta = cartesian_to_spherical(x, y, z)

        check_scalars_close(r, 0.)
        check_scalars_close(phi, 0.)
        check_scalars_close(theta, 0.)


    def test_batch_1(self) -> None:

        n_points = 10
        coords_cart = np.random.normal(size=(n_points, 3))

        coords_sphe = cartesian_to_spherical_batch(coords_cart)
        self.assertFalse(np.any(np.isnan(coords_sphe)))

        coords_cart_2 = spherical_to_cartesian_batch(coords_sphe)
        self.assertFalse(np.any(np.isnan(coords_cart_2)))

        self.assertTrue(np.allclose(coords_cart, coords_cart_2))


    def test_rotate_batch_1(self) -> None:
        """
        Rotation is identity so the arrays should be the same.
        """

        n_points = 10

        coords_cart = np.random.normal(size=(n_points, 3))
        coords_sphe = cartesian_to_spherical_batch(coords_cart)

        rot = np.eye(3)

        coords_sphe_rot = rotate_spherical_points_batch(coords_sphe, rot)

        self.assertTrue(np.allclose(coords_sphe, coords_sphe_rot))

class TestRotatePoints(unittest.TestCase):
    def test_0(self) -> None:
        """
        Applies the identity rotation
        """
        points = np.random.normal(size=3)
        r, phi, theta = cartesian_to_spherical(points[0], points[1], points[2])
        r_rot, phi_rot, theta_rot = rotate_spherical_points(r, phi, theta, np.eye(3))

        self.assertAlmostEqual(r, r_rot)
        self.assertAlmostEqual(phi, phi_rot)
        self.assertAlmostEqual(theta, theta_rot)

    def test_1(self) -> None:
        """
        Applies the rotation that switches the x and y coordinates
        """
        rot = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
        points = np.random.normal(size=3)
        r, phi, theta = cartesian_to_spherical(points[0], points[1], points[2])
        r_rot, phi_rot, theta_rot = rotate_spherical_points(r, phi, theta, rot)
        x_rot, y_rot, z_rot = spherical_to_cartesian(r_rot, phi_rot, theta_rot)

        self.assertTrue(np.allclose([y_rot, x_rot, z_rot], points))


class TestCosCosInnerProduct(unittest.TestCase):
    def test_0(self) -> None:
        """
        Weights are 0.
        """

        point = np.random.uniform(size=3).astype(np.float32)

        weights = np.zeros((5,5), dtype=np.float32)
        val = cos_cos_inner_product(point[0], point[1], point[2], weights)

        self.assertEqual(val, 0.)

    def test_1(self) -> None:
        """
        Weights are 1 with shape (2,2) and point is North pole
        """
        r = 1.
        phi = 0.
        theta = 0.
        weights = np.ones((2,2), dtype=np.float32)

        val = cos_cos_inner_product(r, phi, theta, weights)

        # Value should be equal to the sum of all of the weights

        self.assertAlmostEqual(weights.sum(), val)

    def test_2(self) -> None:
        """
        Weights are random and point is North pole
        """
        r = 1.
        phi = 0.
        theta = 0.
        weights = np.random.normal(size=(5,5)).astype(np.float32)

        val = cos_cos_inner_product(r, phi, theta, weights)

        self.assertAlmostEqual(weights.sum(), val, places=5)


class TestComputeCosCosMCTerm(unittest.TestCase):
    @unittest.skip
    def test_0(self) -> None:
        """
        Point is random, weights are random, and rotation matrix is the identity.
        Tests against cos_cos_inner_product
        """

        point = np.random.uniform(size=3).astype(np.float32).reshape(1,3)

        rot = np.eye(3)
        weights = np.random.normal(size=(5,5)).astype(np.float32)

        val_test = compute_cos_cos_MC_term(point, rot, weights)

        val_expected = cos_cos_inner_product(point[0,0], point[0,1], point[0,2], weights) ** 2

        self.assertAlmostEqual(val_test, val_expected, places=5)

    @unittest.skip('Deprecated')
    def test_1(self) -> None:
        """
        Point is random (and this time there are 3 delta functions) and rotation
        matrix is identity. Tests against cos_cos_inner_product
        """
        point = np.random.uniform(size=(3,3)).astype(np.float32)

        rot = np.eye(3)

        weights = np.random.normal(size=(5,5)).astype(np.float32)

        val_test = compute_cos_cos_MC_term(point, rot, weights)

        val_expected = cos_cos_inner_product(point[0, 0], point[0, 1], point[0, 2], weights)
        val_expected += cos_cos_inner_product(point[1, 0], point[1, 1], point[1, 2], weights)
        val_expected += cos_cos_inner_product(point[2, 0], point[2, 1], point[2, 2], weights)
        val_expected = val_expected ** 2

        self.assertAlmostEqual(val_test, val_expected, places=5)


    def test_2(self) -> None:
        """
        Performance when coords_sphe is empty
        """

        points = np.array([])
        points = points.reshape((0,3))

        rot = np.eye(3)

        weights = np.random.normal(size=(5,5)).astype(np.float32)

        val_test = compute_cos_cos_MC_term(points, rot, weights)

        val_expected = 0.

        self.assertAlmostEqual(val_test, val_expected)


class TestComputeCosCosFeatureViaMC(unittest.TestCase):
    def setUp(self) -> None:
        self.identity_10 = np.array([np.eye(3) for i in range(10)])
        self.identity_20 = np.array([np.eye(3) for i in range(20)])

    
    def test_0(self) -> None:
        point = np.random.uniform(size=(3,3)).astype(np.float32)
        weights = np.random.normal(size=(5,5)).astype(np.float32)

        val_0 = compute_cos_cos_feature_via_MC(point, self.identity_10, weights)
        val_1 = compute_cos_cos_feature_via_MC(point, self.identity_20, weights)

        self.assertAlmostEqual(val_0, val_1)

    def test_1(self) -> None:
        """
        Tests when point is empty
        """

        point = np.array([])
        point = point.reshape((0,3))

        weights = np.random.normal(size=(5,5)).astype(np.float32)
        
        val_test = compute_cos_cos_feature_via_MC(point, self.identity_10, weights)

        val_expected = 0.

        self.assertAlmostEqual(val_test, val_expected)


class TestFactorialCoeff(unittest.TestCase):
    def test_0(self) -> None:
        """
        Hand-testing small m and l
        """
        m = 2
        l = 2
        m_plus_l_factorial = 24
        l_minus_m_factorial = 1
        expected_val = l_minus_m_factorial / m_plus_l_factorial
        out = factorial_coeff(l, m)
        check_scalars_close(expected_val, out)

    def test_1(self) -> None:
        """
        Checks numba_gammaln against scipy.special.gammaln
        """

        x = 3
        expected_val = special.gammaln(x)
        out = numba_gammaln(x)
        check_scalars_close(expected_val, out)

    def test_2(self) -> None:
        """
        Systematic test of factorial_coeff() for all possible calls up to MAX_L = 5
        """

        for l in range(5):
            for m in range(-l, l+1):
                out = factorial_coeff(l, m)
                z = l + m + 1
                m_input = -2 * m 
                expected = special.poch(z, m_input)
                check_scalars_close(out, expected)

class TestRadialFunctions(unittest.TestCase):

    def test_poly_0(self) -> None:

        r = 1.
        params = np.random.uniform(-10, 10, size=5)

        x = eval_polynomial_radial_funcs(r, params)

        self.assertTupleEqual(params.shape, x.shape)

        self.assertTrue(np.allclose(np.ones_like(x), x))


    def test_poly_1(self) -> None:
        r = 2.
        params = np.array([0., 1., 2., 3.])

        expected = np.array([1., 2., 4., 8.])

        out = eval_polynomial_radial_funcs(r, params)

        self.assertTrue(np.allclose(out, expected))


    @unittest.skip
    def test_gauss_0(self) -> None:

        r = 1.

        params = np.array([0., 1., 2.])

        expected = np.array([1 / np.e, 1., 1 / np.e])

        out = eval_Gaussian_radial_funcs(r, params)

        s = f"expected: {expected}, out: {out}"

        self.assertTrue(np.allclose(expected, out), s)


class TestAssociatedLegendreFunctions(unittest.TestCase):
    def test_0(self) -> None:
        """
        Just tests that the function returns without error and the return type
        and shape are correct
        """
        max_L = 5
        out = assoc_Legendre_functions(1., max_L)
        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1))
        self.assertEqual(out.dtype, np.float32)

    def test_1(self) -> None:
        x = np.random.uniform(0, np.pi)
        max_L = 2
        out = assoc_Legendre_functions(x, max_L)

        self.assertEqual(out[0, max_L], 1.)
        self.assertEqual(out[0, max_L+1], 0.)
        self.assertEqual(out[0, max_L-1], 0.)

    def test_2(self) -> None:
        """
        Tests against answers from Wikipedia
        """
        x = np.random.uniform(0, np.pi)
        max_L = 2
        out = assoc_Legendre_functions(x, max_L)

        ######################################################################
        # Expected answers (from Wikipedia's table)
        c = np.cos(x)
        s = np.sin(x)
        val = np.array([[0, 0, 1, 0, 0],
                        [0, 0.5 * s, c, -s, 0],
                        [1 / 8 * ( 1 - c ** 2), 0.5 * c * s, 0.5 * (3 * (c ** 2) - 1), -3 * c * s, 3 * (1 - c ** 2)]])
        self.assertTupleEqual(val.shape, out.shape)
        # print("OUT")
        # print(out.real)
        # print("VAL")
        # print(val)
        check_array_equality(out, val, 'out', 'val')

    def test_3(self) -> None:
        """
        Tests against scipy's lpmn function
        """
        x = np.random.uniform(0, np.pi)
        max_L = 10
        out = assoc_Legendre_functions(x, max_L)

        ######################################################################
        # Check answers via scipy
        val, _ = special.lpmn(max_L, max_L, np.cos(x))
        val = val.T


        ######################################################################
        # Slice the output array
        out_check = out[:, max_L:]

        self.assertTupleEqual(val.shape, out_check.shape)
        check_array_equality(out_check, val, 'out_check', 'val')

    def test_4(self) -> None:
        """
        Tests against scipy's lpmv function
        """
        x = np.random.uniform(0, np.pi)
        max_L = 10
        out = assoc_Legendre_functions(x, max_L)

        ######################################################################
        # Check against scipy
        val = np.zeros_like(out)

        for l in range(max_L + 1):
            l_idx = l
            for m in range(-l, l+1):
                m_idx = max_L + m
                val[l_idx, m_idx] = special.lpmv(m, l, np.cos(x))

        check_array_equality(out, val, 'out', 'val')

    def test_5(self) -> None:
        """
        Tests edge case theta = 0. against scipy's lpmn function
        """
        
        x = 0.
        max_L = 10
        out = assoc_Legendre_functions(x, max_L)
        ######################################################################
        # Check answers via scipy
        val, _ = special.lpmn(max_L, max_L, np.cos(x))
        val = val.T


        ######################################################################
        # Slice the output array
        out_check = out[:, max_L:]

        check_array_equality(out_check, val, 'out_check', 'val')

    def test_6(self) -> None:
        """
        Tests edge case theta = pi. against scipy's lpmn function
        """
        
        x = np.pi
        max_L = 10
        out = assoc_Legendre_functions(x, max_L)
        ######################################################################
        # Check answers via scipy
        val, _ = special.lpmn(max_L, max_L, np.cos(x))
        val = val.T


        ######################################################################
        # Slice the output array
        out_check = out[:, max_L:]

        check_array_equality(out_check, val, 'out_check', 'val')


class TestSphericalHarmonics(unittest.TestCase):
    def setUp(self) -> None:
        pass
        # self.params_0 = np.array([-np.pi / 2])       
        # self.params_1 = np.array([1.])       



    def test_0(self) -> None:
        """
        Just tests that the function returns without error and shapes are correct.
        """
        r = 1.
        phi = 0.
        theta = np.pi / 2

        max_L = 5


        out = eval_spherical_harmonics(r, phi, theta, max_L)

        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1))
        self.assertEqual(out.dtype, np.complex64)

    def test_1(self) -> None:
        """
        Tests up to order 1
        """
        r = 1.
        phi = np.random.uniform(0, np.pi * 2)
        theta = np.random.uniform(0, np.pi)

        max_L = 1
        out_0 = eval_spherical_harmonics(r, phi, theta, max_L)
        # out_1 = eval_spherical_harmonics(r, phi, theta, max_L, self.params_1, False)

        ######################################################################
        # Compute expected value
        val = np.zeros_like(out_0)
        val[0, max_L] = np.sqrt(1 / (4 * np.pi)) 
        val[1, 0] = np.sqrt(6 / (4 * np.pi)) * 1 / 2 * np.sin(theta) * np.exp(1j * -1 * phi)
        val[1, 1] = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        val[1, 2] = np.sqrt(3 / (8 * np.pi)) * -1 * np.sin(theta) * np.exp(1j * phi)

        check_array_equality(out_0, val, 'out_0', 'val')
        # check_array_equality(out_1, val, 'out_1', 'val')

    def test_2(self) -> None:
        """
        Tests against scipy's spherical harmonics functions
        """
        r = 1. 
        phi = np.random.uniform(0, np.pi * 2)
        theta = np.random.uniform(0, np.pi)
        max_L = 10
        out_0 = eval_spherical_harmonics(r, phi, theta, max_L)

        #######################################################################
        # Compare against scipy
        val = np.zeros_like(out_0)
        for l in range(max_L + 1):
            l_idx = l
            for m in range(-l, l+1):
                m_idx = max_L + m
                val[l_idx, m_idx] = special.sph_harm(m, l, phi, theta)

        check_array_equality(out_0, val, 'out', 'val')

    def test_coeff_0(self) -> None:
        val_0_0 = np.sqrt(1 / (4 * np.pi)) 
        self.assertEqual(val_0_0, spherical_harmonic_coeff(0, 0))

        val_1_neg_1 = np.sqrt(6 / (4 * np.pi))
        self.assertEqual(val_1_neg_1, spherical_harmonic_coeff(1, -1))

        val_1_0 = np.sqrt(3 / (4 * np.pi))
        self.assertEqual(val_1_0, spherical_harmonic_coeff(1, 0))

        val_1_1 = np.sqrt(3 / (8 * np.pi))
        self.assertEqual(val_1_1, spherical_harmonic_coeff(1, 1))

    def test_coeff_1(self) -> None:
        for l in range(10):
            for m in range(-l, l+1):
                out = spherical_harmonic_coeff(l, m)
                self.assertFalse(np.isnan(out), f"Error at l: {l}, m: {m}")
                self.assertFalse(np.isinf(out), f"Error at l: {l}, m: {m}")


class TestComputeSingleTerm(unittest.TestCase):
    def test_0(self) -> None:
        """
        Arrays are filled with zeros, so answer should be 0
        """
        arr_0 = np.zeros(5)
        arr_1 = np.zeros(5)
        # w_1 = np.random.normal()
        # w_2 = np.random.normal()
        m = 2
        l = 2

        out = compute_single_term(arr_0, arr_1, m, l)
        self.assertEqual(out, 0.)



    # def test_1(self) -> None:
    #     """
    #     Weights are zero, so answer should be 0
    #     """
    #     arr_0 = np.random.normal(size=5)
    #     arr_1 = np.random.normal(size=5)
    #     w_1 = np.random.normal()
    #     w_2 = 0.
    #     m = 1
    #     l = 2
    #     out = compute_single_term(arr_0, arr_1, w_1, w_2, m, l)
    #     self.assertEqual(out, 0.)

    def test_2(self) -> None:
        """
        Arrays are filled with zeros except the middle is 1 so answer should be norm const = 8 * pi ** 2
        """
        max_L = 3
        arr_0 = np.zeros((2 * max_L + 1))
        arr_1 = np.zeros((2 * max_L + 1))
        arr_0[max_L] = 1.
        arr_1[max_L] = 1.
        l = 0
        m = 0

        out = compute_single_term(arr_0, arr_1, m, l)
        self.assertEqual(out, 8 * np.pi ** 2)

    def test_3(self) -> None:
        """
        Arrays have ones in all entries and weights are 1 so answer should be +- norm const = 8pi**2
        """
        max_L = 3
        arr_0 = np.ones((2 * max_L + 1))
        arr_1 = np.ones((2 * max_L + 1))

        l = 2
        m = 2

        out = compute_single_term(arr_0, arr_1, m, l)
        self.assertEqual(out, 8 * np.pi ** 2 / 5)

        m = 1
        out = compute_single_term(arr_0, arr_1, m, l)
        self.assertEqual(out, -1 * 8 * np.pi ** 2 / 5)

    def test_4(self) -> None:
        max_L = 3
        arr_0 = np.random.normal(size=2*max_L + 1)
        arr_1 = np.random.normal(size=2*max_L + 1)
        # w_0 = np.random.normal()
        # w_1 = np.random.normal()

        l = 3
        m = 3
        sign_m = 1
        val = 0
        for i in range(2*max_L + 1):
            neg_idx = 2*max_L - i
            val += sign_m * arr_0[i] * arr_1[neg_idx]
            sign_m = sign_m * -1
        val = val * 8 * np.pi * np.pi / (2 * l + 1)
        out = compute_single_term(arr_0, arr_1, m, l)
        check_scalars_close(val, out)


    def test_5(self) -> None:
        """
        Tests that the equation for a single term is symmetric about m and -m
        """
        max_L = 3
        arr_0 = np.random.normal(size=(2 * max_L + 1))
        arr_1 = np.random.normal(size=(2 * max_L + 1))
        
        l = 2
        m = 2

        out_1 = compute_single_term(arr_0, arr_1, m, l)
        # self.assertEqual(out, 8 * np.pi ** 2 / 5)

        out_2 = compute_single_term(arr_1, arr_0, -m, l)

        check_scalars_close(out_1, out_2)


class TestComputeRandomFeature(unittest.TestCase):
    def setUp(self) -> None:
        self.random_point_single = np.array([[1., np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi)]])
        self.max_L = 5
        # self.sphe_harm_evals = eval_spherical_harmonics(self.random_point_single[0], 
        #                                                 self.random_point_single[1],
        #                                                 self.random_point_single[2],
        #                                                 self.max_L)
        # self.sphe_harm_evals = self.sphe_harm_evals[np.newaxis]
        self.params_0 = np.array([0.])       
        self.params_1 = np.array([1.])       
        self.precomputed_arr = precompute_array_for_ds(self.random_point_single, self.max_L, self.params_1, True)
        self.weights = np.random.normal(size=(self.max_L+1, 2 * self.max_L + 1, 1))


    def test_0(self) -> None:
        """
        Tests to make sure everything runs and output is correct shape
        """
        out = compute_random_feature(self.precomputed_arr, self.weights)
        self.assertNotAlmostEqual(out, 0.)


    def test_1(self) -> None:
        """
        Inputs are zero arrays. Expected output is a zero._
        """
        max_L = 5
        n_radial_params = 5

        ds_array = np.zeros((max_L + 1, 2 * max_L + 1, n_radial_params, n_radial_params))
        weights = np.zeros((max_L + 1, 2 * max_L + 1, n_radial_params))

        out = compute_random_feature_no_sin(ds_array, weights)

        self.assertEqual(out, 0.)


class TestSphericalHarmonicsFromCoords(unittest.TestCase):
    def setUp(self) -> None:
        pass
        # self.params_0 = np.array([0.])       
        # self.params_1 = np.array([1.])       

    def test_0(self) -> None:
        """
        Tests whether the functions spherical_harmonics_from_coords and eval_spherical_harmonics agree.
        """
        n_points = 5
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T
        self.assertTupleEqual(points.shape, (n_points, 3))

        max_L = 3
        evals = spherical_harmonics_from_coords(points, max_L)
        for i in range(n_points):
            r = rs[i]
            phi = phis[i]
            theta = thetas[i]
            out = eval_spherical_harmonics(r, phi, theta, max_L)
            check_array_equality(out, evals[i])

    def test_1(self) -> None:
        """
        Input is an empty coordinate array of size (0, 3)
        """
        points = np.empty(shape=(0,3))
        max_L = 3
        evals = spherical_harmonics_from_coords(points, max_L)
        self.assertTupleEqual(evals.shape, (0, max_L+1, 2*max_L+1))
    

class TestPrecomputeArrayForDS(unittest.TestCase):
    def setUp(self) -> None:
        self.params_centers = np.array([0., 0.5, 1.])       
        self.params_widths = np.array([0.01, 0.1, 1.0])   
        self.random_4_params = np.random.uniform(0, 1, size=4)


    def test_0(self) -> None:
        """
        Just tests that everything runs without breaking
        """
        n_points = 5
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T

        radial_params = np.random.uniform(0, 1, size=3)

        max_L = 3

        out = precompute_array_for_ds(points, max_L, radial_params, False)
        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1, 3, 3))

        out = precompute_array_for_ds(points, max_L, radial_params, True)
        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1, 3, 3))


    def test_1(self) -> None:
        """
        Empty array of spherical coordinates
        """
        points = np.empty(shape=(0,3))
        max_L = 3

        out = precompute_array_for_ds(points, max_L, self.random_4_params, False)
        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1, 4, 4))
        check_array_equality(out, np.zeros_like(out))

        out = precompute_array_for_ds(points, max_L, self.random_4_params, True)
        self.assertTupleEqual(out.shape, (max_L + 1, 2 * max_L + 1, 4, 4))
        check_array_equality(out, np.zeros_like(out))

    def test_first_element(self) -> None:
        """
        Asserts that the (0, 0) element of the array is equal to the correct thing.
        int_{SO(3)} Y_0^0(x_1) Y_0^0(x_2) dQ = 2 pi
        We are evaluating the Gaussian radial centered at 1 with width 1
        Summing over N^2 terms
        So the answer should be (N^2) * (radial func eval)^2 * 2 pi   
        """

        n_points = 3
        max_L = 5
        rs = np.ones(n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)
        points = np.stack([rs, phis, thetas]).T

        radial_params = np.array([1., 1.])

        radial_func_eval = eval_Gaussian_radial_funcs_from_widths(1., radial_params)[0]

        expected_val = 2 * np.pi * (n_points ** 2) * (radial_func_eval ** 2)

        arr = precompute_array_for_ds(points, max_L, radial_params, False).real

        for i in range(radial_params.shape[0]):
            for j in range(radial_params.shape[0]):
                out_val = arr[0, max_L, i, j]

                check_scalars_close(expected_val, out_val)

    def test_2(self) -> None:
        """
        Asserts tensor is invariant to the ordering of two data points with Gaussian centers radial funcs
        """
        n_points = 2
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T

        max_L = 3

        out_1 = precompute_array_for_ds(points, max_L, self.params_centers, True)

        # perm = np.random.permutation(n_points)
        points_permuted = np.stack([points[1], points[0]])

        self.assertTupleEqual(points_permuted.shape, (2, 3))

        out_2 = precompute_array_for_ds(points_permuted, max_L, self.params_centers, True)

        check_array_equality(out_1, out_2)

    def test_3(self) -> None:
        """
        Asserts tensor is invariant to the ordering of two data points with Gaussian widths radial funcs
        """
        n_points = 2
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T

        max_L = 3

        out_1 = precompute_array_for_ds(points, max_L, self.params_widths, False)

        # perm = np.random.permutation(n_points)
        points_permuted = np.stack([points[1], points[0]])

        self.assertTupleEqual(points_permuted.shape, (2, 3))

        out_2 = precompute_array_for_ds(points_permuted, max_L, self.params_widths, False)

        check_array_equality(out_1, out_2)

    def test_4(self) -> None:
        """
        Asserts tensor is invariant to the ordering of the random data points with Gaussian centers radial funcs
        """
        n_points = 5
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T

        max_L = 3

        out_1 = precompute_array_for_ds(points, max_L, self.params_centers, True)

        perm = np.random.permutation(n_points)
        points_permuted = points[perm]

        out_2 = precompute_array_for_ds(points_permuted, max_L, self.params_centers, True)

        check_array_equality(out_1, out_2)


    def test_5(self) -> None:
        """
        Asserts tensor is invariant to the ordering of the random data points with Gaussian widths radial funcs
        """
        n_points = 5
        rs = np.random.uniform(0, 2, size=n_points)
        phis = np.random.uniform(0, 2 * np.pi, size=n_points)
        thetas = np.random.uniform(0, np.pi, size=n_points)

        points = np.stack([rs, phis, thetas]).T

        max_L = 3

        out_1 = precompute_array_for_ds(points, max_L, self.params_widths, False)

        perm = np.random.permutation(n_points)
        points_permuted = points[perm]

        out_2 = precompute_array_for_ds(points_permuted, max_L, self.params_widths, False)

        check_array_equality(out_1, out_2)

    def test_6_unit_sphere(self) -> None:
        """
        Tests that the tensor is rotationally invariant for random points on the unit sphere with Gaussian center radial funcs
        """
        np.random.seed(0)
        max_L = 5

        n_params = 3
        for n_points in range(1, 11):
            rs = np.ones(n_points)
            phis = np.random.uniform(0, 2 * np.pi, size=n_points)
            thetas = np.random.uniform(0, np.pi, size=n_points)

            points = np.stack([rs, phis, thetas]).T

            # params = np.random.uniform(0, 2, size=n_params)


            ds = DataSample3D.from_spherical_coords(points)
            ds_rotated = ds.rotate(stats.special_ortho_group.rvs(3))



            arr_1_centers = precompute_array_for_ds(ds.coords, max_L, self.params_centers, True)
            arr_2_centers = precompute_array_for_ds(ds_rotated.coords, max_L, self.params_centers, True)
            check_array_equality(arr_1_centers, arr_2_centers, atol=ATOL, rtol=RTOL)

    def test_6_random_radii(self) -> None:
        """
        Tests that the tensor is rotationally invariant for random points with Gaussian center radial funcs
        """
        np.random.seed(0)
        max_L = 5

        n_params = 3
        for n_points in range(1, 11):
            rs = np.random.uniform(0, 2, size=n_points)
            phis = np.random.uniform(0, 2 * np.pi, size=n_points)
            thetas = np.random.uniform(0, np.pi, size=n_points)

            points = np.stack([rs, phis, thetas]).T

            # params = np.random.uniform(0, 2, size=n_params)


            ds = DataSample3D.from_spherical_coords(points)
            ds_rotated = ds.rotate(stats.special_ortho_group.rvs(3))



            arr_1_centers = precompute_array_for_ds(ds.coords, max_L, self.params_centers, True)
            arr_2_centers = precompute_array_for_ds(ds_rotated.coords, max_L, self.params_centers, True)
            check_array_equality(arr_1_centers, arr_2_centers, atol=ATOL, rtol=RTOL)

    def test_7(self) -> None:
        """
        Tests that the tensor is rotationally invariant for random points with Gaussian widths raidal funcs
        """
        np.random.seed(0)
        max_L = 5

        n_params = 3
        for n_points in range(1, 11):
            rs = np.random.uniform(0, 2, size=n_points)
            phis = np.random.uniform(0, 2 * np.pi, size=n_points)
            thetas = np.random.uniform(0, np.pi, size=n_points)

            points = np.stack([rs, phis, thetas]).T

            # params = np.random.uniform(0, 2, size=n_params)


            ds = DataSample3D.from_spherical_coords(points)

            arr_1 = precompute_array_for_ds(ds.coords, max_L, self.params_widths, False)


            ds_rotated = ds.rotate(stats.special_ortho_group.rvs(3))
            arr_2 = precompute_array_for_ds(ds_rotated.coords, max_L, self.params_widths, False)
            check_array_equality(arr_1, arr_2, atol=ATOL, rtol=RTOL)

    def test_8(self) -> None:
        """
        Tests that a tensor with 1 delta function is rotationally invariant for Gaussian centers radial funcs
        """
        max_L = 3
        n_points = 1
        n_params = 3
        for trial in range(10):
            rs = np.random.uniform(0, 2, size=n_points)
            phis = np.random.uniform(0, 2 * np.pi, size=n_points)
            thetas = np.random.uniform(0, np.pi, size=n_points)

            points = np.stack([rs, phis, thetas]).T

            params = np.random.uniform(0, 2, size=n_params)


            ds = DataSample3D.from_spherical_coords(points)

            arr_1 = precompute_array_for_ds(ds.coords, max_L, params, True)


            ds_rotated = ds.rotate(stats.special_ortho_group.rvs(3))

            arr_2 = precompute_array_for_ds(ds_rotated.coords, max_L, params, True)

            check_array_equality(arr_1, arr_2)


    def test_9(self) -> None:
        """
        Tests that a tensor with 1 delta function is rotationally invariant for Gaussian widths radial funcs
        """
        max_L = 3
        n_points = 1
        n_params = 3
        for trial in range(10):
            rs = np.random.uniform(0, 2, size=n_points)
            phis = np.random.uniform(0, 2 * np.pi, size=n_points)
            thetas = np.random.uniform(0, np.pi, size=n_points)

            points = np.stack([rs, phis, thetas]).T

            params = np.random.uniform(0, 2, size=n_params)


            ds = DataSample3D.from_spherical_coords(points)

            arr_1 = precompute_array_for_ds(ds.coords, max_L, params, False)


            ds_rotated = ds.rotate(stats.special_ortho_group.rvs(3))

            arr_2 = precompute_array_for_ds(ds_rotated.coords, max_L, params, False)

            check_array_equality(arr_1, arr_2)


class TestComputeSumOfFeatures(unittest.TestCase):

    def setUp(self) -> None:
        self.n_points = 5
        self.random_points = np.stack([np.random.uniform(0, 1, size=self.n_points),
                                        np.random.uniform(0, np.pi * 2, size=self.n_points),
                                        np.random.uniform(0, np.pi, size=self.n_points)]).T
        self.max_L = 5
        self.radial_params = np.array([1., 2.])       
        self.precomputed_arr = precompute_array_for_ds(self.random_points, self.max_L, self.radial_params, False)
        self.precomputed_arrays = np.stack([self.precomputed_arr, self.precomputed_arr, self.precomputed_arr])
        self.weights = np.random.normal(size=(self.max_L+1, 2 * self.max_L + 1, 2))

    def test_0(self) -> None:
        """
        Just making sure everything runs and compiles without error
        """

        # max_L = 5

        # n_samples = 5
        # n_radial_funcs = 4
        # arr = np.zeros((n_samples, max_L + 1, 2 * max_L + 1, n_radial_funcs))

        # weights = np.random.normal(size=(max_L + 1, 2 * max_L + 1, n_radial_funcs))

        out = compute_sum_of_features(self.precomputed_arrays, self.weights)

        # self.assertEqual(out, 0.)

    def test_1(self) -> None:
        """
        Empty precompute array
        """
        max_L = 5

        n_samples = 5
        n_radial_funcs = 4
        arr = np.zeros((n_samples, max_L + 1, 2 * max_L + 1, n_radial_funcs, n_radial_funcs))

        weights = np.random.normal(size=(max_L + 1, 2 * max_L + 1, n_radial_funcs))

        out = compute_sum_of_features(arr, weights)

        self.assertEqual(out, 0.)



if __name__ == '__main__':
    unittest.main()