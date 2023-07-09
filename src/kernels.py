"""
Optimized kernels that are just-in-time compiled by numba.
"""
from typing import Tuple
import numpy as np
# from scipy import special
import numba
from numba.extending import get_cython_function_address
import numba
import ctypes


@numba.jit(nopython=True)
def spherical_to_cartesian(r: np.float32, phi: np.float32, theta: np.float32) -> Tuple[np.float32, np.float32, np.float32]:
    """Input is a point in spherical coordinates (radius, azimuth, inclination)
    and output is a point in Cartesian coordinates (x, y, z).

    Args:
        r (np.float32): point's radius
        phi (np.float32): azimith angle (longitude -- ranges [0, 2 pi])
        theta (np.float32): inclination angle (latitude -- ranges [0, pi])

    Returns:
        Tuple[np.float32, np.float32, np.float32]: Cartesian coordinates (x,y,z)
    """
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = r * np.cos(theta)
    return x,y,z

@numba.jit(nopython=True)
def spherical_to_cartesian_batch(coords_sphe: np.ndarray) -> np.ndarray:
    """Converts an array of spherical coordinates to cartesian coordinates.

    Args:
        coords_sphe (np.ndarray): Shape (n_deltas, 3). Has columns (r, phi, theta) 

    Returns:
        np.ndarray: Shape (n_deltas, 3). Has columns (x, y, z)
    """
    out = np.full_like(coords_sphe, np.nan, dtype=np.float32)
    for i in range(coords_sphe.shape[0]):
        x, y, z = spherical_to_cartesian(coords_sphe[i, 0],
                                                coords_sphe[i, 1],
                                                coords_sphe[i, 2])
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


@numba.jit(nopython=True)
def cartesian_to_spherical(x: np.float32, y: np.float32, z: np.float32) -> Tuple[np.float32, np.float32, np.float32]:
    """Input is a point in Cartesian coordinates (x, y, z) and output is a point
    in spherical coordinates (radius, azimuth, inclination)

    Args:
        x (np.float32): x coordinate
        y (np.float32): y coordinate
        z (np.float32): z coordinate

    Returns:
        Tuple[np.float32, np.float32, np.float32]: spherical coordinates (r, phi, theta)
    """
    if x == 0 and y == 0 and z == 0:
        return 0., 0., 0.
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return r, phi, theta

@numba.jit(nopython=True)
def cartesian_to_spherical_batch(coords_cart: np.ndarray) -> np.ndarray:
    """Converts an array of cartesian coordinates to spherical coordinates.

    Args:
        coords_cart (np.ndarray): Shape (n_deltas, 3). Has columns (x, y, z)

    Returns:
        np.ndarray: Shape (n_deltas, 3). Has columns (r, phi, theta)
    """
    out = np.full_like(coords_cart, np.nan, dtype=np.float32)
    for i in range(coords_cart.shape[0]):
        r, phi, theta = cartesian_to_spherical(coords_cart[i, 0],
                                                coords_cart[i, 1],
                                                coords_cart[i, 2])
        out[i, 0] = r
        out[i, 1] = phi
        out[i, 2] = theta
    return out

@numba.jit(nopython=True)
def rotate_spherical_points(r: np.float32, phi: np.float32, theta: np.float32, rot: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
    """Transforms the spherical coordinates to Cartesian coordinates, then 
    performs matrix vector product to rotate them, then transforms back into
    spherical coordinates

    Args:
        r (np.float32): point's radius
        phi (np.float32): azimuth angle
        theta (np.float32): inclination angle
        rot (np.ndarray): rotation matrix. Has shape (3, 3)

    Returns:
        Tuple[np.float32, np.float32, np.float32]: spherical coordinates (r, phi, theta)
    """
    # raise NotImplementedError
    x, y, z = spherical_to_cartesian(r, phi, theta)
    x_rot = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z
    y_rot = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z
    z_rot = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z
    return cartesian_to_spherical(x_rot, y_rot, z_rot)

def rotate_spherical_points_batch(coords_sphe: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Transforms the spherical coordinates specified in coords to Cartesian coordinates, 
    then performs matrix vector product to rotate them, then transforms back into spherical
    coordinates. 

    Args:
        coords_sphe (np.ndarray): Real-valued array of shape (n_deltas, 3). Column 0 is the radius, 
        Column 1 is phi (azimuth angle) and Col 2 is theta, the inclination angle.
        rot (np.ndarray): Real-valued array of shape (3, 3). A rotation matrix. 

    Returns:
        np.ndarray: Real-valued with shape (n_deltas, 3)
    """

    coords_cart = spherical_to_cartesian_batch(coords_sphe)
    coords_cart_rotated = np.matmul(rot, coords_cart.T).T
    return cartesian_to_spherical_batch(coords_cart_rotated)


# The following 10-ish lines were used from this super helpful StackOverflow answer:
# https://stackoverflow.com/a/54855769

_double = ctypes.c_double

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_double, _double)
gammaln_float64 = functype(addr)

@numba.jit(nopython=True)
def numba_gammaln(x):
  return gammaln_float64(x)

@numba.jit(nopython=True)
def factorial_coeff(l: int, m: int) -> np.float32:
    """ This function computes (l - m)! / (l+m)!

    Gamma(n) = (n-1)!
    So we want Gamma(l - m + 1) / Gamma(l + m + 1)
    Or, in log space, Exp( LGamma(l - m + 1) - LGamma(l + m + 1))
    
    Used in computing normalization constants for spherical harmonics.

    Args:
        l (int): spherical harmonic index
        m (int): spherical harmonic index

    Returns:
        float32
    """

    numerator = numba_gammaln(l - m + 1)
    denominator = numba_gammaln(l + m + 1)
    return np.exp(numerator - denominator)



@numba.jit(nopython=True, locals={'coeff': numba.float32})
def spherical_harmonic_coeff(l: int, m: int) -> np.float32:
    """sqrt(((2l + 1) * (l - m)!) / (4 * pi * (l+m)!))

    Args:
        l (int): index
        m (int): index

    Returns:
        np.float32: coefficient for computing spherical harmonics
    """
    coeff = factorial_coeff(l, m)
    num = (2 * l + 1) * coeff
    denom = 4 * np.pi
    x = num / denom
    return np.sqrt(x)


@numba.jit(nopython=True)
def eval_polynomial_radial_funcs(r: np.float32, params: np.ndarray) -> np.ndarray:
    out = np.empty_like(params)

    for i, exponent in enumerate(params):
        out[i] = r ** exponent
    return out

@numba.jit(nopython=True)
def eval_Gaussian_radial_funcs_from_widths(r: np.float32, params: np.ndarray) -> np.ndarray:
    out = np.empty_like(params)

    center = 1.
    sqrt_2_pi = np.sqrt(np.pi * 2)    

    for i, sigma in enumerate(params):
        out[i] = np.exp(-1 * ((r - center) ** 2) / ( 2 * sigma ** 2)) / ( sigma * sqrt_2_pi)

    return out

@numba.jit(nopython=True)
def eval_Gaussian_radial_funcs_from_centers(r: np.float32, params: np.ndarray, width: np.float32) -> np.ndarray:
    out = np.empty_like(params)

    # sigma = 0.25
    sigma = width
    sqrt_2_pi = np.sqrt(np.pi * 2)
    for i, center in enumerate(params):
        # TODO: This extra factor of 1.6 is meant to control the maximum value and put it near 1
        out[i] = np.exp(-1 * ((r - center) ** 2) / ( 2 * sigma ** 2)) / (sigma * sqrt_2_pi) # / 1.6

    return out


@numba.jit(nopython=True)
def eval_cos_radial_funcs(r: np.float32, params: np.ndarray) -> np.ndarray:
    out = np.empty_like(params)
    for i, freq in enumerate(params):
        shift = np.pi / 2
        if i % 2:
            shift = 0
        out[i] = np.abs(np.cos(freq * r + shift))
    return out

@numba.jit(nopython=True)
def eval_spherical_harmonics(r: np.float32, 
                                phi: np.float32, 
                                theta: np.float32, 
                                max_L: int) -> np.ndarray:
    """Computes the solid spherical harmonics up to order <max_L>. Produces a complex-valued array
    of size (max_L, 2 * max_L + 1). Does NOT evaluate radial functions.

    Args:
        r (np.float32): point's radius
        phi (np.float32): azimuth angle
        theta (np.float32): inclination angle
        max_L (int): maximum order 


    Returns:
        np.ndarray: complex-valued array of size (max_L + 1, 2 * max_L + 1)
    """

    out = np.zeros((max_L + 1, 2 * max_L + 1), dtype=np.complex64)

    # Start with an array filled with associated Legendre function evaluations
    lpmn = assoc_Legendre_functions(theta, max_L)

    

    
    for l in range(max_L+1):
        for m in range(-l, l+1):
            m_idx = max_L + m
            exp_term = np.exp(1j * m * phi)
            legendre_eval = lpmn[l, m_idx]
            coeff = spherical_harmonic_coeff(l, m)
            out[l, m_idx] = coeff * legendre_eval * exp_term
    return out

@numba.jit(nopython=True)
def precompute_array_for_ds(coords: np.ndarray, 
                            max_L: int, 
                            radial_params: np.ndarray,
                            params_are_centers_bool: bool,
                            normalize_functions_bool: bool=False,
                            width_param: np.float32=1.) -> np.ndarray:
    """Computes the B precomputation array described in appendix section "Precomputation" of the paper.


    Args:
        coords (np.ndarray): spherical coordinates of evaluation points. Has shape
            (n_points, 3)
        max_L (int): Maximum of the spherical harmonics.
        radial_params: np.ndarray: Parameters for different radial functions.
            Has shape (n_params,)
        params_are_centers_bool: bool: If True, the radial_params are interpreted
            as centers for Gaussians with pre-defined widths. If False, the 
            radial_params are interpreted as widths for Gaussians with pre-defined
            centers.  

    Returns:
        np.ndarray: Precomputation A array. Has shape (max_L + 1, 2 * max_L + 1, n_params, n_params)
    """

    n_points, _ = coords.shape
    n_params = radial_params.shape[0]
    out = np.zeros((max_L + 1, 2 * max_L + 1, n_params, n_params), dtype=np.complex64)

    if n_points == 0:
        return out


    # First, evaluate the radial functions. radial_func_evals has shape (n_points, n_params)
    radial_func_evals = np.zeros((n_points, n_params), dtype=np.float32)
    for i in range(n_points):
        rad = coords[i, 0]
        if params_are_centers_bool:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_centers(rad, radial_params, width_param)
        else:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_widths(rad, radial_params)

    # sphe_harm_evals has shape (n_points, max_L + 1, 2 * max_L + 1)
    sphe_harm_evals = spherical_harmonics_from_coords(coords, max_L)


    for k_1 in range(n_params):
        for k_2 in range(n_params):
            for i in range(n_points):
                rad_func_eval_i = radial_func_evals[i, k_1]
                for j in range(n_points):
                    rad_func_eval_j = radial_func_evals[j, k_2]
                    for l in range(max_L + 1):
                        sphe_harm_i = sphe_harm_evals[i, l]
                        sphe_harm_j = sphe_harm_evals[j, l]

                        first_m = -l
                        first_m_idx = max_L + first_m
                        
                        # Call compute_single_term to evaluate the rightmost element of the row
                        val = compute_single_term(sphe_harm_i, sphe_harm_j, first_m, l) * rad_func_eval_j * rad_func_eval_i

                        out[l, first_m_idx, k_1, k_2] += val # * rad_func_eval_i * rad_func_eval_j

                        # Fill in the row by flipping signs
                        for m in range(-l+1, l+1):
                            m_idx = max_L + m
                            val = -1 * val
                            out[l, m_idx, k_1, k_2] += val

    if normalize_functions_bool:
        norm_factor = 1  / (n_points ** 2)
        for l in range(max_L + 1):
            for m in range(-l, l+1):
                m_idx = max_L + m
                out[l, m_idx] = norm_factor * out[l, m_idx]

    return out

@numba.jit(nopython=True)
def spherical_harmonics_from_coords(coords: np.ndarray, 
                                    max_L: int) -> np.ndarray:
    """Takes in an array representing a list of points in spherical coordinates.
    Evaluates the spherical harmonics up to order max_L and returms all of the 
    evaluations in an array. Does NOT evaluate radial functions. 

    Args:
        coords (np.ndarray): spherical coordinates of evaluation points. Has shape
        (n_points, 3)
        max_L (int): Maximum of the spherical harmonics. 

    Returns:
        np.ndarray: spherical harmonics evals. Has shape (n_points, max_L+1, 2 * max_L + 1)
    """
    n_points, _ = coords.shape
    out = np.zeros((n_points, max_L + 1, 2 * max_L + 1), dtype=np.complex64)

    for i in range(n_points):
        r = coords[i, 0]
        phi = coords[i, 1]
        theta = coords[i, 2]
        out[i] = eval_spherical_harmonics(r, phi, theta, max_L)
    return out

@numba.jit(nopython=True)
def assoc_Legendre_functions(theta: np.float64, max_L: int) -> np.ndarray:
    """Computes the values of the associated Legendre functions evaluated at 
    cos(theta) for all values of m and l < max_L. Returns a complex array 
    because the array will be modified in-place by the calling function.

    Args:
        theta (np.float64): angle in [0, pi]
        max_L (int): max order to compute

    Returns:
        np.ndarray: complex-valued array of size (max_L + 1, 2 * max_L + 1)
    """
    out = np.zeros((max_L + 1, 2 * max_L + 1), dtype=np.float32)
    c = np.cos(theta)
    s = np.sin(theta)

    # Base case for the recursion
    out[0, max_L] = 1.

    
    # First, use two different recursive formulas to seed the two rightmost diagonals
    for l in range(0, max_L):
        out[l + 1, max_L + l + 1] = -1 * (2 * l + 1) * s * out[l, max_L + l]
        
        out[l + 1, max_L + l] = c * (2 * l + 1) * out[l, max_L + l]
        
    # Next, use these seeds to fill in the table recursing down the column
    for m in range(0, max_L):
        m_idx = max_L + m
        for l in range(m + 1, max_L):
            numerator = (2 * l + 1) * c * out[l, m_idx] - (l + m) * out[l - 1, m_idx]
            denominator = l - m + 1
            out[l+1, m_idx] = numerator / denominator
            
    # Finally, use a third formula to recover P^l_{-m} from P^l_m
    for l in range(1, max_L + 1):
        for m in range(1, l + 1):
            neg_m_idx = max_L - m
            m_idx = max_L + m
            out[l, neg_m_idx] = factorial_coeff(l, m) * ((-1) ** m) * out[l, m_idx]

    return out

@numba.jit(nopython=True)
def compute_single_term(y_1: np.ndarray, y_2: np.ndarray, m: int, l: int) -> np.complex64:
    """Computes a single term (given by algorithm 1 in the document)

    Args:
        y_1 (np.ndarray): The spherical harmonic evaluations at order l for point 1
        y_2 (np.ndarray): The spherical harmonic evaluations at order l for point 2
        m (int): an index indicating the starting position
        l (int): another index

    Returns:
        np.complex64: The resulting sum, which is a result of integrating the product of rotated 
        spherical harmonic evaluations around SO(3).
    """
    x = 0.
    max_l = int((y_1.shape[0] - 1) / 2)
    for m_idx in range(max_l -l, max_l + l + 1):
        m_1 = max_l - m_idx
        neg_m_idx = 2 * max_l - m_idx
        add_term = y_1[m_idx] * y_2[neg_m_idx] * (-1) ** (m - m_1)
        x += add_term
    x = x * (8 * np.pi ** 2) / (2 * l + 1)

    return x


@numba.jit(nopython=True)
def compute_random_feature(data_sample_array: np.ndarray, weights: np.ndarray) -> np.float64:
    """Computes a random feature for a given set of points and given set of random weights.
    <data_sample_array> should be created by a call to precompute_array_for_ds. 

    Args:
        data_sample_array (np.ndarray): Complex array of size (max_L + 1, 2 * max_L + 1, n_params, n_params)
        weights (np.ndarray): Random weights. Has shape (max_L+1, 2 * max_L + 1, n_params)

    Returns:
        np.float64: Random feature evaluation
    """
    return np.sin(compute_random_feature_no_sin(data_sample_array, weights))
    # max_L_plus_1, _, n_radial_funcs, _ = data_sample_array.shape

    # max_L = max_L_plus_1 - 1

    # out = 0.
    # for k_1 in range(n_radial_funcs):
    #     for k_2 in range(n_radial_funcs):
    #         for l in range(max_L_plus_1):
    #             for m in range(-l, l+1):
    #                 m_idx = m + max_L
    #                 neg_m_idx = -1 * m_idx
    #                 out += weights[l, m_idx, k_1] * weights[l, neg_m_idx, k_2] * data_sample_array[l, m_idx, k_1, k_2]
    # out = np.sin(out)
    # return out


@numba.jit(nopython=True)
def compute_random_feature_no_sin(data_sample_array: np.ndarray, weights: np.ndarray) -> np.float64:
    """Computes a random feature for a given set of points and given set of random weights.
    <data_sample_array> should be created by a call to precompute_array_for_ds. 

    Args:
        data_sample_array (np.ndarray): Complex array of size (max_L + 1, 2 * max_L + 1, n_params, n_params)
        weights (np.ndarray): Random weights. Has shape (max_L+1, 2 * max_L + 1, n_params)

    Returns:
        np.float64: Random feature evaluation
    """
    max_L_plus_1, _, n_radial_funcs, _ = data_sample_array.shape

    max_L = max_L_plus_1 - 1

    out = 0.
    for k_1 in range(n_radial_funcs):
        for k_2 in range(n_radial_funcs):
            for l in range(max_L_plus_1):
                for m in range(-l, l+1):
                    m_idx = m + max_L
                    neg_m_idx = -1 * m_idx
                    out += weights[l, m_idx, k_1] * weights[l, neg_m_idx, k_2] * data_sample_array[l, m_idx, k_1, k_2].real
    return out
    
@numba.jit(nopython=True)
def compute_sum_of_features(array: np.ndarray, freqs: np.ndarray) -> np.complex64:
    """For every array along the 0th axis of <array>, this function calls 
    compute_random_feature() and then sums all of the evaluations. Used by 
    FCHLEncoding

    Args:
        array (np.ndarray): Complex-valued. Has shape (n_samples, max_L + 1, 2 * max_L + 1, n_radial_funcs)
        freqs (np.ndarray): Random frequencies. Has shape (max_L + 1, 2 * max_L + 1, n_radial_funcs)

    Returns:
        np.complex64: Scalar. Sum of random feature evaluations.
    """

    # AXIS = 0
    # b = np.sum(array, axis=AXIS)

    # return compute_random_feature(b, freqs)

    out = 0.
    for a in array:
        out += compute_random_feature(a, freqs)
    return out

@numba.jit(nopython=True)
def compute_multiple_features(precomputed_array: np.ndarray, weights: np.ndarray, intercept: bool) -> np.ndarray:
    """Used by WholeMoleculeDatasetEncodimg

    Args:
        precomputed_array (np.ndarray): Evaluations of the spherical harmonics at a set of delta functions. 
        shape is (max_L, 2 * max_L + 1, n_radial_funcs)
        weights (np.ndarray): Random weights. Has shape (n_features, max_L, 2 * max_L + 1, n_radial_funcs)
        intercept (bool): Whether to add an intercept of 1 at the end of the array

    Returns:
        np.ndarray: complex array of size (n_features + intercept) with feature evaluations for each weight matrix.
    """
    out = np.empty(weights.shape[0] + int(intercept), dtype=np.complex64)

    for i in range(weights.shape[0]):
        w = weights[i]
        out[i] = compute_random_feature(precomputed_array, w)

    if intercept:
        out[-1] = 1.
    return out

@numba.jit(nopython=True)
def compute_feature_matrix_row(sphe_harm_evals: np.ndarray, weights: np.ndarray, intercept: bool) -> np.ndarray:
    """Used by atom_encoding.ElementPairsDatasetEncoding

    Args:
        sphe_harm_evals (np.ndarray): Size (n_atom_pairs, n_deltas, max_L, 2 * max_L + 1)
        weights (np.ndarray): Size (n_features, max_L, 2 * max_L + 1)
        intercept (bool): Whether to add an intercept at end of the row
    Returns:
        np.ndarray: feature vector of size (n_atom_pairs x n_features + intercept)
    """
    n_atom_pairs, n_deltas, max_L, _ = sphe_harm_evals.shape
    max_L = max_L - 1

    n_features, _, _ = weights.shape

    out = np.empty(n_atom_pairs * n_features + int(intercept), dtype=np.complex64)

    for i in range(n_features):
        weights_i = weights[i]
        # evals_i = sphe_harm_evals[i]
        for j in range(n_atom_pairs):
            evals_j = sphe_harm_evals[j]
            out[i * n_atom_pairs + j] = compute_random_feature(evals_j, weights_i)
    if intercept:
        out[-1] = 1.
    return out

@numba.jit(nopython=True)
def cos_cos_inner_product(r: np.float32, phi: np.float32, theta: np.float32, weights: np.ndarray) -> np.float32:
    """Computes the cosine cosine inner product at a single point.

    Args:
        r (np.float32): point's radius
        phi (np.float32): azimuth angle
        theta (np.float32): inclination angle
        weights (np.ndarray): Random weights that specify a given feature. Has shape (max_K, max_L)

    Returns:
        np.float32: inner product value
    """
    feat_sum = 0.0
    for k in range(weights.shape[0]):
        for l in range(weights.shape[1]):
            feat_sum += weights[k,l] * r * np.cos(k * phi) * np.cos(2 * l * theta)

    return feat_sum

@numba.jit(nopython=True)
def compute_cos_cos_MC_term(coords_sphe: np.ndarray, rot: np.ndarray, weights: np.ndarray) -> np.float32:
    """Takes the points in coords_sphe, rotates by the given rotation matrix 
    rot, and then computes the cosine cosine feature at the rotated point 
    and specified weight matrix

    Args:
        coords_sphe (np.ndarray): Points in 3D spherical coordinates. Has shape (n_deltas, 3)
        rot (np.ndarray): Rotation matrix. Has shape (3, 3)
        weights (np.ndarray): Random weights that specify a given feature. Has shape (max_K, max_L)

    Returns:
        np.float32: A single term in the MC sum
    """

    feat_sum = 0.0

    for point in coords_sphe:
        # Rotate points
        r, phi, theta = rotate_spherical_points(point[0], point[1], point[2], rot)
        
        # Compute feature value
        feat_sum += cos_cos_inner_product(r, phi, theta, weights)

    return feat_sum ** 2


@numba.jit(nopython=True)
def compute_cos_cos_feature_via_MC(coords_sphe: np.ndarray, rot_matrices: np.ndarray, weights: np.ndarray) -> np.float32:
    """For every rotation matrix in rot_matrices, this computes a term in the MC integration. It 
    returns an average of all of the terms.

    Args:
        coords_sphe (np.ndarray): Points in 3D spherical coordinates. Has shape (n_deltas, 3)
        rot_matrices (np.ndarray): An array of rotation matrices. Has shape (n_MC_terms, 3, 3)
        weights (np.ndarray): Random weights that specify a given feature. Has shape (max_K, max_L)

    Returns:
        np.float32: The integral value estimated by MC.
    """
    val = 0.0
    for rot in rot_matrices:
        val += compute_cos_cos_MC_term(coords_sphe, rot, weights)

    return val / rot_matrices.shape[0]

