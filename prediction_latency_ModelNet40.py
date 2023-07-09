"""
This script only times inference on ModelNet40. It does not train a real model or evaluate test error.
"""
import argparse
import logging
from timeit import default_timer
from typing import List
import numpy as np
import os
import pickle
from src.data.ModelNet_utils import points_to_shapenet_encoding
from src.DataSet import ShapeNetEncoding
from src.kernels import eval_Gaussian_radial_funcs_from_centers, eval_Gaussian_radial_funcs_from_widths, compute_single_term, compute_random_feature, eval_spherical_harmonics
import numba



import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Set up some constants
FMT = "%(asctime)s:regression_QM7: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'


@numba.jit(nopython=True, parallel=True)
def spherical_harmonics_from_coords_parallel(coords: np.ndarray, 
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

    for i in numba.prange(n_points):
        r = coords[i, 0]
        phi = coords[i, 1]
        theta = coords[i, 2]
        out[i] = eval_spherical_harmonics(r, phi, theta, max_L)
    return out


@numba.jit(nopython=True, parallel=True)
def new_precompute_parallel(coords: np.ndarray, 
                            max_L: int, 
                            radial_params: np.ndarray,
                            params_are_centers_bool: bool,
                            normalize_functions_bool: bool,
                            width_param: np.float32) -> np.ndarray:
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
    for i in numba.prange(n_points):
        rad = coords[i, 0]
        if params_are_centers_bool:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_centers(rad, radial_params, width_param)
        else:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_widths(rad, radial_params)

    # sphe_harm_evals has shape (n_points, max_L + 1, 2 * max_L + 1)
    sphe_harm_evals = spherical_harmonics_from_coords_parallel(coords, max_L)


    for k_1 in range(n_params):
        for k_2 in range(n_params):
                for l in range(max_L + 1):
                    v = 0.
                    first_m = -l
                    first_m_idx = max_L + first_m
                    for i in numba.prange(n_points):

                        rad_func_eval_i = radial_func_evals[i, k_1]
                        sphe_harm_i = sphe_harm_evals[i, l]
                        for j in range(n_points):
                            rad_func_eval_j = radial_func_evals[j, k_2]
                            sphe_harm_j = sphe_harm_evals[j, l]


                        
                            # Call compute_single_term to evaluate the rightmost element of the row
                            val = compute_single_term(sphe_harm_i, sphe_harm_j, first_m, l) * rad_func_eval_j * rad_func_eval_i

                            v += val
                    out[l, first_m_idx, k_1, k_2] = v # * rad_func_eval_i * rad_func_eval_j


    for k_1 in numba.prange(n_params):
        for k_2 in numba.prange(n_params):
            for l in numba.prange(max_L + 1):
                first_m = -l
                first_m_idx = max_L + first_m
                val = out[l, first_m_idx, k_1, k_2]
                # Fill in the row by flipping signs
                for m in range(-l+1, l+1):
                    m_idx = max_L + m
                    val = -1 * val
                    out[l, m_idx, k_1, k_2] = val

    if normalize_functions_bool:
        norm_factor = 1  / (n_points ** 2)
        for l in range(max_L + 1):
            for m in range(-l, l+1):
                m_idx = max_L + m
                out[l, m_idx] = norm_factor * out[l, m_idx]

    return out


@numba.jit(nopython=True, parallel=True)
def precompute_array_for_ds_parallel(coords: np.ndarray, 
                            max_L: int, 
                            radial_params: np.ndarray,
                            params_are_centers_bool: bool,
                            normalize_functions_bool: bool,
                            width_param: np.float32) -> np.ndarray:
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
    for i in numba.prange(n_points):
        rad = coords[i, 0]
        if params_are_centers_bool:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_centers(rad, radial_params, width_param)
        else:
            radial_func_evals[i] = eval_Gaussian_radial_funcs_from_widths(rad, radial_params)

    # sphe_harm_evals has shape (n_points, max_L + 1, 2 * max_L + 1)
    sphe_harm_evals = spherical_harmonics_from_coords_parallel(coords, max_L)


    for i in numba.prange(n_points):
        for k_1 in range(n_params):
            for k_2 in range(n_params):
    # for k_1 in range(n_params):
    #     for k_2 in range(n_params):
    #         for i in range(n_points):
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

@numba.jit(nopython=True, parallel=True)
def compute_multiple_features_parallel(precomputed_array: np.ndarray, weights: np.ndarray, intercept: bool) -> np.ndarray:
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

    for i in numba.prange(weights.shape[0]):
        w = weights[i]
        out[i] = compute_random_feature(precomputed_array, w)

    if intercept:
        out[-1] = 1.
    return out

def test_latency_one_sample(test_dset: ShapeNetEncoding, 
                            n_runs: int,
                            max_L: int,
                            random_weights: np.ndarray,
                            model_weights: np.ndarray,
                            intercept: bool,
                            fp: str) -> None:
    """_summary_

    Args:
        test_dset (FCHLEncoding): _description_
    """
    precomp_times = []
    inner_prods_times = []
    dot_times = []
    n_samples = test_dset.n_samples

    for j in range(n_runs):
        for i in range(n_samples):

            coords_i = test_dset.coords_cart[i]

            # Time precomputation
            time_precomputation_start = default_timer()
            precomputed_array = new_precompute_parallel(coords_i,
                                                        max_L,
                                                        test_dset.radial_params,
                                                        test_dset.PARAMS_ARE_CENTERS_BOOL,
                                                        True,
                                                        test_dset.bump_width)
            time_precomputation = default_timer()

            # print(type(precomputed_arrays))
            # for x in precomputed_arrays:
                # assert np.logical_not(np.any(np.isnan(x)))
            
            # Time inner product step
            time_inner_products_start = default_timer()
            sample_row = compute_multiple_features_parallel(precomputed_array,
            random_weights,
            intercept)
            time_inner_products = default_timer() - time_inner_products_start

            # Time model prediction step
            time_dot_start = default_timer()
            out_probs = np.matmul(model_weights, sample_row)
            x = np.argmax(out_probs)
            time_dot = default_timer() - time_dot_start

            # Update lists
            precomp_times.append(time_precomputation - time_precomputation_start)
            inner_prods_times.append(time_inner_products)
            dot_times.append(time_dot)


    precomp_times = np.array(precomp_times[1:])
    inner_prods_times = np.array(inner_prods_times[1:])
    dot_times = np.array(dot_times[1:])
    logging.info("Finished with latency testing. Averages over %i samples", n_runs)

    logging.info("Mean Precomputation: %f, min: %f, max: %f", precomp_times.mean(), precomp_times.min(), precomp_times.max())
    logging.info("Mean time on random feature computation step: %f, min: %f, max: %f", 
                    inner_prods_times.mean(), 
                    inner_prods_times.min(), 
                    inner_prods_times.max())
    logging.info("Mean time on model evaluation step: %f, min: %f, max: %f", 
                    dot_times.mean(), 
                    dot_times.min(), 
                    dot_times.max())
    out_dd = {'precomputation_step': precomp_times, 
                'inner_product_step': inner_prods_times, 
                'final_prediction_step': dot_times}

    with open(fp, 'wb') as f:
        pickle.dump(out_dd, f)

CONST_NUM_CLASSES = 40

def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Generate design matrix
    3. Fit regression weights
    4. Predict on test data
    5. Record test results
    """

    numba.set_num_threads(args.n_threads)
    
    if os.path.isfile(args.latency_results_fp):
        logging.info("Already found a file at %s. Exiting.", args.latency_results_fp)
        return


    ###########################################################################
    ### LOAD DATA

    samples = np.load(os.path.join(args.data_dir, 'train_samples.npy'))[:args.n_samples, :args.n_deltas]
    labels = np.load(os.path.join(args.data_dir, 'train_labels.npy'))[:args.n_samples]

    test_dset = points_to_shapenet_encoding(samples,
                                                labels,
                                                args.max_L,
                                                args.n_radial_params,
                                                1.0, #Max radial param
                                                0.75) #Bump width

    logging.info("Loaded %i samples for latency testing", len(test_dset))


    ####################################################################
    ### TIMING LATENCY
    logging.info("N_features = %i, Max L: %i, N threads: %i", args.n_features, args.max_L, args.n_threads)
    logging.info("Beginning testing latency; sampling %i times", len(test_dset) * args.n_latency_runs)

    random_weights = np.random.normal(size=(args.n_features, 
                                                args.max_L + 1, 
                                                2 * args.max_L + 1, 
                                                args.n_radial_params))
    # random_weights = np.random.normal(size=test_dset.expected_weights_shape(args.n_features, args.max_L))
    fake_model_weights = np.random.normal(size=(CONST_NUM_CLASSES, test_dset.feature_matrix_ncols(random_weights, intercept=True)))


    # This takes (args.n_samples * args.n_latency_runs) latency samples
    test_latency_one_sample(test_dset,
                                args.n_latency_runs,
                                args.max_L,
                                random_weights,
                                fake_model_weights,
                                True,
                                args.latency_results_fp)


    logging.info("Finished ##################################")
    logging.info("###########################################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir')
    parser.add_argument('-latency_results_fp')
    parser.add_argument('-n_latency_runs', type=int, default=5)
    parser.add_argument('-n_features', type=int)
    parser.add_argument('-n_threads', type=int)
    parser.add_argument('-max_L', type=int, default=5)
    parser.add_argument('-n_samples', type=int, default=1433)    
    parser.add_argument('-n_deltas', type=int, default=1024)    
    parser.add_argument('-n_radial_params', type=int, default=3)    

    a = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)


