"""
This script only times inference on QM7. It does not train a real model or evaluate test error.
"""
import argparse
import logging
from timeit import default_timer
from typing import List
import numpy as np
import pickle

from src.atom_encoding import FCHLEncoding, numba_get_datasamples_for_row
import numba

from src.data.DataSets import load_QM7
from src.multiclass_functions import FCHL_feature_matrix_row
from src.kernels import precompute_array_for_ds

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Set up some constants
FMT = "%(asctime)s:regression_QM7: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'



@numba.jit(nopython=True, parallel=False)
def precompute_function(charges: np.ndarray, 
                        coords: np.ndarray, 
                        charges_lst: List[int], 
                        radial_params: np.ndarray,
                        max_L: np.ndarray) -> List[np.ndarray]:
    """_summary_

    Args:
        charges (np.ndarray): A sample's charges. Has shape (max_n_atoms,)
        coords (np.ndarray): A sample's 3D coordinates. Has shape (max_n_atoms, 3)
        charges_lst (List[int]): The list of all possible charges in the QM* dataset. Has length 5
        ds_lst_to_arr_func (Callable):

    Returns:
        List[np.ndarray]: _description_
    """
    dsamples = numba_get_datasamples_for_row(charges, coords, charges_lst)
    n_dsamples = len(dsamples)
    out = [np.zeros((len(x), max_L + 1, 2 * max_L + 1, radial_params.shape[0], radial_params.shape[0]), dtype=np.complex64) 
                for x in dsamples]
    for i in numba.prange(n_dsamples):
        out[i] = _ds_lst_to_array(dsamples[i], radial_params, max_L)

    return out

@numba.jit(nopython=True)
def _ds_lst_to_array(ds_lst: List[np.ndarray], radial_params: np.ndarray, max_L: int) -> np.ndarray:
    """Given a list of coordinates of point clouds, this produces an array stacking their
        precomputation tensor evaluations. The precomputed tensor evals come 
        from calls to precompute_array_for_ds()

        Args:
            ds_lst (List[np.ndarray]): A variable-length list of DataSamples. 
                Has length n_ds. Each DataSample is assumed to have the same number
                of points n_points.
            radial_params (np.ndarray): Has shape (n_radial_funcs,)
            max_L (int): Determines the maximum order of spherical harmonics used.

        Returns:
            np.ndarray: Complex-valued array with shape 
                (n_ds, max_L + 1, 2 * max_L + 1, n_radial_funcs, n_radial_funcs)
    """
    if len(ds_lst):
        n_ds = len(ds_lst)
        out = np.zeros((n_ds, max_L + 1, 2 * max_L + 1, radial_params.shape[0], radial_params.shape[0]), dtype=np.complex64)

        for i in range(n_ds):
            coords_i = ds_lst[i]
            out[i] = precompute_array_for_ds(coords_i, max_L, radial_params, False, width_param=np.nan)

        return out
    else:
        return np.zeros((1, 0, 0, radial_params.shape[0], radial_params.shape[0]), dtype=np.complex64)


def test_latency_one_sample(test_dset: FCHLEncoding, 
                            n_runs: int,
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
    n_atoms_lst = []
    n_samples = test_dset.n_samples
    for j in range(n_runs):
        for i in range(n_samples):

            coords_i = test_dset.coords_cart[i]
            charges_i = test_dset.charges[i]
            n_atoms_i = np.sum(charges_i != 0)

            # Time precomputation
            time_precomputation_start = default_timer()
            precomputed_arrays = precompute_function(charges_i, 
                                                        coords_i, 
                                                        test_dset.CHARGES_LST, 
                                                        test_dset.RADIAL_POLYNOMIAL_PARAMS,
                                                        test_dset.max_L)
            time_precomputation = default_timer()
            
            # Time inner product step
            time_inner_products_start = default_timer()
            sample_row = FCHL_feature_matrix_row((precomputed_arrays, random_weights, intercept))
            time_inner_products = default_timer() - time_inner_products_start

            # Time dot product step
            time_dot_start = default_timer()
            out = np.dot(sample_row, model_weights)
            time_dot = default_timer() - time_dot_start

            # Update lists
            precomp_times.append(time_precomputation - time_precomputation_start)
            inner_prods_times.append(time_inner_products)
            dot_times.append(time_dot)
            n_atoms_lst.append(n_atoms_i)


    precomp_times = np.array(precomp_times[1:])
    inner_prods_times = np.array(inner_prods_times[1:])
    dot_times = np.array(dot_times[1:])
    n_atoms = np.array(n_atoms_lst[1:])
    logging.info("Finished with latency testing. Averages over %i samples", n_runs)

    logging.info("Mean Precomputation: %f, min: %f, max: %f", precomp_times.mean(), precomp_times.min(), precomp_times.max())
    logging.info("Mean time on inner product step: %f, min: %f, max: %f", 
                    inner_prods_times.mean(), 
                    inner_prods_times.min(), 
                    inner_prods_times.max())
    out_dd = {'precomputation_step': precomp_times, 
                'inner_product_step': inner_prods_times, 
                'final_prediction_step': dot_times, 
                'n_atoms': n_atoms}

    with open(fp, 'wb') as f:
        pickle.dump(out_dd, f)



def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Generate design matrix
    3. Fit regression weights
    4. Predict on test data
    5. Record test results
    """

    ###########################################################################
    ### LOAD DATA

    _, _, test_dset = load_QM7(args.data_fp, n_train=2, n_test=args.n_samples, validation_set_fraction=0.5)

    test_dset.max_L = args.max_L

    logging.info("Loaded %i samples for latency testing", len(test_dset))


    ####################################################################
    ### TIMING LATENCY
    logging.info("N_features = %i", args.n_features)
    logging.info("Beginning testing latency of predicting 1 sample")


    random_weights = np.random.normal(size=test_dset.expected_weights_shape(args.n_features, args.max_L))
    fake_model_weights = np.random.normal(size=test_dset.feature_matrix_ncols(random_weights, intercept=True))


    # This takes (args.n_samples * args.n_latency_runs) latency samples
    test_latency_one_sample(test_dset,
                                args.n_latency_runs,
                                random_weights,
                                fake_model_weights,
                                True,
                                args.latency_results_fp)


    logging.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_fp')
    parser.add_argument('-latency_results_fp')
    parser.add_argument('-n_latency_runs', type=int, default=5)
    parser.add_argument('-n_features', type=int)
    parser.add_argument('-max_L', type=int, default=5)
    parser.add_argument('-n_samples', type=int, default=1433)    

    a = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)


