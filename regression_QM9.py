import argparse
import logging
from timeit import default_timer
import numpy as np
from typing import List, Dict
import os
import datetime
import time
# import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
from scipy import io
from src.atom_encoding import FCHLEncoding
from src.utils import write_result_to_file
from src.data.DataSets import QM9Data
from src.multiclass_functions import feature_matrix_from_dset_FCHL

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Set up some constants
FMT = "%(asctime)s:regression_QM9: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'



def save_experiment(fp: str, 
                        dm: np.ndarray, 
                        weights: np.ndarray, 
                        test_preds: np.ndarray, 
                        test_actual: np.ndarray) -> None:
    raise NotImplementedError


def loss_mae(preds: np.ndarray, actual: np.ndarray) -> np.float32:
    return np.mean(np.abs(preds.flatten() - actual.flatten()))


def load_QM9(data_dir: str, folds: List[int]) -> FCHLEncoding:
    """Loads the specified data folds and returns the corresponding MolleculeDatasetEncoding
    object

    Args:
        data_dir (str): Directory where the QM9 data is stored in folds
        folds (List[int]): List of folds (integers)
        encoding_type (_type_, optional): Type of molecular encoding to create. Defaults to Type[MolleculeDatasetEncoding].

    """
    fp_lst = [os.path.join(data_dir, f"qm9_parsed_{x}.mat") for x in folds]

    q_obj = QM9Data()
    for fp in fp_lst:
        logging.info("Loading file: %s", fp)
        q_obj.extend_dataset(fp)

    return q_obj.get_dataset()

CONST_HARTREE_TO_EV = 27.2114 

def normalize_matrix(mat: np.ndarray, col_means: np.ndarray=None, col_stds: np.ndarray=None) -> np.ndarray:
    """
    Input and output both have shape (n_samples, n_features)
    
    This is meant to remove the mean of each column and scale each column to
    have std = 1.
    """
    logging.info("Normalizing a matrix of size %s", mat.shape)

    if col_means is None:
        col_means = np.mean(mat, axis = 0)

    if col_stds is None:
        col_stds = np.std(mat, axis=0)
    mat = np.add(mat, -1 * col_means, out=mat)
    mat = np.divide(mat, col_stds, out=mat)
    # x = (mat - col_means) / col_stds
    return mat, col_means, col_stds

def scipy_ridge_solver_lsqr(train_features: np.ndarray, 
                            train_y: np.ndarray,
                            test_features: np.ndarray,
                            test_y: np.ndarray,
                            out_fp: str,
                            out_dd: Dict, 
                            l2_reg: List[float],
                            atol,
                            weight_init: np.ndarray=None) -> Dict:
    """Finds ridge regression solutions to ||Ax - y||_2^2 + \lambda || x||_2^2

    Args:
        feature_mat (np.ndarray): Has shape (n_samples, n_features)
        y (np.ndarray): Has shape (n_samples)
        l2_reg (List[float]): List of regularization lambda values

    Returns:
        Dict: Keys are the elements of l2_reg
    """
    solver_stop_reason = 3

    for l2_lambda in l2_reg:
        solver_stop_reason = 3
        # previous_MAE = np.inf
        improved_last_time = True

        out_lsqr = lsqr(train_features, train_y, 
                        damp=np.sqrt(l2_lambda), 
                        x0=weight_init,
                        atol=atol,
                        btol=0.,
                        conlim=0,
                        iter_lim=12_000)
        iter_count = out_lsqr[2]

        weight_init = out_lsqr[0]
        # logging.info("Weight init shape: %s", weight_init.shape)
        solver_stop_reason = out_lsqr[1]

        # Compute train and test error
        train_preds = np.matmul(train_features, weight_init.flatten())
        # logging.info("train preds shape: %s", train_preds.shape)
        train_MAE = loss_mae(train_preds, train_y)

        test_preds = np.matmul(test_features, weight_init.flatten())
        test_MAE = loss_mae(test_preds, test_y)

        # if test_MAE > previous_MAE:

        #     if not improved_last_time:
        #         solver_stop_reason = -1 # Do early stopping if the test error is increasing for 2 logging periods
        #     improved_last_time = False
        # else: 
        #     improved_last_time = True
        # previous_MAE = test_MAE

        out_dd['train_MAE'] = train_MAE
        out_dd['test_MAE'] = test_MAE
        out_dd['iter_count'] = iter_count
        out_dd['solver_stop_reason'] = solver_stop_reason
        out_dd['l2_reg'] = l2_lambda
        out_dd['timestamp'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        out_dd['atol'] = atol
        out_dd['r1nrm'] = out_lsqr[3]
        out_dd['r2nrm'] = out_lsqr[4]
        # time.strftime("%Y-%m-%d %H:%M:%S", time.time())

        write_result_to_file(out_fp, **out_dd)

        logging.info("SOLVER OBJECTIVE VALUE r1norm: %f, r2norm: %f", out_lsqr[3], out_lsqr[4])
    
        logging.info("Final results for n_features=%i and l2_reg=%f: Train MAE (eV): %f", out_dd['n_features'],
        l2_lambda, train_MAE)

        logging.info("Final results for n_features=%i and l2_reg=%f: Test MAE (eV): %f", out_dd['n_features'],
        l2_lambda, test_MAE)

        # serialize_dd = {
        #     'train_preds': train_preds,
        #     'train_y': train_y,
        #     'test_preds': test_preds,
        #     'test_y': test_y,
        #     'weights': weight_init
        # }
        # io.savemat(serialize_fp, serialize_dd)
        return weight_init




def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Generate design matrix
    3. Fit regression weights
    4. Predict on test data
    5. Record test results
    """
    ###########################################################################
    # SET UP SERIALIZATION DIR
    if not os.path.isdir(args.serialize_dir):
        os.mkdir(args.serialize_dir)

    make_serialize_fp = lambda x: os.path.join(args.serialize_dir, x)
    ##########################################################################
    ### LOAD DATA


    train_folds = np.arange(args.n_train_folds)
    test_folds = np.arange(args.n_test_folds) + args.n_train_folds
    logging.info("Using FCHL encoding")
    train_dset = load_QM9(args.data_dir, train_folds)
    test_dset = load_QM9(args.data_dir, test_folds)

    N_TRAIN_TEST = 2000
    train_dset = train_dset.truncate(N_TRAIN_TEST)
    test_dset = test_dset.truncate(N_TRAIN_TEST)

    logging.info("Loaded data. %i train samples and %i test samples", len(train_dset), len(test_dset))

    ##########################################################################
    ### PRECOMPUTE

    logging.info("Precomputation on the train set with n_cores=%s, chunksize=%i, max_L=%i",
                    args.n_cores, 
                    args.precompute_chunksize, 
                    args.max_L)
    t1 = default_timer()
    train_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)

    train_precompute_time = default_timer() - t1
    logging.info("Precomputation on the test set")
    test_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)


    ##########################################################################
    ### LOOP OVER NUMBER OF RANDOM FEATURES

    for n_features in args.n_features:
        logging.info("Working on n_features=%i", n_features)



        ##########################################################################
        ### GENERATE DESIGN MATRIX

        np.random.seed(0)
        random_weights = np.random.normal(scale=args.weight_variance,
                                            size=train_dset.expected_weights_shape(n_features, args.max_L))
        logging.info("Drawing random weights with variance %f and shape %s", args.weight_variance, random_weights.shape)
        t2 = default_timer()
        train_features = feature_matrix_from_dset_FCHL(train_dset,
                                                    random_weights,
                                                    True, 
                                                    args.n_cores,
                                                    args.n_cores,
                                                    args.chunksize,
                                                    args.max_L)

        if np.any(np.isnan(train_features)):
            logging.warn("Found NaNs in the training features")
            logging.warn("Here are the NaNs: %s", np.argwhere(np.isnan(train_features)))

        if np.any(np.isinf(train_features)):
            logging.warn("Found Infs in the training features")
            logging.warn("Here are the Infs: %s", np.argwhere(np.isinf(train_features)))

        logging.info("Train features max: %f and min: %f", train_features.max(), train_features.min())
        # s = linalg.svd(train_features, compute_uv=False)
        # logging.info("SVD returned %i singular vals with range: [%f, %f] and cond number: %f", s.shape[0], s[0], s[-1], s[0] / s[-1])
        # continue


        if args.normalize_features:
            train_features[:, :-1], col_means, col_stds = normalize_matrix(train_features[:, :-1])

        train_features_time = default_timer() - t2

        test_features = feature_matrix_from_dset_FCHL(test_dset,
                                                    random_weights,
                                                    True,
                                                    args.n_cores,
                                                    args.n_cores,
                                                    args.chunksize,
                                                    args.max_L)
        
        if args.normalize_features:
            test_features[:, :-1], _, _ = normalize_matrix(test_features[:, :-1], col_means=col_means, col_stds=col_stds)



        # Just want to be sure I am using 32 bit floating point
        train_features = train_features.astype(np.float32)
        test_features = test_features.astype(np.float32)

        train_y = train_dset.atomization_energies.astype(np.float32) * CONST_HARTREE_TO_EV
        test_y = test_dset.atomization_energies.astype(np.float32) * CONST_HARTREE_TO_EV



        #####################################################################
        ### SERIALIZE

        train_features_fp = make_serialize_fp('train_features.npy')
        np.save(train_features_fp, train_features)
        logging.info("Train features saved to %s", train_features_fp)

        train_y_fp = make_serialize_fp('train_labels.npy')
        np.save(train_y_fp, train_y)
        logging.info("Train labels saved to %s", train_y_fp)
        #######################################################################
        ### FIT WEIGHTS
        experiment_dd = {'n_train': len(train_dset),
                            'n_test': len(test_dset),
                            'n_features': n_features,
                            'n_columns': train_features.shape[1],
                            'radial_params': train_dset.RADIAL_POLYNOMIAL_PARAMS,
                            'normalized_features': args.normalize_features,
                            'train_features_time': train_features_time,
                            'train_precompute_time': train_precompute_time,
                            'max_L': args.max_L}
            


        logging.info("Beginning to fit weights with atol=%f", args.atol)

        out_weights = scipy_ridge_solver_lsqr(train_features,
                                train_y,
                                test_features,
                                test_y,
                                args.results_fp,
                                experiment_dd,
                                args.l2_reg,
                                args.atol,
                                None)



        weights_fp = make_serialize_fp('weights.npy')
        np.save(weights_fp, out_weights)
        logging.info("Weights saved to %s", weights_fp)




        

        



    logging.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir')
    parser.add_argument('-results_fp')
    parser.add_argument('-serialize_dir')
    # parser.add_argument('-save_data_fp')
    # parser.add_argument('-plot_dir')
    parser.add_argument('-n_train_folds', type=int, default=2)
    parser.add_argument('-n_test_folds', type=int, default=2)
    parser.add_argument('-n_features', type=int, nargs='+')
    parser.add_argument('-max_L', type=int, default=5)
    parser.add_argument('-n_cores', type=int, default=None)
    parser.add_argument('-chunksize', type=int, default=50)
    parser.add_argument('-precompute_chunksize', type=int, default=20)
    parser.add_argument('-l2_reg', type=float, nargs='+', default=[0.])
    parser.add_argument('-normalize_features', default=False, action='store_true')
    parser.add_argument('-encoding', default='FCHL')
    parser.add_argument('-weight_variance', type=float, default=1.)
    parser.add_argument('-n_radial_params', type=int, default=1)
    parser.add_argument('-atol', type=float, default=1e-6)
    parser.add_argument('-max_radial_param', type=float, default=10.)

    a = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)


