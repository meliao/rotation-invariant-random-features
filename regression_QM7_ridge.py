import argparse
import logging
from timeit import default_timer
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import io
# from src.atom_encoding import ElementPairsDatasetEncoding, FCHLEncoding, WholeMoleculeDatasetEncodimg

from src.utils import write_result_to_file
# from src.Predictor import OLSPredictor, loss_square
from src.data.DataSets import load_QM7
from src.multiclass_functions import feature_matrix_from_dset_FCHL, get_regression_weights

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Set up some constants
FMT = "%(asctime)s:regression_QM7: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'


def plot_test_preds_and_actual(y: np.ndarray, y_hat: np.ndarray, fp: str, t: str, xlab: str='Actual', ylab: str='Predictions') -> None:
    fig, ax = plt.subplots()

    # idxes = np.argsort(y)
    # y_sorted = y[idxes]
    # y_hat_sorted = y_hat[idxes]

    ax.plot(y, y_hat, '.', alpha=0.5)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    fig.patch.set_facecolor('white')

    ax.set_title(t)

    fig.tight_layout()

    plt.savefig(fp)

    plt.close(fig)


def save_experiment(fp: str, 
                        dm: np.ndarray, 
                        weights: np.ndarray, 
                        test_preds: np.ndarray, 
                        test_actual: np.ndarray) -> None:
    raise NotImplementedError


def loss_mae(preds: np.ndarray, actual: np.ndarray) -> np.float32:
    return np.mean(np.abs(preds.flatten() - actual.flatten()))

CONST_KCALPERMOL_TO_EV = 0.043
CONST_NUM_SAMPLES = 7165


def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Generate design matrix
    3. Fit regression weights
    4. Predict on test data
    5. Record test results
    """
    np.random.seed(args.seed)

    perm = np.random.permutation(CONST_NUM_SAMPLES)

    if not os.path.isdir(args.save_data_dir):
        os.mkdir(args.save_data_dir)

    ##########################################################################
    ### LOAD DATA

    if args.encoding == 'FCHL':
        logging.info("Using FCHL encoding")
        train_dset, validation_dset, test_dset = load_QM7(args.data_fp, 
                                                            args.n_train, 
                                                            args.n_test, 
                                                            args.validation_set_fraction,
                                                            radial_params=np.array(args.radial_params),
                                                            perm=perm)
    else:
        raise ValueError(f"Unrecognized encoding: {args.encoding}")



    logging.info("Loaded data. %i train samples and %i test samples", len(train_dset), len(test_dset))

    ##########################################################################
    ### PRECOMPUTE


    logging.info("Precomputation on the train set with n_cores=%i, chunksize=%i, max_L=%i",
                    args.n_cores, 
                    args.precompute_chunksize, 
                    args.max_L)
    t1 = default_timer()
    train_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)

    train_precompute_time = default_timer() - t1

    logging.info("Precomputation on the validation set")
    validation_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)

    logging.info("Precomputation on the test set")
    test_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)


    ##########################################################################
    ### LOOP OVER NUMBER OF RANDOM FEATURES

    for n_features in args.n_features:
        logging.info("Working on n_features=%i", n_features)



        ##########################################################################
        ### GENERATE DESIGN MATRIX

        random_weights = np.random.normal(scale=args.weight_variance,
                                            size=train_dset.expected_weights_shape(n_features, args.max_L))
        logging.info("Drawing random weights with variance %f and shape %s", args.weight_variance, random_weights.shape)
        t2 = default_timer()
        train_features = feature_matrix_from_dset_FCHL(dset=train_dset,
                                                        weights=random_weights,
                                                        intercept=True, 
                                                        n_cores=args.n_cores,
                                                        chunksize=args.chunksize,
                                                        precompute_chunksize=args.precompute_chunksize,
                                                        max_L=args.max_L,
                                                        detailed_time_logging=False)

        train_features_time = default_timer() - t2

        val_features = feature_matrix_from_dset_FCHL(dset=validation_dset,
                                                        weights=random_weights,
                                                        intercept=True,
                                                        n_cores=args.n_cores,
                                                        chunksize=args.chunksize,
                                                        precompute_chunksize=args.precompute_chunksize,
                                                        max_L=args.max_L,
                                                        detailed_time_logging=False)

        test_features = feature_matrix_from_dset_FCHL(dset=test_dset,
                                                        weights=random_weights,
                                                        intercept=True, 
                                                        n_cores=args.n_cores,
                                                        chunksize=args.chunksize,
                                                        precompute_chunksize=args.precompute_chunksize,
                                                        max_L=args.max_L,
                                                        detailed_time_logging=False)

        train_labels = train_dset.atomization_energies * CONST_KCALPERMOL_TO_EV
        val_labels = validation_dset.atomization_energies * CONST_KCALPERMOL_TO_EV
        test_labels = test_dset.atomization_energies * CONST_KCALPERMOL_TO_EV

        #######################################################################
        ### FIT WEIGHTS
    

        logging.info("Beginning to fit weights")
        t3 = default_timer()
        weights_dd, _ = get_regression_weights(train_features, train_labels, args.l2_reg)
        train_weights_time = default_timer() - t3

        train_mae_lst = []
        val_mae_lst = []
        l2_reg_lst = []
        test_mae_dd = {}

        #######################################################################
        ### EVALUATE TRAIN/VAL ERROR AND PRINT RESULTS

        for l2_reg, weights in weights_dd.items():
            train_preds = np.matmul(train_features, weights).flatten()
            train_mae = loss_mae(train_preds, train_labels.flatten())


            val_preds = np.matmul(val_features, weights).flatten()
            val_mae = loss_mae(val_preds, val_labels.flatten())
            
            test_preds = np.matmul(test_features, weights).flatten()
            test_mae = loss_mae(test_preds, test_labels)

            logging.info("Train results for n_features=%i: l2_reg: %f, train MAE (eV): %f, val MAE (eV): %f, test MAE (eV): %f", 
                                                                                n_features,
                                                                                l2_reg,
                                                                                train_mae,
                                                                                val_mae,
                                                                                test_mae)
            train_mae_lst.append(train_mae)
            l2_reg_lst.append(l2_reg)
            val_mae_lst.append(val_mae)

            #######################################################################
            ### SAVE RESULTS

            experiment_dd = {'n_train': len(train_dset),
                                'n_val': len(validation_dset),
                                'n_test': len(test_dset),
                                'n_features': n_features,
                                'train_features_time': train_features_time,
                                'train_precompute_time': train_precompute_time,
                                'train_weights_time': train_weights_time,
                                'total_train_time': train_features_time + train_precompute_time + train_weights_time,
                                'train_mae': train_mae,
                                'val_mae': val_mae,
                                'test_mae': test_mae,
                                'max_L': args.max_L,
                                'l2_reg': l2_reg,
                                'weight_stddev': args.weight_variance,
                                'radial_params': train_dset.RADIAL_POLYNOMIAL_PARAMS}
        
            write_result_to_file(args.results_fp, **experiment_dd)

        # Find l2 reg value with lowest val mae
        optimal_idx = np.argmin(val_mae_lst)
        optimal_l2_reg = l2_reg_lst[optimal_idx]
        opt_weights = weights_dd[optimal_l2_reg]

        train_mae_opt = train_mae_lst[optimal_idx]
        val_mae_opt = val_mae_lst[optimal_idx]
        logging.info("After validation, optimal L2 regularization value is %s, train MAE (eV): %f, val MAE (eV): %f", 
                        optimal_l2_reg, 
                        train_mae_opt, 
                        val_mae_opt)



        ########################################################################
        ### EVALUATE TEST ERROR

        test_preds = np.matmul(test_features, opt_weights).flatten()
        test_mae = loss_mae(test_preds, test_labels)

        logging.info("Test MAE (ev): %f", test_mae)





        ######################################################################
        ### SERIALIZE RESULTS

        serialize_dd = {
            'test_preds': test_preds,
            'test_y': test_labels,
            'model_weights': opt_weights,
            'random_weights': random_weights
        }
        make_serialize_fp = lambda x: os.path.join(args.save_data_dir, x)
        for k, v in serialize_dd.items():
            k_fp = make_serialize_fp(f"{k}_n_features_{n_features}.npy")
            np.save(k_fp, v)
    
    logging.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_fp')
    parser.add_argument('-results_fp')
    # parser.add_argument('-save_data_fp')
    # parser.add_argument('-plot_dir')
    parser.add_argument('-n_train', type=int, default=2)
    parser.add_argument('-validation_set_fraction', type=float, default=0.1)
    parser.add_argument('-n_test', type=int, default=2)
    parser.add_argument('-n_features', type=int, nargs='+')
    parser.add_argument('-max_L', type=int, default=5)
    parser.add_argument('-n_cores', type=int, default=None)
    parser.add_argument('-chunksize', type=int, default=50)
    parser.add_argument('-precompute_chunksize', type=int, default=10)
    parser.add_argument('-l2_reg', type=float, nargs='+', default=[0.])
    parser.add_argument('-normalize_features', default=False, action='store_true')
    parser.add_argument('-encoding', default='FCHL')
    parser.add_argument('-weight_variance', type=float, default=1.)
    parser.add_argument('-n_radial_params', type=int, default=1)
    parser.add_argument('-max_radial_param', type=float, default=10.)
    parser.add_argument('-radial_params', type=float, nargs='+', default=[2 / 2.355, 4 / 2.355])
    parser.add_argument('-save_data_dir')
    parser.add_argument('-seed', default=1234, type=int)
    

    a = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)


