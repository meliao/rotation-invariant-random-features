import logging
import argparse
from typing import Dict, List, Tuple
from timeit import default_timer
import pickle
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
import h5py
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)

from scipy import io

from src.utils import write_result_to_file
from src.data.ModelNet_utils import points_to_shapenet_encoding
from src.multiclass_functions import feature_matrix_from_dset_ModelNet

FMT = "%(asctime)s:classification_ModelNet: %(levelname)s - %(message)s"
TIMEFMT = '%Y-%m-%d %H:%M:%S'


# Read numpy array data and label from h5_filename
def load_h5(h5_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    From the SPHNet codebase 
    https://github.com/adrienPoulenard/SPHnet/blob/d30e341aaddae4d45ea537ec4787f0e72b6463c1/SPHnet/utils/data_prep_utils.py#L130
    """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_files(data_path: str, files_list_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    From the SPHNet codebase
    https://github.com/adrienPoulenard/SPHnet/blob/d30e341aaddae4d45ea537ec4787f0e72b6463c1/SPHnet/utils/data_prep_utils.py#L137

    Returns data, labels.
    data has shape (n_samples, 2048, 3)
    labels has shape (n_samples,)
    """
    files_list = [line.rstrip().split('/')[-1] for line in open(files_list_path)]
    # print(files_list)
    # raise ValueError
    data = []
    labels = []
    for i in range(len(files_list)):
        data_, labels_ = load_h5(os.path.join(data_path, files_list[i]))
        data.append(data_)
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()
    assert data.shape[0] == labels.shape[0]
    return data, labels


def create_dataset(data_dir: str, 
                    files_list_fp: str, 
                    n: int=None,
                    max_n_deltas: int=500,
                    train_dset_bool: bool=False) -> Tuple[np.ndarray]:
    
    points, labels = load_h5_files(data_dir, os.path.join(data_dir, files_list_fp))
    # print(f"points shape: {points.shape}, labels shape: {labels.shape}")
    # point_clouds = []
    # labels = []




    if n is not None:
        n_eff = n
    else:
        n_eff = points.shape[0]

        # points_out = np.full((n, max_n_deltas, 3), np.nan, dtype=np.float32)
        # labels_out = np.full((n,), np.nan, dtype=np.float32)


    keep_idxes = np.random.choice(np.arange(points.shape[0]), n_eff, replace=False)
    # logging.info("KEEP_IDXES_SHAPE: %s", keep_idxes.shape)
    
    if train_dset_bool:
        # n_eff = points.shape[0]
        n_val_eff = int(np.floor(0.1 * n_eff))
        n_train_eff = n_eff - n_val_eff
        # logging.info("N_EFF: %i, N_TRAIN_EFF: %i, N_VAL_EFF: %i", n_eff, n_train_eff, n_val_eff)
        points_train_out = np.full((n_train_eff, max_n_deltas, 3), np.nan, dtype=np.float32)
        labels_train_out = np.full((n_train_eff,), np.nan, dtype=np.float32)
        points_val_out = np.full((n_val_eff, max_n_deltas, 3), np.nan, dtype=np.float32)
        labels_val_out = np.full((n_val_eff,), np.nan, dtype=np.float32)

        for i, idx in enumerate(keep_idxes[:n_train_eff]):
            # Fill the points_train_out and labels_train_out arrays 
            # logging.info("In loop. I=%i, IDX=%i", i, idx)
            labels_train_out[i] = labels[idx]
            deltas_to_keep = np.random.choice(np.arange(points.shape[1]), max_n_deltas, replace=False)
            modelnet_obj = points[idx]
            modelnet_obj_downsampled = modelnet_obj[deltas_to_keep]            
            points_train_out[i] = modelnet_obj_downsampled

        for i, idx in enumerate(keep_idxes[n_train_eff:]):
            # Do the same for the validation dataset
            labels_val_out[i] = labels[idx]
            deltas_to_keep = np.random.choice(np.arange(points.shape[1]), max_n_deltas, replace=False)
            modelnet_obj = points[idx]
            modelnet_obj_downsampled = modelnet_obj[deltas_to_keep]            
            points_val_out[i] = modelnet_obj_downsampled

        # print(points_train_out[-3])
        assert not np.any(np.isnan(points_train_out))
        assert not np.any(np.isnan(labels_train_out))
        assert not np.any(np.isnan(points_val_out))
        assert not np.any(np.isnan(labels_val_out))

        return points_train_out, labels_train_out, points_val_out, labels_val_out
    else:
        points_out = np.full((n_eff, max_n_deltas, 3), np.nan, dtype=np.float32)
        labels_out = np.full((n_eff,), np.nan, dtype=np.float32)

        for i, idx in enumerate(keep_idxes):
            labels_out[i] = labels[idx]

            deltas_to_keep = np.random.choice(np.arange(points.shape[1]), max_n_deltas, replace=False)

            modelnet_obj = points[idx]
            modelnet_obj_downsampled = modelnet_obj[deltas_to_keep]

            points_out[i] = modelnet_obj_downsampled

        assert not np.any(np.isnan(points_out))
        assert not np.any(np.isnan(labels_out))
        return points_out, labels_out
    

def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Input and output both have shape (n_samples, n_features)
    
    This is meant to remove the mean of each column and scale each column to
    have std = 1.
    """
    x = (mat - np.mean(mat, axis=0)) / np.std(mat, axis=0)
    return x


def compute_top_k_accuracy(prediction_mat: np.ndarray, 
                            classes: np.ndarray, 
                            true_labels: np.ndarray, 
                            k: List[int]) -> List[float]:
    """_summary_

    Args:
        prediction_mat (np.ndarray): Has shape (n_samples, n_classes)
        true_labels (np.ndarray): Has shape (n_samples)
        k (List[int]): List of k's for which we want to return top-k accuracy

    Returns:
        List[float]: Top-k accuracies for the values of k
    """
    out = [0.] * len(k)
    # Loop over all of the test examples
    for i in range(prediction_mat.shape[0]):
        
        preds_i = prediction_mat[i]

        sorted_preds = np.flip(preds_i.argsort())

        for j, k_val in enumerate(k):
            top_k_pred_indices = sorted_preds[:k_val]
            top_k_classes = classes[top_k_pred_indices]

            if true_labels[i] in top_k_classes:
                out[j] += 1
    out = np.array(out)

    return out / prediction_mat.shape[0]

def standardize_matrix_columns(mat: np.ndarray, col_means: np.ndarray=None, col_stds: np.ndarray=None) -> np.ndarray:
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

TRAIN_LABELS = 'train_labels.npy'
TEST_LABELS = 'test_labels.npy'
TRAIN_SAMPLES_FP_DD = {
    'z': 'train_samples_z.npy',
    'so3': 'train_samples_so3.npy',
    'original': 'train_samples.npy',
}
TEST_SAMPLES_FP_DD = {
    'z': 'test_samples_z.npy',
    'so3': 'test_samples_so3.npy',
    'original': 'test_samples.npy',
}

def load_data(data_dir: str,
                train_class: str,
                test_class: str,
                val_set_fraction: float,
                n_deltas: int,
                n_train: int=None,
                n_test: int=None) -> Dict:
    """Loads samples and labels for train, test, val datasets.

    Args:
        data_dir (str): Directory where ModelNet40 data is written
        train_class (str): Setting. One of z, so3, or original
        test_class (str): One of z, so3, or original
        val_set_fraction (float): How much of the training set goes into the val
        set.

    Returns:
        Dict: Keys are strings and vals are numpy arrays.
    """
    logging.info("Using train datset: %s, and test dataset: %s", train_class, test_class)
    points_samples = np.random.permutation(1024)[:n_deltas]
    fp_train_labels = os.path.join(data_dir, TRAIN_LABELS)
    train_labels_pre = np.load(fp_train_labels)
    train_samples_pre = np.load(os.path.join(data_dir, TRAIN_SAMPLES_FP_DD[train_class]))

    if n_test is not None:
        test_labels = np.load(os.path.join(data_dir, TEST_LABELS))[:n_test]
        test_samples = np.load(os.path.join(data_dir, TEST_SAMPLES_FP_DD[test_class]))[:n_test, points_samples]
    else:
        test_labels = np.load(os.path.join(data_dir, TEST_LABELS))
        test_samples = np.load(os.path.join(data_dir, TEST_SAMPLES_FP_DD[test_class]))[:, points_samples]      

    # figure out how many samples belong in the train and val sets

    if n_train is not None:

        n_samples = n_train
    else:
        n_samples = train_samples_pre.shape[0]
    n_val = int(np.floor(val_set_fraction * n_samples))
    n_train = n_samples - n_val

    perm = np.random.permutation(n_samples)
    train_idxes = perm[:n_train]
    val_idxes = perm[n_train:]

    train_samples = train_samples_pre[train_idxes][:, points_samples]
    val_samples = train_samples_pre[val_idxes][:, points_samples]
    train_labels = train_labels_pre[train_idxes]
    val_labels = train_labels_pre[val_idxes]

    out_dd = {
        'train_samples': train_samples,
        'train_labels': train_labels,
        'val_samples': val_samples,
        'val_labels': val_labels,
        'test_samples': test_samples,
        'test_labels': test_labels
    }
    return out_dd



def main(args: argparse.Namespace) -> None:

    ###########################################################################
    # SET UP SERIALIZATION DIR
    if not os.path.isdir(args.serialize_dir):
        os.mkdir(args.serialize_dir)

    make_serialize_fp = lambda x: os.path.join(args.serialize_dir, x)


    ###########################################################################
    # SET SEED
    np.random.seed(args.seed)

    ###########################################################################
    # LOAD DATA
    logging.info("Beginning data loading")

    data_dd = load_data(args.data_dir,
                        args.train_class,
                        args.test_class,
                        args.validation_set_fraction,
                        args.max_n_deltas,
                        n_train=args.n_train,
                        n_test=args.n_test)

    train_dset = points_to_shapenet_encoding(data_dd['train_samples'], 
                                                data_dd['train_labels'], 
                                                args.max_L, 
                                                args.n_radial_params, 
                                                args.max_radial_param,
                                                args.bump_width)

    val_dset = points_to_shapenet_encoding(data_dd['val_samples'], 
                                                data_dd['val_labels'], 
                                                args.max_L, 
                                                args.n_radial_params, 
                                                args.max_radial_param,
                                                args.bump_width)

    test_dset = points_to_shapenet_encoding(data_dd['test_samples'], 
                                                data_dd['test_labels'], 
                                                args.max_L, 
                                                args.n_radial_params, 
                                                args.max_radial_param,
                                                args.bump_width)

    n_train = len(train_dset)
    n_val = len(val_dset)
    n_test = len(test_dset)

    logging.info("Loaded train and test data. n_train: %i, n_val: %i, n_test: %i", 
                                                                    n_train, 
                                                                    n_val,
                                                                    n_test)


    ###########################################################################
    # PRINT STATISTICS ABOUT TRAIN SET
    # logging.info()

    # logging.info("Train dset coords shape: %s", train_dset.coords_cart.shape)
    # train_dset_radii = np.linalg.norm(train_dset.coords_cart, axis=-1)
    # logging.info("Train dset radii shape: %s", train_dset_radii.shape)
    # logging.info("Max radii: %f and min radii: %f", train_dset_radii.max(), train_dset_radii.min())
    logging.info("Train dset radial params: %s and are_centers_bool: %s, and bump_width: %s", 
                        train_dset.radial_params, 
                        train_dset.PARAMS_ARE_CENTERS_BOOL,
                        train_dset.bump_width)

    ###########################################################################
    # DO PRECOMPUTATION FIRST

    logging.info("Precomputation on the train set with n_cores=%i, chunksize=%i, max_L=%i", args.n_cores, args.precompute_chunksize, args.max_L)
    t1 = default_timer()
    train_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)


    logging.info("Precomputation on the validation set")
    val_dset.precompute(n_cores=args.n_cores, chunksize=args.chunksize, max_L=args.max_L)
    
    train_precompute_time = default_timer() - t1
    
    logging.info("Precomputation on the test set")
    test_dset.precompute(n_cores=args.n_cores, chunksize=args.precompute_chunksize, max_L=args.max_L)


    ##########################################################################
    # LOOP OVER NUMBER OF RANDOM FEATURES

    for n_features in args.n_features:
        logging.info("Working on n_features=%i", n_features)

        random_weights = np.random.normal(scale=args.weight_variance,
                                        size=(n_features, 
                                                args.max_L + 1, 
                                                2 * args.max_L + 1, 
                                                args.n_radial_params))
        logging.info("Drawing random weights with variance %f and shape %s", args.weight_variance, random_weights.shape)
        t2 = default_timer()
        train_features = feature_matrix_from_dset_ModelNet(dset=train_dset, 
                                                            weights=random_weights, 
                                                            intercept=False, 
                                                            n_cores=args.n_cores, 
                                                            chunksize=args.chunksize,
                                                            precompute_chunksize=args.precompute_chunksize,
                                                            max_L=args.max_L,
                                                            detailed_time_logging=False)
        logging.info("Before standardizing, train features max: %f, and min: %f", train_features.max(), train_features.min())
    

        if args.standardize_matrix_cols:
            train_features, means, stds = standardize_matrix_columns(train_features)


        val_features = feature_matrix_from_dset_ModelNet(dset=val_dset,
                                                            weights=random_weights,
                                                            intercept=False,
                                                            n_cores=args.n_cores,
                                                            chunksize=args.chunksize,
                                                            precompute_chunksize=args.precompute_chunksize,
                                                            max_L=args.max_L,
                                                            detailed_time_logging=False)
        if args.standardize_matrix_cols:
            val_features, _, _ = standardize_matrix_columns(val_features, col_means=means, col_stds=stds)


        train_features_time = default_timer() - t2


        test_features = feature_matrix_from_dset_ModelNet(dset=test_dset, 
                                                            weights=random_weights, 
                                                            intercept=False, 
                                                            n_cores=args.n_cores, 
                                                            chunksize=args.chunksize,
                                                            precompute_chunksize=args.precompute_chunksize,
                                                            max_L=args.max_L,
                                                            detailed_time_logging=False)

        if args.standardize_matrix_cols:
            test_features, _, _ = standardize_matrix_columns(test_features, col_means=means, col_stds=stds)

        s = np.linalg.svd(train_features, compute_uv=False)

        logging.info("Train features shape: %s, singular value range [%f, %f]", 
                                                                            train_features.shape,
                                                                            s[0], 
                                                                            s[-1])

        ###########################################################################
        # FIT REGRESSION WEIGHTS

        logging.info("Beginning to fit weights")


        # should have keys [1, 5, 10]
        train_acc_dd = {}
        val_acc_dd = {}
        test_acc_dd = {}
        for k in [1, 5, 10]:
            train_acc_dd[k] = {}
            test_acc_dd[k] = {}
            val_acc_dd[k] = {}
        lr_obj = LogisticRegression(max_iter=20000, 
                                    warm_start=True)
        

        for l2_reg_val in args.l2_reg:


            logging.info("Fitting logistic regression for C=%f", l2_reg_val)
            lr_obj.set_params(C=l2_reg_val)
            t3 = default_timer()
            lr_obj.fit(train_features, train_dset.labels)

            train_optimization_time = default_timer() - t3
            # has shape (n_train, n_classes)
            prob_predictions_train = lr_obj.predict_proba(train_features)

            k_vals = [1, 5, 10]

            k_accuracies = compute_top_k_accuracy(prob_predictions_train, lr_obj.classes_, train_dset.labels, k_vals)

            for i, k in enumerate(k_vals):
                train_acc_dd[k][l2_reg_val] = k_accuracies[i]

            # logging.info("Train results for n_features=%i: l2_reg: %f, top-1: %f, top-5: %f, top-10: %f", n_features,
            #                                                                                                 l2_reg_val,
            #                                                                                                 k_accuracies[0], 
            #                                                                                                 k_accuracies[1], 
            #                                                                                                 k_accuracies[2])

            prob_predictions_val = lr_obj.predict_proba(val_features)
            k_accuracies_val = compute_top_k_accuracy(prob_predictions_val, lr_obj.classes_, val_dset.labels, k_vals)

            for i, k in enumerate(k_vals):
                val_acc_dd[k][l2_reg_val] = k_accuracies_val[i]

            prob_predictions_test = lr_obj.predict_proba(test_features)

            k_accuracies_test = compute_top_k_accuracy(prob_predictions_test, lr_obj.classes_, test_dset.labels, k_vals)

            for i, k in enumerate(k_vals):
                test_acc_dd[k][l2_reg_val] = k_accuracies_test[i]

          


            experiment_dd = {'n_train': n_train,
                                'n_test': n_test,
                                'n_features': n_features,
                                'test_top_1': k_accuracies_test[0], 
                                'test_top_5': k_accuracies_test[1],
                                'test_top_10': k_accuracies_test[2],
                                'train_top_1': k_accuracies[0],
                                'train_top_5': k_accuracies[1],
                                'train_top_10': k_accuracies[2],
                                'val_top_1': k_accuracies_val[0],
                                'val_top_5': k_accuracies_val[1],
                                'val_top_10': k_accuracies_val[2],
                                'max_L': args.max_L,
                                'weight_variance': args.weight_variance,
                                'l2_reg': l2_reg_val,
                                'n_radial_funcs': args.n_radial_params,
                                'max_n_deltas': args.max_n_deltas,
                                'max_radial_param': args.max_radial_param,
                                'min_singular_val': s[-1],
                                'max_singular_val': s[0],
                                'bump_width': args.bump_width,
                                'precompute_time': train_precompute_time,
                                'feature_matrix_time': train_features_time,
                                'total_train_time': train_features_time + train_precompute_time + train_optimization_time,
                                'optimization_time': train_optimization_time
                                }
            write_result_to_file(args.results_fp, **experiment_dd)

        ############################################################################
        # LOGGING THE TRAIN RESULTS

        for l2_reg_val in args.l2_reg:
            logging.info("Train results for n_features=%i: l2_reg: %f, top-1: %f, top-5: %f, top-10: %f", 
            n_features,
            l2_reg_val,
            train_acc_dd[1][l2_reg_val], 
            train_acc_dd[5][l2_reg_val], 
            train_acc_dd[10][l2_reg_val])

        for l2_reg_val in args.l2_reg:
            logging.info("Validation results for n_features=%i: l2_reg: %f, top-1: %f, top-5: %f, top-10: %f", 
            n_features,
            l2_reg_val,
            val_acc_dd[1][l2_reg_val], 
            val_acc_dd[5][l2_reg_val], 
            val_acc_dd[10][l2_reg_val])

        for l2_reg_val in args.l2_reg:
            logging.info("Test results for n_features=%i: l2_reg: %f, top-1: %f, top-5: %f, top-10: %f", 
            n_features,
            l2_reg_val,
            test_acc_dd[1][l2_reg_val], 
            test_acc_dd[5][l2_reg_val], 
            test_acc_dd[10][l2_reg_val])


        # ###########################################################################
        # # SERIALIZE EXPERIMENT
        #     out_dd = {'train_features': train_features,
        #                 'test_features': test_features,
        #                 'train_labels': train_dset.labels,
        #                 'test_labels': test_dset.labels,
        #                 # 'best_top_1_weights': all_weights_dd[best_top_1_lambda],
        #                 # 'best_top_10_weights': all_weights_dd[best_top_10_lambda]
        #                 }

        # serialize_fp = make_serialize_fp(f"n_features_{n_features}.pickle")
        # logging.info("Serializing to %s", serialize_fp)
        # with open(serialize_fp, 'wb') as f:
        #     pickle.dump(out_dd, f)
        
    
    logging.info("Finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir')
    # parser.add_argument('-plot_fp')
    parser.add_argument('-results_fp')
    parser.add_argument('-serialize_dir')
    parser.add_argument('-train_class')
    parser.add_argument('-test_class')
    parser.add_argument('-n_features', type=int, nargs='+')
    parser.add_argument('-n_train', type=int, default=None)
    parser.add_argument('-n_test', type=int, default=None)
    parser.add_argument('-max_L', type=int, default=5)
    parser.add_argument('-max_n_deltas', type=int, default=2000)
    parser.add_argument('-n_radial_params', type=int, default=5)
    parser.add_argument('-max_radial_param', type=float, default=1.)
    parser.add_argument('-weight_variance', type=float, default=1.)
    parser.add_argument('-bump_width', type=float, default=0.25)
    parser.add_argument('-standardize_matrix_cols', action='store_true', default=False)

    parser.add_argument('-n_cores', type=int, default=None)
    parser.add_argument('-chunksize', type=int, default=50)
    parser.add_argument('-precompute_chunksize', type=int, default=5)
    parser.add_argument('-l2_reg', type=float, nargs='+', default=[0.])
    parser.add_argument('-normalize_features', default=False, action='store_true')
    parser.add_argument('-seed', default=1234, type=int)
    parser.add_argument('-validation_set_fraction', default=0.1)

    a = parser.parse_args()


    logging.basicConfig(level=logging.DEBUG,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)
