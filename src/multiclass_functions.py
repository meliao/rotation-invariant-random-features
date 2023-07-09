import logging
from typing import Type, Dict, List, Tuple
import multiprocessing
from timeit import default_timer
from scipy.sparse.linalg import lsqr, svds
from scipy import linalg
from src.DataSet import DataSet, ShapeNetEncoding
from src.atom_encoding import N_CHARGES, FCHL_feature_matrix_row_from_fp, numba_FCHL_feature_matrix_row, FCHLEncoding
from src.kernels import compute_sum_of_features
import numpy as np
import numba

FEATURE_MATRIX_NORM_CONSTANT = lambda x: 1 / 10


def feature_matrix_from_dset_FCHL(dset: FCHLEncoding, 
                                    weights: np.ndarray, 
                                    intercept: bool, 
                                    n_cores: int, 
                                    chunksize: int,
                                    precompute_chunksize: int,
                                    max_L: int,
                                    detailed_time_logging: bool=False) -> np.ndarray:
    """Takes in an FCHLEncoding object, as well as a bunch of random weights, and produces
    a feature matrix.

    Args:
        dset (Type[DataSet]): Implements the methods of a DataSet
        weights (np.ndarray): Random weights. Has shape (n_features, *) where the final dimensions
        are determined by the type of <dset>
        intercept (bool): Whether to include an intercept.

    Returns:
        np.ndarray: Feature matrix. Has shape (len(dset), dset.feature_matrix_ncol(weights, intercept))
    """
    
    time_precompute_start = default_timer()
    dset.precompute(n_cores=n_cores, chunksize=precompute_chunksize, max_L=max_L)

    precomputation_time = default_timer() - time_precompute_start

    # Determine final output size 
    n_data_points = len(dset)
    n_columns = dset.feature_matrix_ncols(weights, intercept)

    logging.info("Feature matrix should be of size (%i, %i)", n_data_points, n_columns)

    # Determine whether we are using multiprocessesing or not
    time_feature_matrix_start = default_timer()

    if n_cores is None:
        logging.info("Creating feature matrix without multiprocessesing")
        features = np.empty((n_data_points, n_columns))

        # Loop over the dataset
        for i in range(len(dset)):

            arrays = dset.precomputed_arrays[i]
            features[i] = numba_FCHL_feature_matrix_row(arrays, weights, intercept)
            if i % 100 == 0:
                logging.debug("Working on data sample %i / %i", i, n_data_points)

    else:
        n_clients = n_cores - 1
        logging.info("Creating feature matrix with 1 server and %i clients and chunksize %i", n_clients, chunksize)
        with multiprocessing.Pool(n_clients) as pool_obj:
            arguments = [(i, weights, intercept) for i in dset.precomputed_arrays]
            features_ret = pool_obj.map(FCHL_feature_matrix_row, arguments, chunksize)
        features = np.array(features_ret)

    feature_creation_time = default_timer() - time_feature_matrix_start

    time_dd = {
        'precomputation_time': precomputation_time,
        'feature_matrix_time': feature_creation_time,
    }
    logging.info("Precomputation time: %f, time per sample: %f", precomputation_time, precomputation_time / n_data_points)
    logging.info("Feature matrix creation time: %f, time per sample: %f", feature_creation_time, feature_creation_time / n_data_points)
    
    if detailed_time_logging:
        return features, time_dd
    else:
        return features


@numba.jit(nopython=True)
def sum_of_features_parallel(a) -> np.float32:
    return compute_sum_of_features(a[0], a[1])

# TODO: Factor this out to be able to parallelize it with an argument to a function
@numba.jit(nopython=True, parallel=True)
def FCHL_feature_matrix_row(t: Tuple[List[np.ndarray], np.ndarray, bool]) -> np.ndarray:
    """_summary_

    Args:
        t (Tuple[List[np.ndarray], np.ndarray, bool]): Wrapper with the following entries:
        array_lst (List[np.ndarray]): 
        weights (np.ndarray): Has shape (n_random_features, max_L + 1, 2 * max_L + 1)
        intercept (bool): Whether or not to add an intercept

    Returns:
        np.ndarray: _description_
    """
    array_lst = t[0] # This has 25 entries
    weights = t[1]
    intercept = t[2]

    out = np.full((weights.shape[0] * N_CHARGES * N_CHARGES + int(intercept)), np.nan, dtype=np.float32)

    # Parallelizes over the first axis of the weights array
    for w_idx in numba.prange(weights.shape[0]):
        w = weights[w_idx]
        for a_idx, a in enumerate(array_lst):

            # a is an array with shape (n_ds, max_L + 1, 2 * max_L + 1, n_radial_funcs, n_radial_funcs)

            val = compute_sum_of_features(a, w).real

            # if np.isnan(val):
            #     logging.warn("NaN at sample_idx: %i, weight_idx: %i, a_idx: %i", i, w_idx, a_idx)

            out[a_idx + w_idx * N_CHARGES * N_CHARGES] = val   
    if intercept:
        out[-1] = 1.

    return out

def feature_matrix_from_dset_ModelNet(dset: ShapeNetEncoding, 
                                    weights: np.ndarray, 
                                    intercept: bool, 
                                    n_cores: int, 
                                    chunksize: int,
                                    precompute_chunksize: int,
                                    max_L: int,
                                    detailed_time_logging: bool=False) -> np.ndarray:
    """Takes in an ShapeNetEncoding object, as well as a bunch of random weights, and produces
    a feature matrix.

    Args:
        dset (ShapeNetEncoding): 
        weights (np.ndarray): Random weights. Has shape (n_features, *) where the final dimensions
        are determined by the type of <dset>
        intercept (bool): Whether to include an intercept.

    Returns:
        np.ndarray: Feature matrix. Has shape (len(dset), dset.feature_matrix_ncol(weights, intercept))
    """
    
    time_precompute_start = default_timer()
    dset.precompute(n_cores=n_cores, chunksize=precompute_chunksize, max_L=max_L)

    precomputation_time = default_timer() - time_precompute_start

    # Determine final output size 
    n_data_points = len(dset)
    n_columns = dset.feature_matrix_ncols(weights, intercept)

    logging.info("Feature matrix should be of size (%i, %i)", n_data_points, n_columns)

    # Determine whether we are using multiprocessesing or not
    time_feature_matrix_start = default_timer()

    if n_cores is None:
        logging.info("Creating feature matrix without multiprocessesing")
        features = np.empty((n_data_points, n_columns))

        # Loop over the dataset
        for i in range(len(dset)):

            # arrays = dset.precomputed_arrays[i]
            features[i] = dset.feature_matrix_row(i, weights, intercept)
            if i % 100 == 0:
                logging.debug("Working on data sample %i / %i", i, n_data_points)

    else:
        n_clients = n_cores - 1
        logging.info("Creating feature matrix with 1 server and %i clients and chunksize %i", n_clients, chunksize)
        with multiprocessing.Pool(n_clients) as pool_obj:
            arguments = [(i, weights, intercept) for i in range(n_data_points)]
            features_ret = pool_obj.map(dset.feature_matrix_row_parallel, arguments, chunksize)
        features = np.array(features_ret)


    feature_creation_time = default_timer() - time_feature_matrix_start

    time_dd = {
        'precomputation_time': precomputation_time,
        'feature_matrix_time': feature_creation_time,
    }
    logging.info("Precomputation time: %f, time per sample: %f", precomputation_time, precomputation_time / n_data_points)
    logging.info("Feature matrix creation time: %f, time per sample: %f", feature_creation_time, feature_creation_time / n_data_points)
    
    if detailed_time_logging:
        return features, time_dd
    else:
        return features



def _least_squares_soln(u: np.ndarray, s: np.ndarray, v_t: np.ndarray, labels: np.ndarray, l2_reg: float) -> np.ndarray:
    """
        w = (X^TX + lambda I)^{-1}X^Ty

        Using SVD: X = USV^T then we have 
        w = V(S^2 + lambda I)^{-1} S^T U^Ty

        where (S^2 + lambda I)^{-1} S^T =: D is a diagonal matrix with easy entries to compute
    Args:
        u, s, v_t (np.ndarray): Result of calling np.linalg.svd(feature_matrix, full_matrices=False)
        labels (np.ndarray): Labels that we are trying to fit
        l2_reg (float): Also known as lambda, the regularization parameter

    Returns:
        np.ndarray: One-dimension array of weights.
    """
    s_2 = np.square(s)

    # out = []
    # for reg in l2_reg:            
    pseudo_inv_denom = s_2 + l2_reg * np.ones_like(s)

    pseudo_inv_diag = np.divide(s, pseudo_inv_denom)
    pseudo_inv_diag = pseudo_inv_diag[:,np.newaxis]
    
    
    scaled_u_t = np.multiply(pseudo_inv_diag, u.transpose())
    A = np.matmul(v_t.transpose(), scaled_u_t)
    weights = np.matmul(A, labels).flatten()

    return weights


def _find_positive_class(pos_class: int, labels: np.ndarray) -> np.ndarray:
    bool_arr = labels == pos_class
    return bool_arr.astype(int)

def get_multiclass_weights(feature_mat: np.ndarray, labels: np.ndarray, l2_reg: List[float]) -> Dict:
    """_summary_

    Args:
        feature_mat (np.ndarray): _description_
        labels (np.ndarray): _description_
        l2_reg (np.ndarray): _description_

    Returns:
        Dict: _description_
    """
    
    u, s, v_t = np.linalg.svd(feature_mat, full_matrices=False)

    logging.info("Ridge regression. SVD returned %i singular vals. Minimum singular val: %f", s.shape[0], s[-1])

    out_dd = {}
    for lambda_val in l2_reg:
        lambda_dd = {}
        for lab in np.unique(labels):

            labels_i = _find_positive_class(lab, labels)

            weights = _least_squares_soln(u, s, v_t, labels_i, lambda_val)
            lambda_dd[lab] = weights

        out_dd[lambda_val] = lambda_dd
    return out_dd, s


def get_regression_weights_scipy_solve(X: np.ndarray, y: np.ndarray, l2_reg: float=0.001) -> np.ndarray:
    """
    First, form normal equation. Then, use linalg.solve() to solve the square linear 
    system. 
    """
    logging.info("Forming the Normal equations")
    X_t_X = np.matmul(X.transpose(), X)
    logging.info("Formed X^TX")
    X_t_y = np.matmul(X.transpose(), y)
    logging.info("Formed X^Ty")

    for i in range(X_t_X.shape[0]):
        X_t_X[i, i] += l2_reg

    logging.info("Finished conditioning")

    logging.info("Calling scipy.linalg.solve")

    ret_val = linalg.solve(X_t_X, X_t_y)
    logging.info("scipy.linalg.solve returned weights of shape: %s", ret_val.shape)
    return ret_val




def get_regression_weights(feature_mat: np.ndarray, y: np.ndarray, l2_reg: List[float]) -> Dict:
    """Finds ridge regression solutions to ||Ax - y||_2^2 + \lambda || x||_2^2

    Args:
        feature_mat (np.ndarray): Has shape (n_samples, n_features)
        y (np.ndarray): Has shape (n_samples)
        l2_reg (List[float]): List of regularization lambda values

    Returns:
        Dict: Keys are the elements of l2_reg
    """
    u, s, v_t = linalg.svd(feature_mat, full_matrices=False)
    logging.info("SVD returned %i singular vals with range: [%f, %f] and cond number: %f", s.shape[0], s[0], s[-1], s[0] / s[-1])

    out_dd = {}
    for lambda_val in l2_reg:

        weights = _least_squares_soln(u, s, v_t, y, lambda_val)

        out_dd[lambda_val] = weights

    return out_dd, s

def get_regression_weights_truncated_SVD(feature_mat: np.ndarray, y: np.ndarray, l2_reg: List[float], truncate_k: int) -> Dict:
    """This function solves the ridge regression problem approximately by finding a truncated SVD and using that to solve 
    the problem. This is equivalent to exactly solving the ridge regression problem on the best (Frobenius-norm) rank-K approximation
    of the original feature matrix. 

    Args:
        feature_mat (np.ndarray): Feature matrix. Has shape (n_rows, n_cols)
        y (np.ndarray): A vector of responses. Has shape (n_rows,)
        l2_reg (List[float]): List of regularization lambda values.
        truncate_k (int): How many singular values to compute

    Returns:
        Dict: Keys are elements of <l2_reg>
    """

    # First compute truncated SVD


    # Let feature_mat have shape (n_rows, n_cols)
    # Let m = min(n_rows, n_cols)
    # u has shape (n_rows, k)
    # s has shape (k,)
    # v_t has shape (k, n_cols)

    logging.info("Called SVDS with truncation level %i", truncate_k)
    u, s, v_t = scipy.sparse.linalg.svds(feature_mat, truncate_k)
    logging.info("SVDS returned.")

    # u_trunc = u
    # s_trunc = s
    # v_t_trunc = v_t

    # u has shape (n_rows, m), s has shape (m,) v_t has shape (m, n_cols)
    # u, s, v_t = linalg.svd(feature_mat, full_matrices=False)

    # u_trunc = u[:, :truncate_k]
    # s_trunc = s[:truncate_k]
    # v_t_trunc = v_t[:truncate_k, :]

    out_dd = {}
    for lambda_val in l2_reg:
        weights = _least_squares_soln_truncated_SVD(u, s, v_t, y, lambda_val)
        out_dd[lambda_val] = weights

    return out_dd


def _least_squares_soln_truncated_SVD(u: np.ndarray, 
                                      s: np.ndarray, 
                                      v_t: np.ndarray, 
                                      labels: np.ndarray, 
                                      l2_reg: float) -> np.ndarray:
    """
    Suppose original feature matrix has shape (n_rows, n_cols)
    u has shape (n_rows, k)
    s has shape (k,)
    v_t has shape (k, n_cols)
    y has shape (n_rows,)
    
    Want to return w = V(S^2 + lambda I)^{-1}SU^T y
    """
    
    s_squared = np.square(s)
    pinv_denom = s_squared + l2_reg * np.ones_like(s)
    pinv = np.divide(s,  pinv_denom)
    
    V_pinv = np.multiply(v_t.transpose(), pinv)
    
    A = np.matmul(V_pinv, u.transpose())
    weights = np.matmul(A, labels)
    return weights

def compute_top_k_accuracy(features: np.ndarray, weights_dd: Dict[int, np.ndarray], true_labels: np.ndarray, k: List[int]) -> List[float]:
    
    all_preds = np.full((features.shape[0], len(weights_dd)), np.nan)

    all_classes = np.array(list(weights_dd.keys()))
    all_classes = np.sort(all_classes)

    for i, cls in enumerate(all_classes):
        cls_weights = weights_dd[cls]
        # if v:
        #     print(f"Features shape: {features.shape}, weights shape: {cls_weights.shape}")
        all_preds[:, i] = np.matmul(features, cls_weights)
    out = [0.] * len(k)
    # Loop over all of the test examples
    for i in range(features.shape[0]):
        
        preds_i = all_preds[i]

        sorted_preds = np.flip(preds_i.argsort())

        for j, k_val in enumerate(k):
            top_k_pred_indices = sorted_preds[:k_val]
            top_k_classes = all_classes[top_k_pred_indices]

            if true_labels[i] in top_k_classes:
                out[j] += 1
    out = np.array(out)

    return out / features.shape[0]

