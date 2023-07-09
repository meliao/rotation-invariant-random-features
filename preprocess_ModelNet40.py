import logging
import argparse
from typing import Dict, List, Tuple
import os

import numpy as np
import h5py
logging.getLogger('numba').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)

from scipy import io
from scipy import stats

from src.utils import write_result_to_file
from src.data.ModelNet_utils import points_to_shapenet_encoding
from src.multiclass_functions import feature_matrix_from_dset_ModelNet

FMT = "%(asctime)s:classification_ShapeNet: %(levelname)s - %(message)s"
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
    


def main(args) -> None:
    """
    1. Load the ModelNet hdf5 files.
    2. Randomly generate z axis rotations.
    3. Randomly generate SO(3) rotations.
    4. Save data into .npy files.
    """

    np.random.seed(1234)

    if not os.path.isdir(args.data_dir_out):
        os.mkdir(args.data_dir_out)

    ###############################################
    # LOAD HDF5 FILES

    train_samples, train_labels = create_dataset(args.data_dir_in, 'train_files.txt', None, args.n_deltas, False)
    test_samples, test_labels = create_dataset(args.data_dir_in, 'test_files.txt', None, args.n_deltas, False)

    logging.info("Loaded %i train and %i test samples", train_samples.shape[0], test_samples.shape[0])


    ###############################################
    # SAVE UNPERTURBED FILES
    logging.info("Saving unperturbed files")

    train_samples_fp = os.path.join(args.data_dir_out, 'train_samples.npy')
    np.save(train_samples_fp, train_samples)
    train_labels_fp = os.path.join(args.data_dir_out, 'train_labels.npy')
    np.save(train_labels_fp, train_labels)

    test_samples_fp = os.path.join(args.data_dir_out, 'test_samples.npy')
    np.save(test_samples_fp, test_samples)
    test_labels_fp = os.path.join(args.data_dir_out, 'test_labels.npy')
    np.save(test_labels_fp, test_labels)



    ###############################################
    # RANDOMLY GENERATE Z ROTATIONS
    logging.info("Creating z perturbations")

    so_2_sampler = stats.special_ortho_group(2)

    z_rotations_train = np.zeros((train_samples.shape[0], 3, 3))
    z_rotations_train[:, 2, 2] = np.ones_like(z_rotations_train[:, 2, 2])
    z_rotations_train[:, :2, :2] = so_2_sampler.rvs(train_samples.shape[0])

    train_samples_z = np.full_like(train_samples, np.nan)
    for i in range(train_samples.shape[0]):
        train_samples_z[i] = np.matmul(train_samples[i], z_rotations_train[i])

    train_samples_z_fp = os.path.join(args.data_dir_out, 'train_samples_z.npy')
    np.save(train_samples_z_fp, train_samples_z)

    z_rotations_test = np.zeros((test_samples.shape[0], 3, 3))
    z_rotations_test[:, 2, 2] = np.ones_like(z_rotations_test[:, 2, 2])
    z_rotations_test[:, :2, :2] = so_2_sampler.rvs(test_samples.shape[0])

    test_samples_z = np.full_like(test_samples, np.nan)
    for i in range(test_samples.shape[0]):
        test_samples_z[i] = np.matmul(test_samples[i], z_rotations_test[i])
        
    test_samples_z_fp = os.path.join(args.data_dir_out, 'test_samples_z.npy')
    np.save(test_samples_z_fp, test_samples_z)   

    #################################################
    # RANDOMLY GENERATE SO(3) ROTATIONS

    logging.info("Creating SO(3) perturbations")

    so_3_sampler = stats.special_ortho_group(3)
    so3_rotations_train = so_3_sampler.rvs(train_samples.shape[0])

    train_samples_so3 = np.full_like(train_samples, np.nan)
    for i in range(train_samples.shape[0]):
        train_samples_so3[i] = np.matmul(train_samples[i], so3_rotations_train[i])

    fp_train_samples_so3 = os.path.join(args.data_dir_out, 'train_samples_so3.npy')
    np.save(fp_train_samples_so3, train_samples_so3)


    so3_rotations_test = so_3_sampler.rvs(test_samples.shape[0])
    test_samples_so3 = np.full_like(test_samples, np.nan)
    for i in range(test_samples.shape[0]):
        test_samples_so3[i] = np.matmul(test_samples[i], so3_rotations_test[i])

    fp_test_samples_so3 = os.path.join(args.data_dir_out, 'test_samples_so3.npy')
    np.save(fp_test_samples_so3, test_samples_so3)

    logging.info("Finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir_in')
    parser.add_argument('-data_dir_out')
    parser.add_argument('-n_deltas', type=int)

    a = parser.parse_args()


    logging.basicConfig(level=logging.DEBUG,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)
