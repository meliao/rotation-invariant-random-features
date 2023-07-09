import numpy as np
import unittest
from src.data.DataClasses import DataSample2D, DataSample3D

def random_data_sample_2d(n: int) -> DataSample2D:
    return DataSample2D.from_cartesian_coords(np.random.normal(size=(n,2)))

def random_data_sample_3d(n: int) -> DataSample3D:
    return DataSample3D.from_cartesian_coords(np.random.normal(size=(n,3)))


def check_array_equality(a: np.ndarray, 
                            b: np.ndarray, 
                            a_name: str='a', 
                            b_name: str='b', 
                            msg: str='',
                            atol: float=1e-8, 
                            rtol: float=1e-05) -> None:
    max_diff = np.max(np.abs(a - b))
    samp_n = 5
    difference_count = np.logical_not(np.isclose(a, b, atol=atol, rtol=rtol)).sum()

    # Compute relative difference
    x = a.flatten()
    y = b.flatten()
    bool_arr = x >= 1e-15
    rel_diffs = np.abs((x[bool_arr] - y[bool_arr]) / x[bool_arr])
    if rel_diffs.size == 0:
        return
    max_rel_diff = np.max(rel_diffs)
    s = msg + "Arrays differ in {} / {} entries. Max absolute diff: {}; max relative diff: {}".format(difference_count, 
                                                                                            a.size, 
                                                                                            max_diff,
                                                                                            max_rel_diff)
    assert np.allclose(a, b, atol=atol, rtol=rtol), s


def check_scalars_close(a,
                        b, 
                        a_name: str='a', 
                        b_name: str='b', 
                        msg: str='',
                        atol=1e-08, 
                        rtol=1e-05):
    max_diff = np.max(np.abs(a - b))
    s = msg + 'Max diff: {:.8f}, {}: {}, {}: {}'.format(max_diff, a_name, a, b_name, b)
    assert np.allclose(a, b, atol=atol, rtol=rtol), s


def check_no_nan_in_array(arr: np.ndarray) -> None:

    nan_points = np.argwhere(np.isnan(arr))

    s = f"Found NaNs in arr of shape {arr.shape}. Some of the points are at indices {nan_points.flatten()[:5]}"

    assert not np.any(np.isnan(arr)), s