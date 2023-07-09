from typing import Tuple, List
import logging
import multiprocessing
import numpy as np
from src.data.DataClasses import DataSample3D

from src.kernels import precompute_array_for_ds, compute_multiple_features

class DataSet:
    def __init__(self) -> None:
        self.precomputed_bool = False
    def feature_matrix_row(self, i: int, weights: np.ndarray, intercept: bool) -> np.ndarray:
        """
        Given a sample index, an array of weights, and the option to add an intercept,
        this function returns a feature matrix row at the given index."""
        pass

    def feature_matrix_row_parallel(self, t: Tuple[int, np.ndarray, bool]) -> np.ndarray:
        return self.feature_matrix_row(t[0], t[1], t[2])

    def feature_matrix_ncols(self, weights: np.ndarray, intercept: bool) -> int:
        """
        Computes the size of a feature matrix row when using the given weights and 
        the intercept
        """
        pass

    def precompute(self, n_cores: int, chunksize: int, max_L: int) -> None:
        """
        Should be called before assembling a feature matrix. Can accept keyword
        arguments
        """
        self.precomputed_bool = True

    def expected_weights_shape(self, max_L: int) -> Tuple[int]:
        """
        Given a max_L parameter, this should return the expected shape of weight
        matrix required
        """
        pass


class ShapeNetEncoding(DataSet):
    def __init__(self, 
                coords_cart: List[np.ndarray], 
                max_L: int, 
                radial_params: np.ndarray, 
                labels: np.ndarray=None,
                bump_width: float=None, 
                poly_rad_funcs_bool: bool=True) -> None:
        """_summary_

        Args:
            coords_cart (np.ndarray): _description_
            max_L (int): _description_
        """
        super().__init__()

        self.coords_cart = coords_cart
        self.n_samples = len(coords_cart)
        self.max_L = max_L
        self.labels = labels
        self.energies = labels 

        self.precompute_arrays = None
        self.radial_params = radial_params
        self.PARAMS_ARE_CENTERS_BOOL = poly_rad_funcs_bool
        self.bump_width = bump_width


    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> DataSample3D:
        return DataSample3D.from_cartesian_coords(self.coords_cart[item])

    def precompute(self, n_cores: int, chunksize: int, max_L: int) -> None:
        if self.precomputed_bool:
            return
        super().precompute(n_cores, chunksize, max_L)
        logging.debug("Beginning pre-computation")
        n_points = len(self)
        if n_cores is None:
            self.precompute_arrays = []
            for i in range(n_points):
                ds_i = self[i]
                self.precompute_arrays.append(precompute_array_for_ds(ds_i.coords, 
                                                                        max_L, 
                                                                        self.radial_params, 
                                                                        self.PARAMS_ARE_CENTERS_BOOL,
                                                                        True,
                                                                        self.bump_width))
                logging.debug("Done precomputation %i / %i", i, n_points)
        else:
            n_clients = n_cores - 1
            with multiprocessing.Pool(n_clients) as pool_obj: 
                logging.debug("Preprocessesing with 1 server and %i clients", n_clients)
                arguments = []
                for i in range(n_points):
                    ds_i = self[i]
                    arguments.append((ds_i.coords, 
                                        max_L, 
                                        self.radial_params, 
                                        self.PARAMS_ARE_CENTERS_BOOL,
                                        self.bump_width))
                self.precompute_arrays = pool_obj.map(self.precompute_parallel, arguments, chunksize)

        logging.debug("Finished pre-computation")

    @staticmethod
    def precompute_parallel(args: Tuple[np.ndarray, int, np.ndarray, bool, float]) -> np.ndarray:
        return precompute_array_for_ds(args[0], args[1], args[2], args[3], True, args[4])
        
    def feature_matrix_ncols(self, weights: np.ndarray, intercept: bool) -> int:
        return weights.shape[0] + int(intercept)

    def feature_matrix_row(self, i: int, weights: np.ndarray, intercept: bool) -> np.ndarray:
        """Computes a row of the feature matrix. A row is just the random feature
        evaluated on all of the atoms. No special weighting or pair encoding.

        Args:
            i (int): sample index
            weights (np.ndarray): Weights array. Assumed to have size (n_features, max_L + 1, 2 * max_L + 1)
            intercept (bool): whether to add an intercept column at the end of the row

        Returns:
            np.ndarray: 1D array with size given by self.feature_matrix_ncols()
        """
        arr_i = self.precompute_arrays[i]

        x  = compute_multiple_features(arr_i, weights, intercept)

        return x.real
