"""
This module provides class definitions for encoding the atomic types
"""
import multiprocessing
from typing import Generator, List, Tuple
import logging
import pickle
import os
import numpy as np
import numba
from src.DataSet import DataSet
from src.kernels import cartesian_to_spherical_batch, compute_feature_matrix_row, compute_multiple_features, compute_random_feature, compute_sum_of_features, precompute_array_for_ds

from src.data.DataClasses import DataSample3D

class MolleculeDatasetEncoding(DataSet):
    def __init__(self, coords_cart: np.ndarray, charges: np.ndarray, energies: np.ndarray=None, radial_params: np.ndarray=None) -> None:
        """
        coords_cart has shape (n_samples, max_n_atoms, 3)
        charges has shape (n_samples, max_n_atoms)
        """
        self.coords_cart = coords_cart
        self.charges = charges
        assert self.charges.shape[0] == self.coords_cart.shape[0]
        self.n_samples = charges.shape[0]

        self.energies = energies
        if self.energies is not None:
            assert self.energies.shape[0] == self.n_samples
            self.energies = self.energies.flatten()


        self.precomputed_arrays = None
        self.precomputed_bool = False
        self.precomputed_filepaths = []
        # self.RADIAL_POLYNOMIAL_PARAMS = np.array([-2.])
        if radial_params is not None:
            self.RADIAL_POLYNOMIAL_PARAMS = radial_params
        else:
            self.RADIAL_POLYNOMIAL_PARAMS = np.array([2, 4]) / 2.355

    def __len__(self) -> int:
        return self.n_samples


    def feature_matrix_row(self, i: int, weights: np.ndarray, intercept: bool) -> np.ndarray:
        """
        Given a sample index, an array of weights, and the option to add an intercept, this function
        returns a feature matrix row at the given index
        """
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
        """Assumes self.precomputed_arrays is already set. This will serialize 
        self.precomputed_arrays and store the filepaths to each one. 
        
        At the time of parallel random feature computation, we want to pass to each worker
        a filepath, and then the worker loads the filepath and returns the result.

        Args:
            n_cores (int): _description_
            chunksize (int): _description_
            max_L (int): _description_

        Returns:
            _type_: _description_
        """
        # serialize_dir = '.tmp_A_arrays_{}'
        # x = 0
        # while os.path.isdir(serialize_dir.format(x)):
        #     x += 1
        # os.mkdir(serialize_dir.format(x))
        # for i in range(len(self.precomputed_arrays)):
        #     ser_fp = os.path.join(serialize_dir.format(x), f"{i}.pickle")
        #     dump_obj = self.precomputed_arrays[i]
        #     with open(ser_fp, 'wb') as f:
        #         pickle.dump(dump_obj, f)
        #     self.precomputed_filepaths.append(ser_fp)
        return super().precompute(n_cores, chunksize, max_L)


def FCHL_feature_matrix_row_from_fp(t: Tuple[str, np.ndarray, bool]) -> np.ndarray:
    """Creates a feature matrix row in the FCHL style

    Args:
        fp (str): Filepath where the precomputation is serialized
        weights (np.ndarray): Random weights
        intercept (bool): Whether to include an intercept

    Returns:
        np.ndarray: a row in an FCHL feature matrix
    """
    fp = t[0]
    weights = t[1]
    intercept = t[2]
    with open(fp, 'rb') as f:
        arrays = pickle.load(f)
    return numba_FCHL_feature_matrix_row(arrays, weights, intercept)


# CHARGES_LIST = [1, 6, 7, 8, 16]

# CHARGE_PAIRS_LIST = [[1, 6],
#                     [1, 7],
#                     [1, 8],
#                     [1, 16],
#                     [6, 7],
#                     [6, 8],
#                     [6, 16],
#                     [7, 8],
#                     [7, 16],
#                     [8, 16]]


CHARGES_LIST_QM9 = [1, 6, 7, 8, 9]
CHARGES_LIST_QM7 = [1, 6, 7, 8, 16]

CHARGE_PAIRS_LIST = [[1, 6],
                    [1, 7],
                    [1, 8],
                    [1, 9],
                    [6, 7],
                    [6, 8],
                    [6, 9],
                    [7, 8],
                    [7, 9],
                    [8, 9]]



N_CHARGE_PAIRS = 10
N_CHARGES = 5

class ElementPairsDatasetEncoding(MolleculeDatasetEncoding):

    def __init__(self, coords_cart: np.ndarray, charges: np.ndarray, energies: np.ndarray = None) -> None:
        super().__init__(coords_cart, charges, energies)
        # self.sphe_harm_evals = None
        self.precomputed_arrays = None
        

    @staticmethod
    def _get_atoms_two_charges(coords: np.ndarray, charges: np.ndarray, charge_1: int, charge_2: int) -> np.ndarray:
        
        bool_arr_1 = charges == charge_1
        bool_arr_2 = charges == charge_2
        bool_arr = np.logical_or(bool_arr_1, bool_arr_2)
        out = coords[bool_arr].reshape((-1, 3))
        return out
    
    def  __getitem__(self, item: int) -> Generator[DataSample3D, None, None]:
        """
        Given a dataset index, this returns a generator, which yeilds DataSamples
        which include each pair of atom types
        """
        coords = self.coords_cart[item]
        charges = self.charges[item]
        for charge_pair in CHARGE_PAIRS_LIST:
            coords_cart = self._get_atoms_two_charges(coords, charges, charge_pair[0], charge_pair[1])
            ds = DataSample3D.from_cartesian_coords(coords_cart)
            yield ds

    def get_list_of_samples(self) -> List[DataSample3D]:
        """Creates a list of data samples.

        Returns:
            List[DataSample3D]: _description_
        """
        ds_lst = []
        for i in range(self.n_samples):
            new_lst_i = [j for j in self[i]]
            ds_lst.extend(new_lst_i)

        return ds_lst


    def truncate(self, n: int) -> None:
        coords_cart = self.coords_cart[:n]
        charges = self.charges[:n]

        if self.energies is not None:
            energies = self.energies[:n]
        else:
            energies = None
        return ElementPairsDatasetEncoding(coords_cart, charges, energies)

    def _simgle_padded_precompute_array(self, i: int, max_L: int) -> np.ndarray:
        """Assembles the precomputation arrays of all of the atom pairs of 
        sample i.

        TODO: Find a smart way to pad the different sizes of # of delta functions
        across atom pairs and then handle the padding downstream in one 
        of the kernel functions.

        Args:
            i (int): Sample index
            max_L (int): Determines how many spherical harmonics to evaluate

        Returns:
            np.ndarray: complex-valued array of size (n_atom_pairs, n_deltas, n_deltas, max_L+1, 2 * max_L + 1)
            where n_deltas is max number of deltas across all atom pairs in sample
        """        
        raise NotImplementedError
        # pairs_lst = [x for x in self[i]]

        # max_n_deltas = np.max([x.n_points for x in pairs_lst])
        # out = np.zeros((N_CHARGE_PAIRS, max_n_deltas, max_n_deltas, max_L + 1, 2 * max_L + 1), dtype=np.complex64)
        # for i, ds_i in enumerate(pairs_lst):
        #     out[i, :, : :ds_i.n_points] = ds_i.get_spherical_harmonic_evals(max_L)

        # return out

    def feature_matrix_ncols(self, weights: np.ndarray, intercept: bool) -> int:
        """Determines the size of a feature matrix row.

        Args:
            weights (np.ndarray): The weights used to generate the feature matrix.
            Assumed to have size (n_features, max_L + 1, 2 * max_L + 1)
            intercept (bool): Whether the feature matrix has an intercept column

        Returns:
            int: Number of columns in the feature matrix
        """
        n_features, _, _, _ = weights.shape
        return N_CHARGE_PAIRS * n_features + int(intercept)


    def feature_matrix_row(self, i: int, weights: np.ndarray, intercept: bool) -> np.ndarray:
        """Computes a row of the feature matrix.

        Args:
            i (int): Sample index
            weights (np.ndarray): Weights array. Assumed to have size (n_features, max_L + 1, 2 * max_L + 1)
            intercept (bool): whether to add an intercept column at the end of the row

        Returns:
            np.ndarray: 1D array with size given by self.feature_matrix_ncols()
        """
        arrays = self.precomputed_arrays[i]

        all_features = []
        for w in weights:
            out_lst = [compute_random_feature(x, w) for x in arrays]
            all_features.extend(out_lst)
        if intercept: 
            all_features.append(1.)
        out_arr = np.array(all_features)
        return out_arr.real


    def precompute(self, n_cores: int, chunksize: int, max_L: int) -> None:
        """Pre-assembles a list of spherical harmonic evals

        Args:
            max_L (int): determines the number of spherical harmonic evals to 
            compute
        """
        super().precompute(n_cores, chunksize, max_L)

        logging.debug("Beginning pre-computation")
        self.precomputed_arrays = []
        for i in range(self.n_samples):
            arr_lst_i = [precompute_array_for_ds(x.coords, max_L, self.RADIAL_POLYNOMIAL_PARAMS, True) for x in self[i]]
            self.precomputed_arrays.append(arr_lst_i)


            # TODO:
            # self.precomputed_arrays.append(self._precompute_array(i, max_L))
        logging.debug("Finished pre-computation")
        

class FCHLEncoding(MolleculeDatasetEncoding):
    def __init__(self, 
                    coords_cart: np.ndarray, 
                    charges: np.ndarray, 
                    thermochemical_energies: np.ndarray=None, 
                    internal_energies: np.ndarray=None,
                    atomization_energies: np.ndarray=None,
                    QM7_bool: bool=False,
                    radial_params: np.ndarray=None) -> None:

        # atomization_energies = None
        if thermochemical_energies is not None and internal_energies is not None:
            atomization_energies = internal_energies - thermochemical_energies

        super().__init__(coords_cart, charges, atomization_energies, radial_params=radial_params)
        self.precomputed_arrays = None
        self.n_rf_evals = 0
        self.max_L = None

        if QM7_bool:
            self.CHARGES_LST = CHARGES_LIST_QM7
        else:
            self.CHARGES_LST = CHARGES_LIST_QM9


        self.atomization_energies = atomization_energies
        self.internal_energies = internal_energies
        self.thermochemical_energies = thermochemical_energies

        if self.atomization_energies is not None:
            self.atomization_energies = self.atomization_energies.flatten()

        if self.internal_energies is not None:
            self.internal_energies = self.internal_energies.flatten()
            self.thermochemical_energies = self.thermochemical_energies.flatten()
        # if atomization_energies is not None and internal_energies is not None:
        #     self.energy_diffs = atomization_energies - internal_energies

        logging.info("Initialized an FCHL encoding. Charges list: %s, Radial params: %s", self.CHARGES_LST, self.RADIAL_POLYNOMIAL_PARAMS)

    def truncate(self, n: int) -> None:
        coords_cart = self.coords_cart[:n]
        charges = self.charges[:n]

        atomization_energies = None
        thermo_energies = None
        internal_energies = None
        if self.atomization_energies is not None:
            atomization_energies = self.atomization_energies[:n]
        if self.thermochemical_energies is not None:
            thermo_energies = self.thermochemical_energies[:n]
        if self.internal_energies is not None:
            internal_energies = self.internal_energies[:n]
        # else:
        #     thermo_energies = None
        #     internal_energies = None
        qm7_bool = 16 in self.CHARGES_LST
        
        return FCHLEncoding(coords_cart, charges, thermochemical_energies=thermo_energies, internal_energies=internal_energies, QM7_bool=qm7_bool, radial_params=self.RADIAL_POLYNOMIAL_PARAMS)

    def precompute(self, n_cores: int, chunksize: int, max_L: int) -> None:

        if self.precomputed_bool:
            return
        logging.debug("Beginning pre-computation")

        self.max_L = max_L

        if n_cores is None:

            # precomputed_arrays has length self.n_samples
            # Each entry is a list of length N_CHARGES * N_CHARGES
            # Each entry in the inner list is an array of size (n_ds, max_L + 1, 2 * max_L + 1, n_radial_funcs)
            # where n_ds changes depending on the number oof atoms of a certain molecule type.
            self.precomputed_arrays = []

            for i in range(self.n_samples):

                # Just using the parallel function in series here
                self.precomputed_arrays.append(self._precompute_parallel(i))
        else:
            n_clients = n_cores - 1 
            with multiprocessing.Pool(n_clients) as pool_obj:
                logging.debug("Preprocessesing with 1 server and %i clients", n_clients)
                args = np.arange(self.n_samples)
                self.precomputed_arrays = pool_obj.map(self._precompute_parallel, args, chunksize=chunksize)
        super().precompute(n_cores, chunksize, max_L)

        logging.debug("Finished pre-computation")


    def _precompute_parallel(self, i: int) -> List[np.ndarray]:
        """This is a parallel helper function called when doing parallel 
        precomputation. It works on a single datasample, specified by i

        Args:
            i (int): Index of the datasample to work on 

        Returns:
            List[np.ndarray]: A list of length N_CHARGES * N_CHARGES, where each 
            element is an array of size (n_ds, max_L + 1, 2 * max_L + 1, n_radial_funcs)
        """

        out = []
        coords_i = self.coords_cart[i]
        charges_i = self.charges[i]
        datasamples_i = numba_get_datasamples_for_row(charges_i, coords_i, self.CHARGES_LST)
        # datasamples_i = self.get_datasamples_for_row(i)
        for ds_lst in datasamples_i: #loop of length N_CHARGES * N_CHARGES
            out.append(self._ds_lst_to_array(ds_lst))

        return out

    def feature_matrix_ncols(self, weights: np.ndarray, intercept: bool) -> int:
        return weights.shape[0] * N_CHARGES * N_CHARGES + int(intercept)

    def expected_weights_shape(self, n_features: int, max_L: int) -> Tuple[int]:
        return (n_features, max_L + 1, 2 * max_L + 1, self.RADIAL_POLYNOMIAL_PARAMS.shape[0])

    def _ds_lst_to_array(self, ds_lst: List[np.ndarray]) -> np.ndarray:
        """Given a list of DataSamples, this produces an array stacking their
        precomputation tensor evaluations. The precomputed tensor evals come 
        from calls to precompute_array_for_ds()

        Args:
            ds_lst (List[np.ndarray]): A variable-length list of DataSamples. 
                Has length n_ds. Each DataSample is assumed to have the same number
                of points n_points.

        Returns:
            np.ndarray: Complex-valued array with shape 
                (n_ds, max_L + 1, 2 * max_L + 1, n_radial_funcs)
        """
        if len(ds_lst):
            n_ds = len(ds_lst)
            out = np.zeros((n_ds, 
                                self.max_L + 1, 
                                2 * self.max_L + 1, 
                                self.RADIAL_POLYNOMIAL_PARAMS.shape[0], 
                                self.RADIAL_POLYNOMIAL_PARAMS.shape[0]), dtype=np.complex64)

            for i in range(n_ds):
                coords_i = ds_lst[i]
                out[i] = precompute_array_for_ds(coords=coords_i, 
                                                    max_L=self.max_L, 
                                                    radial_params=self.RADIAL_POLYNOMIAL_PARAMS, 
                                                    params_are_centers_bool=False,
                                                    normalize_functions_bool=False,
                                                    width_param=np.nan)

            return out
        else:
            empty_return_shape = (1,
                                    0,
                                    0,
                                    self.RADIAL_POLYNOMIAL_PARAMS.shape[0],
                                    self.RADIAL_POLYNOMIAL_PARAMS.shape[0])
            return np.zeros(empty_return_shape, dtype=np.complex64)


@numba.jit(nopython=True)
def numba_get_datasamples_for_row(charges_i: np.ndarray, coords_i: np.ndarray, charges_lst: List[int]) -> List[List[np.ndarray]]:
    """Given a sample index <i>, this function returns a list of lists of lists
    of coordinate arrays.

    [[DS(H) @ H_0, DS(H) @ H_1, ...], 
        [DS(C) @ H_0, DS(C) @ H_1, ...], 
        [DS(N) @ H_0, DS(N) @ H_1, ...],
        ...]

    Args:
        charges_i (np.ndarray): Has shape (n_atoms,)
        coords_i (np.ndarray): Has shape (n_atoms, 3). This is CARTESIAN COORDINATES

    Returns:
        List[List[np.ndarray]]: Outer list has length N_CHARGES ** 2. Inner
        lists have variable length, and some may be empty. 
    """
    out = []

    for charge_1 in charges_lst:
        for charge_2 in charges_lst:
            charges = (charge_1, charge_2)
            out.append(numba_get_datasamples(charges_i, coords_i, charges))
    return out


@numba.jit(nopython=True)
def numba_get_datasamples(charges_i: np.ndarray, coords_i: np.ndarray, charges: Tuple[int, int]) -> List[np.ndarray]:
    """Given charge 1 and charge 2, return a list of all of the atoms of type charge 2, centered
    at each appearance of atoms of charge 1.

    In other words, return

    [DS(<charge_2>) @ <charge_1>_0, DS(<charge_2>) @ <charge_1>_1, DS(<charge_2>) @ <charge_1>_2, ...]

    Args:
        charges_i (np.ndarray): Has shape (n_atoms,)
        coords_i (np.ndarray): Has shape (n_atoms, 3). This is CARTESIAN COORDINATES
        charges (Tuple[int, int]): A pair of atomic charges. The first charge chooses the atoms 
            at which we center the molecule. The second charge chooses which atoms we keep in the 
            DataSample.

    Returns:
        List[np.ndarray]: Variable length list. Will have length == the number of atoms with 
            charge <charges[0]> in the molecule. This is SPHERICAL COORDINATES
    """
    # logging.info("New call to get_datasamples(). Charges are %s", charges)

    if charges[0] == charges[1]:
        # Here we do something a little bit different when the charges are 
        # the same. We center the molecule on a given atom, and then 
        # we remove that atom from the data sample. 
        bool_arr = charges_i == charges[0]

        coords_arr = coords_i[bool_arr].reshape((-1, 3))
        # logging.info("Charges are the same; found coords of shape %s", coords_arr.shape)
        out = []
        for idx, loc in enumerate(coords_arr):
            # mask = np.ones(coords_arr.shape[0], dtype=bool)
            mask = np.array([True] * coords_arr.shape[0])
            mask[idx] = False

            new_coords = coords_arr[mask]
            new_coords = new_coords - loc
            new_coords_sphe = cartesian_to_spherical_batch(new_coords)
            out.append(new_coords_sphe)
        return out
    else:

        bool_arr_1 = charges_i == charges[0]
        bool_arr_2 = charges_i == charges[1]

        coords_1 = coords_i[bool_arr_1].reshape((-1, 3))
        coords_2 = coords_i[bool_arr_2].reshape((-1, 3))
        # logging.info("Charges are different; found coords of shape %s for 1 and %s for 2", coords_1.shape, coords_2.shape)

        out = []

        if coords_2.shape[0]:
            for loc in coords_1:
                coords_new = coords_2 - loc
                coords_new_sphe = cartesian_to_spherical_batch(coords_new)
                out.append(coords_new_sphe)

        return out


@numba.jit(nopython=True)
def numba_FCHL_feature_matrix_row(array_lst: List[np.ndarray], weights: np.ndarray, intercept: bool) -> np.ndarray:
    """_summary_

    Args:
        array_lst (List[np.ndarray]): Length N_CHARGES ** 2, each array has shape (n_s, max_L + 1, 2 * max_L + 1, n_radial_funcs)
        weights (np.ndarray): Random weights. Has shape (n_features, max_L + 1, 2 * max_L + 1, n_radial_funcs)

    Returns:
        np.ndarray: Of shape (n_features * N_CHARGES ** 2)
    """
    out = np.full((weights.shape[0] * (N_CHARGES ** 2) + int(intercept),), np.nan, dtype=np.float32)
    for w_idx in range(weights.shape[0]):
        w = weights[w_idx]
        for a_idx in range(len(array_lst)):
            a = array_lst[a_idx]
            val = compute_sum_of_features(a, w).real
            out[a_idx + w_idx * N_CHARGES * N_CHARGES] = val

    if intercept:
        out[-1] = 1.

    return out

class WholeMoleculeDatasetEncodimg(MolleculeDatasetEncoding):
    def __init__(self, coords_cart: np.ndarray, charges: np.ndarray, energies: np.ndarray = None) -> None:
        super().__init__(coords_cart, charges, energies)
        self.precompute_arrays = None

    def __getitem__(self, item: int) -> Generator[DataSample3D, None, None]:
        charges = self.charges[item]
        coords = self.coords_cart[item]
        coords_idx = charges != 0
        ds_i = DataSample3D.from_cartesian_coords(coords[coords_idx])

        return ds_i

    def feature_matrix_ncols(self, weights: np.ndarray, intercept: bool) -> int:
        """Determines the size of a feature matrix row.

        Args:
            weights (np.ndarray): The weights used to generate the feature matrix.
            Assumed to have size (n_features, max_L + 1, 2 * max_L + 1)
            intercept (bool): Whether the feature matrix has an intercept column

        Returns:
            int: Number of columns in the feature matrix
        """
        return weights.shape[0] + int(intercept)

    

    def precompute(self, n_cores: int, chunksize: int, max_L: int) -> None:
        super().precompute(n_cores, chunksize, max_L)
        logging.debug("Beginning pre-computation")
        n_points = len(self)
        if n_cores is None:
            self.precompute_arrays = []
            for i in range(n_points):
                ds_i = self[i]
                self.precompute_arrays.append(precompute_array_for_ds(ds_i.coords, max_L, self.RADIAL_POLYNOMIAL_PARAMS, True))
                logging.debug("Done precomputation %i / %i", i, n_points)
        else:
            n_clients = n_cores - 1
            with multiprocessing.Pool(n_clients) as pool_obj: 
                logging.debug("Preprocessesing with 1 server and %i clients", n_clients)
                arguments = []
                for i in range(n_points):
                    ds_i = self[i]
                    arguments.append((ds_i.coords, max_L, self.RADIAL_POLYNOMIAL_PARAMS, True))
                self.precompute_arrays = pool_obj.map(self.precompute_parallel, arguments, chunksize)

        logging.debug("Finished pre-computation")

    @staticmethod
    def precompute_parallel(args: Tuple[np.ndarray, int, np.ndarray, bool]) -> np.ndarray:
        return precompute_array_for_ds(args[0], args[1], args[2], args[3])

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