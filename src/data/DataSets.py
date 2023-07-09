import logging
from typing import Tuple, Type, List

from scipy import io
import numpy as np
from src.atom_encoding import ElementPairsDatasetEncoding, FCHLEncoding, MolleculeDatasetEncoding


class QM7Data:
    def __init__(self, fp: str, 
                    encoding_type: Type[MolleculeDatasetEncoding]=ElementPairsDatasetEncoding,
                    keep_Coloumb_mat: bool=False) -> None:
        data = io.loadmat(fp)

        # Atomization energies (labels)
        self.T = data['T'].flatten()
        # Splits for CV
        self.P = data['P']
        # Cartesian coordinates of atoms (size 7165 x 23 x 3)
        self.R = data['R']
        # Charges of atoms (size 7165 x 23)
        self.Z = data['Z']

        # Keep track of the folds that have been released so far
        self.next_fold = 0

        if keep_Coloumb_mat:
            self.X = data['X']

        self.encoding_type = encoding_type

    # def get_dataset(self, n_folds: int) -> FCHLEncoding:
    #     if n_folds + self.next_fold >= 6:
    #         raise ValueError("Not enough CV folds in the dataset.")
    #     logging.info("QM7 dataset. Releasing folds %i - %i", self.next_fold, self.next_fold + n_folds)
    #     idxes = self.P[self.next_fold: self.next_fold + n_folds].flatten()

    #     charges = self.Z[idxes]
    #     coords = self.R[idxes]
    #     energies = self.T[idxes]

    #     self.next_fold += n_folds

    #     dset = self.encoding_type(coords, charges, atomization_energies=energies, QM7_bool=True)
    #     return dset

    def get_charges_coords_energies(self, n_folds: int) -> Tuple[np.ndarray]:
        if n_folds + self.next_fold >= 6:
            raise ValueError("Not enough CV folds in the dataset.")
        logging.info("QM7 dataset. Releasing folds %i - %i", self.next_fold, self.next_fold + n_folds)
        idxes = self.P[self.next_fold: self.next_fold + n_folds].flatten()

        charges = self.Z[idxes]
        coords = self.R[idxes]
        energies = self.T[idxes]

        self.next_fold += n_folds

        return (charges, coords, energies)

    def get_HCNO_molecules(self) -> None:
        logging.warn("Removing all S atoms from QM7 Dataset")
        keep_bool_arr = np.empty((self.Z.shape[0]), dtype=bool)
        for i in range(self.Z.shape[0]):
            charges_i = self.Z[i]
            contains_S = np.any(charges_i == 16)

            keep_bool_arr[i] = np.logical_not(contains_S)

        self.T = self.T[keep_bool_arr]
        # self.P = self.P[keep_bool_arr]
        self.Z = self.Z[keep_bool_arr]
        self.R = self.R[keep_bool_arr]


def load_QM7(fp: str, 
                n_train: int=2, 
                n_test: int=2,
                validation_set_fraction: float=0.1,
                radial_params: np.ndarray=None,
                perm: np.ndarray=None) -> Tuple[FCHLEncoding, FCHLEncoding, FCHLEncoding]:
    """Loads the QM7 dataset but NEVER loads the final fold (which will be 
    used as a held-out validation set). 

    Args:
        fp (str): path to the QM7 file
        n_train_folds (int, optional): Number of folds (each of size 1433) to be included in the train dataset. Defaults to 2.
        n_test_folds (int, optional): Number of folds (each of size 1433) to be included in the test dataset. Defaults to 2.

    Returns:
        Tuple[ElementPairsDatasetEncoding, ElementPairsDatasetEncoding]: train data and test data
    """
    q = QM7Data(fp, encoding_type=FCHLEncoding)

    n_validation = int(np.floor(n_train * validation_set_fraction))
    n_train_eff = n_train - n_validation

    assert n_train + n_test <= q.T.shape[0]



    if perm is None:
        perm = np.random.permutation(q.T.shape[0])


    train_idxes = perm[:n_train_eff]
    val_idxes = perm[n_train_eff:n_train]
    test_idxes = perm[n_train:n_train + n_test]

    assert np.intersect1d(train_idxes, val_idxes).shape[0] == 0
    assert np.intersect1d(train_idxes, test_idxes).shape[0] == 0
    assert np.intersect1d(test_idxes, val_idxes).shape[0] == 0

    train_dset = FCHLEncoding(q.R[train_idxes],
                                q.Z[train_idxes],
                                atomization_energies=q.T[train_idxes],
                                QM7_bool=True,
                                radial_params=radial_params)

    val_dset = FCHLEncoding(q.R[val_idxes],
                                q.Z[val_idxes],
                                atomization_energies=q.T[val_idxes],
                                QM7_bool=True,
                                radial_params=radial_params)

    test_dset = FCHLEncoding(q.R[test_idxes],
                            q.Z[test_idxes],
                            atomization_energies=q.T[test_idxes],
                            QM7_bool=True,
                            radial_params=radial_params)

    return (train_dset, val_dset, test_dset)




CHARGES_DICT = {'H': 1,
               'C': 6,
               'N': 7,
               'O': 8,
               'F': 9}

def parse_xyz_file(fp: str) -> Tuple[np.ndarray, np.ndarray, np.float32]:
    """
    Returns coords (n_atoms, 3), charges (n_atoms), features (16,)
    """
    
    with open(fp) as f:
        n_atoms = int(f.readline().strip())
        
        features = np.array([float(x) for x in f.readline().split()[1:]])
        
        coords = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        charges = np.full((n_atoms,), np.nan, dtype=np.int32)
        
        for i in range(n_atoms):
            line_lst = f.readline().split()
            charge_str = line_lst[0]
            charges[i] = CHARGES_DICT[charge_str]

            for j in range(1, 4):
                try:
                    coords[i, j-1] = float(line_lst[j])
                except ValueError:
                    # raise ValueError(fp)
                    vals = line_lst[j].split("*^")
                    logging.warn("Saw a *^ character in file %s", fp)
                    coords[i, j-1] = float(vals[0]) * (10 ** int(vals[1]))
    return coords, charges, features


def parse_chunk(fp_lst: List[str], out_fp: str) -> None:
    """
    Parses all the files in fp_lst, and then saves the output into
    out_fp with objects
    n_atoms: (n_molecules,)
    coords: (n_molecules, max_n_atoms, 3)
    charges: (n_molecules, max_n_atoms)
    features: (n_molecules, 16)
    """
    
    n_molecules = len(fp_lst)
    
    n_atoms = []
    coords = []
    charges = []
    features = []
    
    for fp in fp_lst:
        coords_i, charges_i, features_i = parse_xyz_file(fp)
        
        n_atoms.append(coords_i.shape[0])
        coords.append(coords_i)
        charges.append(charges_i)
        features.append(features_i)
        
    max_n_atoms = 29 # FCHL paper says max number of atoms across QM9 dataset is 29
    
    coords_out = np.full((n_molecules, max_n_atoms, 3), np.nan, np.float32)
    charges_out = np.full((n_molecules, max_n_atoms), np.nan, np.float32)
    features_out = np.full((n_molecules, 16), np.nan, np.float32)
    
    for i in range(n_molecules):
        n_atoms_i = n_atoms[i]
        coords_out[i, :n_atoms_i] = coords[i]
        charges_out[i, :n_atoms_i] = charges[i]
        features_out[i] = features[i]
        
    n_atoms_out = np.array(n_atoms)
    
    # print(f"{n_atoms_out.shape}\n{coords_out.shape}\n{charges_out.shape}\n{energies_out.shape}")

    out_dd = {'coords': coords_out,
             'charges': charges_out, 
             'n_atoms': n_atoms_out,
             'features': features_out}
    
    io.savemat(out_fp, out_dd)

CONST_ANGSTROM_TO_BOHR_RADIUS = 1.88973

INTERNAL_ENERGIES_0K = {
    ### THIS IS FOR INTERNAL ENERGIES AT 0K
    1: -0.500273, # Hydrogen
    6: -37.846772, # Carbon
    7: -54.583861, # Nitrogen
    8: -75.064579, # Oxygen
    9: -75.064579 # Flourine
    
}

class QM9Data:
    def __init__(self) -> None:

        
        self.charges = None
        self.coords = None
        self.n_atoms = None
        self.features = None
        self.n_samples = 0
        
    def extend_dataset(self, fp: str) -> None:
        data_dd = io.loadmat(fp)
        # energies = data_dd['energies'].flatten()
        n_atoms = data_dd['n_atoms'].flatten()
        n_new_points = n_atoms.shape[0]
        
        if self.n_samples:
            self.charges = np.concatenate((self.charges, data_dd['charges']))
            self.coords = np.concatenate((self.coords, data_dd['coords']))
            self.n_atoms = np.concatenate((self.n_atoms, n_atoms))
            self.features = np.concatenate((self.features, data_dd['features']))
        else:
            self.charges = data_dd['charges']
            self.coords = data_dd['coords']
            self.features = data_dd['features']
            self.n_atoms = n_atoms
    
        self.n_samples += n_new_points
        for x in [self.charges, self.coords, self.features, self.n_atoms]:
            assert x.shape[0] == self.n_samples

    def get_dataset(self) -> FCHLEncoding:
        
        thermo_energies = self.get_thermochemical_energies().flatten()

        # THE INTERNAL ENERGIES CORRESPONDS TO THE U0 FEATURE IN THE QM9 DATASET.
        # In the QM9 dataset, it is listed as the 13th feature. In the parsing step, 
        # I am removing the first feature because it's the string 'gdb9' tag, and then
        # we are using 0-based indexing.
        internal_energies = self.features[:, 11].flatten() 
        return FCHLEncoding(self.coords * CONST_ANGSTROM_TO_BOHR_RADIUS, 
                                self.charges, 
                                thermochemical_energies=thermo_energies, 
                                internal_energies=internal_energies)

    def get_thermochemical_energies(self) -> np.ndarray:
        thermochemical_energies = np.zeros_like(self.features[:, 11])
        
        for i in range(self.features.shape[0]):
            charges = self.charges[i, :self.n_atoms[i]]
            
            for charge in charges:
                thermochemical_energies[i] += INTERNAL_ENERGIES_0K[charge]
                
        return thermochemical_energies


    def get_QM7_HCON_molecules(self) -> None:
        logging.warn("Truncating QM9 dataset to only include 7 heavy atoms and only elements [H, C, O, N]")
        keep_bool_arr = np.empty((self.charges.shape[0]), dtype=bool)
        for i in range(self.charges.shape[0]):
            charges_i = self.charges[i]
            contains_F = np.any(charges_i == 9)
            if contains_F:
                keep_bool_arr[i] = False
            else:
                n_H_atoms = np.sum(charges_i == 1)
                n_heavy_atoms = self.n_atoms[i] - n_H_atoms
                if n_heavy_atoms <= 7:
                    keep_bool_arr[i] = True

                else:
                    keep_bool_arr[i] = False
        
        self.charges = self.charges[keep_bool_arr]
        self.coords = self.coords[keep_bool_arr]
        self.n_atoms = self.n_atoms[keep_bool_arr]
        self.features = self.features[keep_bool_arr]
        self.n_samples = self.charges.shape[0]

