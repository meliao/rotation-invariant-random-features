import argparse
import logging
from typing import Callable, Dict
from timeit import default_timer
import numpy as np
import os

import torch
from torch import nn


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



def loss_mae(preds: np.ndarray, actual: np.ndarray) -> np.float32:
    return np.mean(np.abs(preds.flatten() - actual.flatten()))

CONST_KCALPERMOL_TO_EV = 0.043

KEYS = ['train_features', 'train_labels', 'val_features', 'val_labels', 'test_features', 'test_labels']


def load_data(save_data_dir: str) -> Dict:
    """Loads random features that were serialized.

    Args:
        save_data_dir (str): Directory to search in

    Returns:
        Dict: keys are strings and values are numpy arrays
    """
    return {x: np.load(os.path.join(save_data_dir, x + '.npy')) for x in KEYS}

def build_features(args: argparse.Namespace) -> Dict:
    ##########################################################################
    ### LOAD DATA

    if args.encoding == 'FCHL':
        logging.info("Using FCHL encoding")
        if args.radial_params is not None:
            r = np.array(args.radial_params)
        else:
            r = None
        train_dset, validation_dset, test_dset = load_QM7(args.data_fp, 
                                                            args.n_train, 
                                                            args.n_test, 
                                                            args.validation_set_fraction,
                                                            radial_params=r)
    else:
        raise ValueError(f"Unrecognized encoding: {args.encoding}")



    logging.info("Loaded QM7 data. %i train samples, %i val samples, %i test samples", 
                    len(train_dset), len(validation_dset), len(test_dset))

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



    logging.info("Working on n_features=%i", args.n_features)



    ##########################################################################
    ### GENERATE DESIGN MATRIX

    random_weights = np.random.normal(scale=args.weight_variance,
                                        size=train_dset.expected_weights_shape(args.n_features, args.max_L))
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

    out_dd = {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels
    }
    return out_dd

def serialize_data(data_dd: Dict, save_data_dir: str) -> None:
    """_summary_

    Args:
        data_dd (Dict): Contains keys KEYS
        save_data_dir (str): Directory where data should be saved.
    """
    for k in KEYS:
        np.save(os.path.join(save_data_dir, k + '.npy'), data_dd[k])

class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int, width: int=1, depth: int=1) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.width = width
        self.depth = depth

        self.linear_lst = nn.ModuleList([nn.Linear(input_dim, width)])

        for i in range(depth - 1):
            self.linear_lst.append(nn.Linear(width, width))

        if depth == 0:
            self.linear_lst = []
            self.final_lin_layer = nn.Linear(input_dim, 1)
        else:
            self.final_lin_layer = nn.Linear(width, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        for f in self.linear_lst:
            x_in = f(x_in)
            x_in = self.relu(x_in)

        x_out = self.final_lin_layer(x_in)
        return x_out



def train(model: nn.Module,
            n_epochs: int,
            lr_init: float,
            weight_decay: float,
            momentum: float,
            eta_min: float,
            train_loader: torch.utils.data.DataLoader,
            device: torch.cuda.Device,
            n_epochs_per_log: int,
            log_function: Callable=None) -> None:
    """_summary_

    Args:
        model (nn.Module): _description_
        n_epochs (int): _description_
        lr_init (float): _description_
        train_loader (torch.utils.data.DataLoader): _description_
        val_loader (torch.utils.data.DataLoader): _description_
    """
    optimizer = torch.optim.Adam(model.parameters(), lr_init,
                                weight_decay=weight_decay,
                                ) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)
    logging.info("Beginning model training for %i epochs", n_epochs)
    t1 = default_timer()
    model = model.to(device)
    for epoch in range(n_epochs):
        
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
        
            # loss_mae = torch.mean(torch.abs(y.flatten() - pred.flatten()))
            # loss_mae.backward()
            loss_mse = torch.mean(torch.square(y.flatten() - pred.flatten()))
            loss_mse.backward()
            optimizer.step()
            
        scheduler.step()
        
        if epoch % n_epochs_per_log == 0:

            if log_function is not None:
                log_function(model, epoch)
            else:
                logging.info("Epoch %i / %i. Loss value = %f", epoch, n_epochs, loss_mse.detach().cpu().item())
    t2 = default_timer()
    logging.info("Optimization is complete in %f seconds", t2 - t1)
    return model.to('cpu')



class LinearData(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y
        
        self.n_rows, self.n_cols = X.shape
        
    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Generate design matrix
    3. Fit regression weights
    4. Predict on test data
    5. Record test results
    """

    if not os.path.isdir(args.save_data_dir):
        os.mkdir(args.save_data_dir)

    ##########################################################################
    ### LOAD SERIALIZED FEATURES IF POSSIBLE
    if np.all([os.path.isfile(os.path.join(args.save_data_dir, x + '.npy')) for x in KEYS]):
        logging.info("Loading pre-serialized features from %s", args.save_data_dir)

        data_dd = load_data(args.save_data_dir)

    else:
        logging.info("Generating random features")
        data_dd = build_features(args)

        logging.info("Serializing data")
        serialize_data(data_dd, args.save_data_dir)

    ##########################################################################
    ### LOG DATASET SIZES
    for k in ['train', 'val', 'test']:
        k_features = data_dd[k + '_features']
        k_labels = data_dd[k + '_labels']
        logging.info("%s features shape: %s, %s labels shape: %s", k, k_features.shape, k, k_labels.shape)

    
    #########################################################################
    ### CREATE DATASETS AND DATALOADERS

    train_dataset = LinearData(torch.from_numpy(data_dd['train_features']), 
                                                    torch.from_numpy(data_dd['train_labels']))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataset = LinearData(torch.from_numpy(data_dd['val_features']), 
                                                torch.from_numpy(data_dd['val_labels']))
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataset = LinearData(torch.from_numpy(data_dd['test_features']), 
                                                torch.from_numpy(data_dd['test_labels']))
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    torch.manual_seed(args.seed)

    model = ShallowMLP(data_dd['train_features'].shape[1], width=args.width, depth=args.depth)
    logging.info("Initialized model with width %i and depth %i", model.width, model.depth)

    # Initialize the final bias with the mean of the training labels
    # torch.nn.init.constant_(model.linear_1.bias, torch.mean(train_dataset.y))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Training on device: %s", device)


    def log_function(model, epoch):
        model = model.to('cpu')
        with torch.no_grad():
            # 1. Evaluate on train set

            train_preds = model(train_dataset.X)
            train_y = train_dataset.y

            train_mse = torch.mean(torch.square(train_preds.flatten() - train_y.flatten()))
            train_mae = torch.mean(torch.abs(train_preds.flatten() - train_y.flatten()))

            val_preds = model(val_dataset.X)
            val_y = val_dataset.y

            val_mse = torch.mean(torch.square(val_preds.flatten() - val_y.flatten()))
            val_mae = torch.mean(torch.abs(val_preds.flatten() - val_y.flatten()))

            weight_norm = torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 2)
            logging.info("Epoch %i/%i. Train MSE: %f, Train MAE: %f", epoch, args.n_epochs, train_mse.item(), train_mae.item())
            logging.info("\t Val MSE: %f, Val MAE: %f", val_mse.item(), val_mae.item())
            logging.info("\t Weight L2 norm: %f", weight_norm.item())

            train_dd = {
                # Optimization info
                'epoch': epoch,
                'train_mse': train_mse.item(),
                'train_mae': train_mae.item(),
                'val_mse': val_mse.item(),
                'val_mae': val_mae.item(),
                'weight_norm': weight_norm.item(),

                # Experiment info
                'n_train': len(train_dataset),
                'n_val': len(val_dataset),
                'width': args.width,
                'depth': args.depth,
                'lr_init': args.lr_init,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'eta_min': args.eta_min
            }

            write_result_to_file(args.train_results_fp, **train_dd)

        model = model.to(device)
        


    model = train(model=model,
                            n_epochs=args.n_epochs,
                            lr_init=args.lr_init,
                            weight_decay=args.weight_decay,
                            momentum=0.,
                            eta_min=args.eta_min,
                            train_loader=train_dataloader,
                            device=device,
                            n_epochs_per_log=5,
                            log_function=log_function)


    # train_preds = model(train_dataset.X).flatten().detach().numpy()
    # train_preds_fp = os.path.join(args.save_data_dir, 'train_preds.npy')
    # np.save(train_preds_fp, train_preds)


    if args.ridge_check:

        ridge_vals = [10., 1., 0.1, 0.01, 0.001, 0.]
        weights_dd, _ = get_regression_weights(data_dd['train_features'], data_dd['train_labels'], ridge_vals)

        for l2_reg, weights in weights_dd.items():
            train_preds = np.matmul(data_dd['train_features'], weights).flatten()
            train_mae = loss_mae(train_preds, data_dd['train_labels'].flatten())


            val_preds = np.matmul(data_dd['val_features'], weights).flatten()
            val_mae = loss_mae(val_preds, data_dd['val_labels'].flatten())
            logging.info("Ridge Regression Check: l2_reg: %f, train MAE (eV): %f, val MAE (eV): %f", 
                                                                                    l2_reg,
                                                                                    train_mae,
                                                                                    val_mae)

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
    parser.add_argument('-n_features', type=int)
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
    parser.add_argument('-radial_params', type=float, nargs='+', default=None)
    parser.add_argument('-save_data_dir')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-width', type=int, default=2)
    parser.add_argument('-depth', type=int, default=2)
    parser.add_argument('-n_epochs', type=int, default=10_000)
    parser.add_argument('-lr_init', type=float, default=1e-03)
    parser.add_argument('-weight_decay', type=float, default=1e-03)
    parser.add_argument('-train_results_fp', type=str)
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-eta_min', type=float, default=1e-06)
    parser.add_argument('-ridge_check', action='store_true')

    a = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format=FMT,
                        datefmt=TIMEFMT)
    main(a)


