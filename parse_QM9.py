import logging
import os
import numpy as np
import argparse 

from src.data.DataSets import parse_chunk, QM9Data

BAD_INDICES = [21725, 87037, 59827, 117523, 128113, 129053, 129152, 129158, 
                130535, 6620, 59818, 21725, 59827, 128113, 129053, 129152, 
                130535, 6620, 59818]

def main(args: argparse.Namespace) -> None:

    file_lst = os.listdir(args.in_dir)
    n_files = len(file_lst)

    logging.info("Parsing %i files", n_files)

    shuffle_pre = np.random.permutation(n_files) + 1
    shuffle = [i for i in shuffle_pre if i not in BAD_INDICES]
    shuffle = np.array(shuffle)
    # shuffle = np.arange(n_files) + 1

    ##########################################################################
    # REMOVE BAD MOLECULES FROM THE LIST OF FILES


    fp_lst = np.array([os.path.join(args.in_dir, f"dsgdb9nsd_{x:06d}.xyz") for x in shuffle])

    n_folds = n_files // args.n_molecules_per_file + 1
    

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    for i in range(n_folds):

        fp_lst_partial = fp_lst[i * args.n_molecules_per_file:(i + 1) * args.n_molecules_per_file]

        out_fp = os.path.join(args.out_dir, f"qm9_parsed_{i}.mat")

        parse_chunk(fp_lst_partial, out_fp)
        logging.info("Wrote to file: %s", out_fp)


    q_obj = QM9Data()
    for fp in os.listdir(args.out_dir):
        q_obj.extend_dataset(os.path.join(args.out_dir, fp))

    logging.info("Final q_obj shapes: charges: %s, coords: %s, n_atoms: %s, features: %s",
                    q_obj.charges.shape,
                    q_obj.coords.shape,
                    q_obj.n_atoms.shape,
                    q_obj.features.shape)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-in_dir', default='data/qm9/molecules')
    parser.add_argument('-out_dir', default='data/qm9/parsed')
    parser.add_argument('-n_molecules_per_file', type=int, default=10_000)

    a = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(a)