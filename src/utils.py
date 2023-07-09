import os
import logging

def write_result_to_file(fp: str, missing_str: str='', **trial) -> None:
    """Write a line to a tab-separated file saving the results of a single
        trial.
    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None
    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        header_line = "\t".join(header_lst) + "\n"
        with open(fp, 'w') as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, 'a') as f:
        f.write(trial_line)


def plot_dir_setup(dir: str) -> None:
    """Creates a new directory if it doesn't already exist

    Args:
        dir (str): folder to be set up
    """
    if not os.path.isdir(dir):
        logging.info("Creating plotting directory at %s", dir)
        os.mkdir(dir)

    else:
        logging.info("Plotting in directory %s", dir)