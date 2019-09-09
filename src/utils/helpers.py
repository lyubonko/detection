import os
import numpy as np
from types import SimpleNamespace

import core.utils as utils


def next_expr_name(path_dir, n_digit_length, id_only=False):

    myhost = os.uname()[1]
    myid = os.getlogin()

    if id_only:
        start_pattern = myid.lower() + '_' + "e"
    else:
        start_pattern = myid.lower() + '_' + myhost.lower() + '_' + "e"

    len_start = len(start_pattern)
    len_end = len_start + n_digit_length
    present_numbers = [int(x[len_start:len_end]) for x in os.listdir(path_dir) if x.startswith(start_pattern)]
    next_number = np.max([0] + present_numbers) + 1
    return start_pattern + str(next_number).zfill(n_digit_length)


def dict2str(d, start_n=0):
    """
    Convert dict or SimpleNamespace to string.
    Primary used to print settings (from param file).

    :param d: dict or SimpleNamespace to print
    :param start_n: number of white spaces before output
    :return: string with dict
    """
    res = ""
    prefix_val = " " * start_n
    if isinstance(d, SimpleNamespace):
        d = d.__dict__
    sorted_keys = sorted(d.keys())
    for k in sorted_keys:
        if isinstance(d[k], dict) or isinstance(d[k], SimpleNamespace):
            res += prefix_val + str(k) + ": " + "\n" + dict2str(d[k], start_n + 2)
        else:
            res += prefix_val + str(k) + ": " + str(d[k]) + "\n"
    return res


def init_experiment_settings(params):

    if utils.is_main_process():
        # new experiment name
        if len(params.experiment_name) == 0:
            params.experiment_name = next_expr_name(params.output_dir, 4)

        # create folder for this experiment
        params.output_dir = os.path.join(params.output_dir, params.experiment_name)
        os.mkdir(params.output_dir)

        # copy of param file
        save_param_filename = os.path.join(params.output_dir, params.experiment_name + "_params.txt")
        print(dict2str(params), file=open(save_param_filename, 'w'))
