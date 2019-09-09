from argparse import ArgumentParser
from types import SimpleNamespace
import os
import yaml


def convert_dict_namespace(dict_convert):
    """
    Convert params from dictionary to SimpleNamespace type (in order to use dotted notation)
    :param dict_convert: dictionary to convert
    :return: SimpleNamespace
    """
    for key in dict_convert:
        if isinstance(dict_convert[key], dict):
            dict_convert[key] = convert_dict_namespace(dict_convert[key])
    return SimpleNamespace(**dict_convert)


def parse_arguments():
    """
    The function reads files with settings ('param file').
    The param files are stored in PROJECT_ROOT/experiments/cfgs
    The param file provides default values for all parameters.
    A common line arguments are used to set specific values, different from default.

    :return: paraters in SimpleNamespace format
    """

    parser = ArgumentParser(description=__doc__)  # TODO: check what is mean

    parser.add_argument('--param_file', type=str, help='configure file with parameters')

    parser.add_argument('--device', default=None, type=str, help='device')
    parser.add_argument('--workers', default=None, type=int, help='number of data loading workers')
    parser.add_argument('--output_dir', default=None, type=str, help='path where to save')

    parser.add_argument('--test_only', default=None, type=bool, help="Only test the model")
    parser.add_argument('--pretrained_path', default=None, type=str, help="Use pre-trained models from the modelzoo")

    parser.add_argument('--world_size', default=None, type=int,  help='number of distributed processes')
    parser.add_argument('--dist_url', default=None, type=str, help='url used to set up distributed training')

    parser.add_argument('--DATASET_name', default=None, type=str, help='dataset')
    parser.add_argument('--DATASET_path', default=None, type=str, help='path to dataset')
    parser.add_argument('--DATASET_batch_size', default=None, type=int, help='train batch size')
    parser.add_argument('--DATASET_aspect_ratio_group_factor', default=None, type=int)

    parser.add_argument('--TRAIN_epochs', default=None, type=int, help='number of total epochs to run')
    parser.add_argument('--TRAIN_lr', default=None, type=float, help='initial learning rate')
    parser.add_argument('--TRAIN_momentum', default=None, type=float, help='momentum')
    parser.add_argument('--TRAIN_weight_decay', default=None, type=float, help='weight decay')
    parser.add_argument('--TRAIN_lr_step_size', default=None, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--TRAIN_lr_steps', default=None, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--TRAIN_lr_gamma', default=None, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--TRAIN_resume', default=None, type=str, help='resume from checkpoint')

    parser.add_argument('--EVAL_eval_metric', default=None, type=str, help='figure of merit to select best model')
    parser.add_argument('--EVAL_save_results', default=None, type=bool, help='save or not results')
    parser.add_argument('--EVAL_make_log', default=None, type=bool, help='make log durint eval')

    parser.add_argument('--MODEL_name', default=None, type=str, help='model name')

    parser.add_argument('--LOG_print_freq', default=None, type=int, help='print frequency')
    parser.add_argument('--LOG_print_freq_test', default=None, type=int, help='print frequency test')
    parser.add_argument('--LOG_with_tensorboard', default=None, type=bool, help='tensorboard on')
    parser.add_argument('--LOG_smooth_window', default=None, type=int, help='smooth window for meters')

    args = parser.parse_args()

    # parse param file
    with open(args.param_file, 'r') as f:
        params = yaml.safe_load(f)

    # add host & id to params
    params['host'] = os.uname()[1]
    params['user_id'] = os.getlogin()

    # addition configurations (check if something was set using common line)
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            # check capital letter (this indicates that parameter is folded)
            # note: this code only handles the 'two-foldness' of parameters, like param[TRAIN][lr]
            if k[0].isupper():
                part1, part2 = k.split('_', 1)  # split at first occurrence of '_'
                params[part1][part2] = args.__dict__[k]
            else:
                params[k] = args.__dict__[k]

    # convert params from dictionary to SimpleNamespace type
    params = convert_dict_namespace(params)

    return params
