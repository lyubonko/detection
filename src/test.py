import os

import torch
import torch.utils.data

from core.engine import evaluate

from core import utils

from datasets.dataset_factory import prepare_data
from models.model_factory import prepare_model
from utils.parse_args import parse_arguments


def main(args, metric_logger):
    device = torch.device(args.device)

    metric_logger.print_out("Loading data")
    data_loader, data_loader_test, train_sampler, num_classes = prepare_data(args)

    metric_logger.print_out("Creating model")
    model, model_without_ddp = prepare_model(num_classes, device, args.pretrained_path,
                                             args.MODEL.name, args.distributed, args.gpu)

    if args.EVAL.save_results:
        save_eval_file = os.path.join(os.path.dirname(args.pretrained_path),
                                      os.path.basename(args.pretrained_path)[:-4] + '_results.json')
    else:
        save_eval_file = None

    evaluate(model, data_loader_test, device, metric_logger, args.LOG.print_freq_test, file_save=save_eval_file)
    return


if __name__ == "__main__":
    # parse argument string
    params = parse_arguments()

    # handle distributive training
    utils.init_distributed_mode(params)

    # Set up Logger
    if params.LOG.make_log:
        log_file = os.path.join(os.path.dirname(params.pretrained_path),
                                os.path.basename(params.pretrained_path)[:-4] + '_eval.txt')
    else:
        log_file = None

    logger = utils.MetricLogger(log_file=log_file,
                                log_file_epoch=None,
                                log_file_epoch_test=None,
                                log_file_iter=None)

    main(params, logger)
