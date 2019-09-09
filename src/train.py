r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import datetime
import os
import sys
import time

import torch
import torch.utils.data

from core.engine import train_one_epoch, evaluate
from core import utils
from core.optimizer_scheduler import prepare_optimizer
from datasets.dataset_factory import prepare_data
from models.model_factory import prepare_model
from utils.parse_args import parse_arguments
from utils.helpers import init_experiment_settings


def main(args, metric_logger):
    device = torch.device(args.device)

    metric_logger.print_out("Loading data")
    data_loader, data_loader_test, train_sampler, num_classes = prepare_data(args)

    metric_logger.print_out("Creating model")
    model, model_without_ddp = prepare_model(num_classes, device, args.pretrained_path,
                                             args.MODEL.name, args.distributed, args.gpu)

    metric_logger.print_out("Setting optimizer & scheduler")
    optimizer, lr_scheduler = prepare_optimizer(args, model)

    # how we select best model, the value suppose to be - the bigger the better
    figure_of_merit_best = 0

    if args.TRAIN.resume:
        metric_logger.print_out("Training is resumed from: {}".format(args.TRAIN.resume))
        checkpoint = torch.load(args.TRAIN.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        figure_of_merit_best = checkpoint['accuracy']

    metric_logger.print_out("Start training")
    start_time = time.time()
    for epoch in range(lr_scheduler.last_epoch, args.TRAIN.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, metric_logger, args.LOG.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'accuracy': figure_of_merit_best
            },
                os.path.join(args.output_dir, args.experiment_name + "_checkpoint_last.pth"))
            metric_logger.print_out("Last checkpoint saved [epoch: {}]".format(epoch))

        # evaluate after every epoch
        coco_evaluator = evaluate(model, data_loader_test, device, metric_logger, args.LOG.print_freq_test)

        # check for best model
        figure_of_merit_current = coco_evaluator.coco_eval[args.EVAL.eval_metric].stats[0]
        if args.output_dir and figure_of_merit_current >= figure_of_merit_best:
            # update the figure of merit
            figure_of_merit_best = figure_of_merit_current
            # save checkpoint
            utils.save_on_master(model_without_ddp.state_dict(),
                                 os.path.join(args.output_dir, args.experiment_name + "_model_best.pth"))
            metric_logger.print_out("Best model saved [epoch: {}]".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    metric_logger.print_out('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    # parse argument string
    params = parse_arguments()

    # handle distributive training
    utils.init_distributed_mode(params)

    # experiment name & path to save output
    init_experiment_settings(params)

    # Set up Logger
    log_file = os.path.join(params.output_dir, params.experiment_name + "_log.txt")
    log_file_iter = os.path.join(params.output_dir, params.experiment_name + "_log_iter.txt")
    log_file_epoch = os.path.join(params.output_dir, params.experiment_name + "_log_epoch.txt")
    log_file_epoch_test = os.path.join(params.output_dir, params.experiment_name + "_log_epoch_test.txt")
    log_tb_file = params.output_dir if params.LOG.with_tensorboard else None
    logger = utils.MetricLogger(smooth_window=params.LOG.smooth_window,
                                log_file=log_file,
                                log_file_epoch=log_file_epoch,
                                log_file_epoch_test=log_file_epoch_test,
                                log_file_iter=log_file_iter,
                                log_tb_file=log_tb_file
                                )

    main(params, logger)
