import math
import sys
import time
import torch
import json as json

import torchvision.models.detection.mask_rcnn

from core.coco_utils import get_coco_api_from_dataset
from core.coco_eval import CocoEvaluator
from core import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, metric_logger, print_freq):
    model.train()

    metric_logger.renew(epoch_size=len(data_loader), delimiter="  ", epoch=epoch, train=True)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            metric_logger.print_out("Loss is {}, stopping training".format(loss_value))
            metric_logger.print_out(str(loss_dict_reduced))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.end_epoch()


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, metric_logger, print_freq, file_save=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    metric_logger.renew(epoch_size=len(data_loader), delimiter="  ", train=False)
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    all_results = []
    for image, targets in metric_logger.log_every(data_loader, print_freq, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # TODO: add also mask to results
        if file_save is not None:
            for image_id in res.keys():
                for b in range(len(res[image_id]['labels'])):
                    # boxes xyxy -> xywh
                    current_box = res[image_id]['boxes'][b].numpy().tolist()
                    current_box[2] -= current_box[0]
                    current_box[3] -= current_box[1]
                    all_results.append({"image_id": int(image_id),
                                        "category_id": int(res[image_id]['labels'][b].numpy()),
                                        "bbox": current_box,
                                        "score": float(res[image_id]['scores'][b].numpy())})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate(make_print=True)
    coco_evaluator.summarize(make_print=True)

    for k in coco_evaluator.coco_eval.keys():
        acc_name = k + "_" + "mAP"
        # here I add only the main figure of merit, could be extended
        acc_val = coco_evaluator.coco_eval[k].stats[0]

        metric_logger.add_meter(acc_name, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.meters[acc_name].update(acc_val)

    metric_logger.end_epoch()
    metric_logger.print_out("Averaged stats: {}".format(str(metric_logger)))

    # save results
    if file_save is not None:
        with open(file_save, 'w') as outfile:
            json.dump(all_results, outfile)

    torch.set_num_threads(n_threads)
    return coco_evaluator
