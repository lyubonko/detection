import torch

from models.faster_rcnn import fasterrcnn_resnet_fpn
from models.model_penn import fasterrcnn_resnet50_fpn_2c


def get_model(name, num_classes, pretrained_path=None):

    with_mask = name.find('mask') != -1
    resnet_backbone = name[name.find('resnet'):name.find('resnet') + 9].strip('_')  # 'resnet18', 'resnet34', 'resnet50'

    if num_classes == 91:  # COCO
        model = fasterrcnn_resnet_fpn(num_classes=num_classes,
                                      pretrained_path=pretrained_path, mask=with_mask,
                                      backbone=resnet_backbone)
    else:
        # TODO: make it nicer
        model = fasterrcnn_resnet50_fpn_2c(num_classes, pretrained=True)

    return model


def prepare_model(num_classes, device, pretrained_path, model_name, distributed, ids_gpu):

    pretrained_path = pretrained_path if len(pretrained_path) > 0 else None
    model = get_model(model_name, num_classes, pretrained_path=pretrained_path)
    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ids_gpu])
        model_without_ddp = model.module

    return model, model_without_ddp
