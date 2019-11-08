import torch

from models.faster_rcnn import fasterrcnn_resnet_fpn
from models.centernet.ctdet import CenterNetDetect


def get_model_faster(name, num_classes, pretrained_path=None):

    with_mask = name.find('mask') != -1
    resnet_backbone = name[name.find('resnet'):name.find('resnet') + 9].strip('_')  # 'resnet18', 'resnet34', 'resnet50'

    model = fasterrcnn_resnet_fpn(num_classes=num_classes,
                                  pretrained_path=pretrained_path,
                                  mask=with_mask,
                                  backbone=resnet_backbone)

    return model


def get_model_center(model_params, loss_params=None, pretrained_path=None):

    heads = {'hm': model_params.heads_hm, 'wh': model_params.heads_wh, 'reg': model_params.heads_reg}
    model = CenterNetDetect(model_params.num_classes, model_params.arch, heads, model_params.head_conv,
                            max_per_image=model_params.max_per_image, loss_params=loss_params)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

    return model


def prepare_model(num_classes, device, pretrained_path, model_params, distributed, ids_gpu, loss_params=None):

    pretrained_path = pretrained_path if len(pretrained_path) > 0 else None

    if model_params.name == 'center':
        model = get_model_center(model_params, loss_params, pretrained_path=pretrained_path)
    else:
        model = get_model_faster(model_params.name, num_classes, pretrained_path=pretrained_path)
    model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ids_gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    return model, model_without_ddp
