import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN


def fasterrcnn_resnet_fpn(pretrained_path=None,
                          backbone='resnet50',
                          num_classes=91,
                          pretrained_backbone=True,
                          mask=False,
                          **kwargs):
    """
    Based on torchvision.models.detection.faster_rcnn
    """
    if pretrained_path is not None:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone, pretrained_backbone)

    if mask:
        model = MaskRCNN(backbone, num_classes, **kwargs)
    else:
        model = FasterRCNN(backbone, num_classes, **kwargs)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

    return model
