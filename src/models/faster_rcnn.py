import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def fasterrcnn_resnet_fpn(pretrained_path=None,
                          backbone='resnet50',
                          num_classes=91,
                          pretrained_backbone=True,
                          mask=False,
                          hidden_layer=256,
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

    # === handle non-standard case (different number of classes)
    if num_classes != 91:
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # === Mask-RCNN or not
    if mask:
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

    return model
