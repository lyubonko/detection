import math

import numpy as np
import torch
from torch import nn

from models.centernet.data_utils import _sigmoid, gen_oracle_map, draw_umich_gaussian, gaussian_radius
from models.centernet.transform_frcnn import GeneralizedRCNNTransform

from models.centernet.decode import ctdet_decode
from models.centernet.losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.centernet.model import create_model


class CenterNetDetect(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, num_classes, arch, heads, head_conv,
                 max_per_image=100,
                 device=torch.device('cuda'),
                 reg_offset=True, down_ratio=4,
                 # transform parameters
                 min_size=512, max_size=512,   # original (Faster RCNN): min_size=800, max_size=1333
                 image_mean=(0.485, 0.456, 0.406),
                 image_std=(0.229, 0.224, 0.225),
                 # output
                 output_h=128, output_w=128,
                 # loss parameters
                 loss_params=None
                 ):
        super(CenterNetDetect, self).__init__()

        self.model = create_model(arch, heads, head_conv)
        self.max_per_image = max_per_image
        self.num_classes = num_classes

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.reg_offset = reg_offset
        self.down_ratio = down_ratio

        if loss_params is not None:
            self.loss = CenterNetLoss(opt_wh_weight=loss_params.wh_weight,
                                      opt_off_weight=loss_params.off_weight,
                                      opt_hm_weight=loss_params.hm_weight)
        else:
            self.loss = CenterNetLoss()  # opt_eval_oracle_hm=True, opt_eval_oracle_wh=True, opt_reg_offset=True

        self.data = CenterNetData()

        self.output_h = output_h
        self.output_w = output_w

        self.device = device

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # (FRCNN part) prepare the image for the net
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # (Center-Net) process image
        outputs = self.model(images.tensors)[-1]

        # (FRCNN part)
        if self.training:
            add_target = self.data.additional_gt_batch(targets, self.output_h, self.output_w,
                                                       downratio=self.down_ratio, reg_offset=True)
            add_target_ = [{k: add_target[k].to(self.device) for k in add_target.keys()}]

            losses = self.loss([outputs], add_target_[-1])
            return losses

        hm = outputs['hm'].sigmoid_()
        wh = outputs['wh']
        reg = outputs['reg'] if self.reg_offset else None

        dets = ctdet_decode(hm, wh, reg=reg, K=self.max_per_image)
        # to make compatible to Faster RCNN
        n_images = dets.shape[0]
        detections = [None] * n_images
        for i in range(n_images):
            detections[i] = {}
            detections[i]['boxes'] = dets[i, :, :4] * self.down_ratio  # TODO: better to check this factor
            detections[i]['scores'] = dets[i, :, 4]
            detections[i]['labels'] = dets[i, :, 5]

        # (FRCNN part)
        # this part brings back detection back to original image sizes
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        for d in detections:
            for i in range(d['labels'].size(0)):
                d['labels'][i] = self.data.valid_ids[int(d['labels'][i])]

        return detections


class CenterNetLoss(torch.nn.Module):
    """
    The loss calculation part.
    """

    # Default values are from my experiments with 'DFN resnet 18'

    def __init__(self, opt_mse_loss=False, opt_reg_loss='l1', opt_dense_wh=False,
                 opt_norm_wh=False, opt_cat_spec_wh=False, opt_reg_offset=True,
                 opt_num_stacks=1, opt_device='cuda',
                 # loss weights
                 opt_wh_weight=0.1, opt_off_weight=0.01, opt_hm_weight=1):
        super(CenterNetLoss, self).__init__()

        # (probably) used for guided supervision (hourglass or something similar)
        self.num_stacks = opt_num_stacks

        self.mse_loss = opt_mse_loss
        self.device = opt_device
        self.dense_wh = opt_dense_wh
        self.cat_spec_wh = opt_cat_spec_wh
        self.reg_offset = opt_reg_offset

        self.wh_weight = opt_wh_weight
        self.off_weight = opt_off_weight
        self.hm_weight = opt_hm_weight

        self.crit = torch.nn.MSELoss() if opt_mse_loss else FocalLoss()

        if opt_reg_loss == 'l1':
            self.crit_reg = RegL1Loss()
        elif opt_reg_loss == 'sl1':
            self.crit_reg = RegLoss()
        else:
            self.crit_reg = None

        if opt_dense_wh:
            self.crit_wh = torch.nn.L1Loss(reduction='sum')
        else:
            if opt_norm_wh:
                self.crit_wh = NormRegL1Loss()
            else:
                if opt_cat_spec_wh:
                    self.crit_wh = RegWeightedL1Loss()
                else:
                    self.crit_wh = self.crit_reg

    def forward(self, outputs, batch):
        """
        :param outputs: the net output, which contains the following something like this
            hm torch.Size([1, 80, 128, 128])
            wh torch.Size([1, 2, 128, 128])
            reg torch.Size([1, 2, 128, 128])
        :param batch: what has come from sample, contains
            'hm', 'wh', 'ind', 'reg'
            (optionally) 'dense_wh_mask', 'cat_spec_mask', 'cat_spec_wh'
        :return: losses related to the CenterNet
        """

        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.num_stacks):
            output = outputs[s]
            if not self.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            # === calculation of the the hm-loss
            hm_loss += self.crit(output['hm'], batch['hm']) / self.num_stacks

            # === calculation of the offset-loss
            if self.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / self.num_stacks

            # === calculation of the offset-loss
            if self.reg_offset and self.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / self.num_stacks

        loss_stats = {'hm_loss': self.hm_weight * hm_loss,
                      'wh_loss': self.wh_weight * wh_loss}

        if self.reg_offset and self.off_weight > 0:
            loss_stats.update({'off_loss': self.off_weight * off_loss})

        return loss_stats


class CenterNetData(object):

    def __init__(self):
        self.num_classes = 80  # from COCO
        self.max_objs = 128  # from COCO
        self.valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

    def additional_gt(self, bboxes, labels, output_h, output_w, downratio=4, reg_offset=True):

        num_objs = len(labels)

        # === [C, W/R, H/R] to store keypoints outputs, shorthand for 'heatmap'
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)

        # === ??? it seems stems from COCO: max_objs = 128, but what is it?
        # => my guess is that this is maximum number of objects in image at this dataset, 128 boxes in image is max.
        # => I can check this

        # === store 'SIZES' of box (width and height), for each box
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)

        # === store 'OFFSETS' for each box
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)

        # === placeholder for indexes of a box center calculated from left-top of the box
        ind = np.zeros((self.max_objs), dtype=np.int64)

        # === some 'regression mask' ?, but for what not clear
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # === how to draw gaussians
        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            # === get annotation
            bbox = bboxes[k] / downratio
            cls_id = int(self.cat_ids[int(labels[k])])

            # === sizes of box
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # === seems we get 'size adaptive standard deviation'
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                # === center of the box
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                # === center of the box as 'int'
                ct_int = ct.astype(np.int32)
                # === draw gaussian in 'hm'
                draw_gaussian(hm[cls_id], ct_int, radius)

                # === store box width and height in 'wh', weird '1. * '
                wh[k] = 1. * w, 1. * h

                # === somehow index of box center calculated from left-top of the box
                ind[k] = ct_int[1] * output_w + ct_int[0]

                # === this what will be used for offset calculations
                reg[k] = ct - ct_int

                # === not yet clear - what is it
                reg_mask[k] = 1

        ret = {'hm': torch.from_numpy(hm), 'reg_mask': torch.from_numpy(reg_mask),
               'ind': torch.from_numpy(ind), 'wh': torch.from_numpy(wh)}
        # === this seems to be special for 'detection' problem (not clear but it is False by default)
        if reg_offset:
            ret.update({'reg': torch.from_numpy(reg)})

        return ret

    def additional_gt_batch(self, targets, output_h, output_w, reg_offset=True, downratio=4):

        batch_size = len(targets)

        hm_batch = torch.zeros((batch_size, self.num_classes, output_h, output_w), dtype=torch.float32)
        wh_batch = torch.zeros((batch_size, self.max_objs, 2), dtype=torch.float32)
        reg_batch = torch.zeros((batch_size, self.max_objs, 2), dtype=torch.float32)
        ind_batch = torch.zeros((batch_size, self.max_objs), dtype=torch.int64)
        reg_mask_batch = torch.zeros((batch_size, self.max_objs), dtype=torch.uint8)

        for (i, t) in enumerate(targets):
            add_gt = self.additional_gt(t["boxes"], t["labels"], output_h, output_w,
                                        downratio=downratio, reg_offset=reg_offset)

            hm_batch[i] = add_gt['hm']
            wh_batch[i] = add_gt['wh']
            reg_mask_batch[i] = add_gt['reg_mask']
            ind_batch[i] = add_gt['ind']

            if reg_offset:
                reg_batch[i] = add_gt['reg']

        ret = {'hm': hm_batch, 'reg_mask': reg_mask_batch, 'ind': ind_batch, 'wh': wh_batch}

        if reg_offset:
            ret.update({'reg': reg_batch})

        return ret
