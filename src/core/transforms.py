import random
import torch
import math
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# ===== new transforms


class TransformsCOCO(object):
    """
    notes: 'Lambda' and  'ToTensor' only acting on image,
           for generalized Lambda (both image + bboxes) use LambdaCOCO

    for free: 'Compose', 'RandomTransforms', 'RandomApply', 'RandomOrder', 'RandomChoice',
    """

    def __init__(self, transform_image, *args, **kwargs):
        assert transform_image in ['ColorJitter', 'Normalize', 'ToTensor', 'ToPILImage', 'Lambda', 'Grayscale'], \
            "The transform only works for ['ColorJitter', 'Normalize', 'ToTensor', 'ToPILImage', 'Lambda', 'Grayscale']"
        self.image_transform = getattr(transforms, transform_image)(*args, **kwargs)

    def __call__(self, image, target):
        image = self.image_transform(image)
        return image, target

    def __repr__(self):
        return self.image_transform.__repr__()


class RandomCropCOCO(object):
    """
    This random crop strategy is described in original paper "SSD: Single Shot MultiBox Detector"
    """

    def __init__(self, min_scale=0.3, max_aspect_ratio=2.):
        self.min_scale = min_scale
        self.max_aspect_ratio = max_aspect_ratio

    @staticmethod
    def box_iou(box1, box2):
        """Compute the intersection over union of two set of boxes.

        The box order must be (xmin, ymin, xmax, ymax).

        Args:
          box1: (tensor or np.array) bounding boxes, sized [N,4].
          box2: (tensor or np.array) bounding boxes, sized [M,4].

        Return:
          (tensor or np.array) iou, sized [N,M].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        """

        if box1.shape[1] != 4 or box2.shape[1] != 4:
            raise IndexError

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        iou = inter / (area1[:, None] + area2 - inter)

        return iou

    @staticmethod
    def box_clamp(boxes, xmin, ymin, xmax, ymax):
        """Clamp boxes.

        Args:
          boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [N,4].
          xmin: (number) min value of x.
          ymin: (number) min value of y.
          xmax: (number) max value of x.
          ymax: (number) max value of y.

        Returns:
          (tensor) clamped boxes.
        """
        boxes[:, 0].clamp_(min=xmin, max=xmax)
        boxes[:, 1].clamp_(min=ymin, max=ymax)
        boxes[:, 2].clamp_(min=xmin, max=xmax)
        boxes[:, 3].clamp_(min=ymin, max=ymax)
        return boxes

    def __call__(self, image, target):

        width, height = image.size
        bbox = target["boxes"]
        labels = target["labels"]

        params = [(0, 0, width, height)]  # crop roi (x,y,w,h) out
        for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
            for _ in range(100):
                scale = random.uniform(self.min_scale, 1)
                aspect_ratio = random.uniform(
                    max(1 / self.max_aspect_ratio, scale * scale),
                    min(self.max_aspect_ratio, 1 / (scale * scale)))
                w = int(width * scale * math.sqrt(aspect_ratio))
                h = int(height * scale / math.sqrt(aspect_ratio))

                x = random.randint(0, width - w)
                y = random.randint(0, height - h)

                roi = torch.Tensor([[x, y, x + w, y + h]])
                ious = self.box_iou(bbox, roi)
                if ious.min() >= min_iou:
                    params.append((x, y, w, h))
                    break

        random_indx = random.choice(range(len(params)))
        x, y, w, h = params[random_indx]
        image = image.crop((x, y, x + w, y + h))

        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        mask = (center[:, 0] >= x) & (center[:, 0] <= x + w) \
               & (center[:, 1] >= y) & (center[:, 1] <= y + h)
        if mask.any():
            boxes = bbox[mask] - torch.Tensor([x, y, x, y])
            boxes = self.box_clamp(boxes, 0, 0, w, h)
            labels = labels[mask]
        else:
            boxes = torch.Tensor([[0, 0, 0, 0]])
            labels = torch.Tensor([0])

        target["boxes"] = boxes
        target["labels"] = labels

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
