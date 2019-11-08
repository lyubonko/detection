import random
import torch
import math
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import warnings

import numpy as np
import numbers

_IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


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

        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        mask = (center[:, 0] >= x) & (center[:, 0] <= x + w) \
               & (center[:, 1] >= y) & (center[:, 1] <= y + h)

        if mask.any():
            image = image.crop((x, y, x + w, y + h))
            boxes = bbox[mask] - torch.Tensor([x, y, x, y])
            boxes = self.box_clamp(boxes, 0, 0, w, h)
            labels = labels[mask]

            target["boxes"] = boxes
            target["labels"] = labels

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


class RandomCropCOCOSimple(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(scale, ratio, size):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = size[0] * size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = size[0] / size[1]
        if (in_ratio < min(ratio)):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w

    def __call__(self, image, target):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """

        width, height = image.size
        bbox = target["boxes"]

        i, j, h, w = self.get_params(self.scale, self.ratio, image.size)
        image = F.resized_crop(image, i, j, h, w, (height, width), self.interpolation)

        n_boxes = bbox.size(0)
        bbox_new = torch.zeros_like(bbox)
        bbox_new[:, 0] = torch.max(bbox[:, 0] - j, torch.zeros((n_boxes))) * width / w
        bbox_new[:, 1] = torch.max(bbox[:, 1] - i, torch.zeros((n_boxes))) * height / h
        bbox_new[:, 2] = torch.min(bbox[:, 2] - j, w * torch.ones((n_boxes))) * width / w
        bbox_new[:, 3] = torch.min(bbox[:, 3] - i, h * torch.ones((n_boxes))) * height / h

        target["boxes"] = bbox_new

        return image, target

    def __repr__(self):
        interpolate_str = str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class LightingCOCO(object):
    """Lighting noise(AlexNet - style PCA - based noise)
    based on https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
    """

    def __init__(self, alphastd, eigval=_IMAGENET_PCA['eigval'], eigvec=_IMAGENET_PCA['eigvec']):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, image, target):
        if self.alphastd == 0:
            return image, target

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return image.add(rgb.view(3, 1, 1).expand_as(image)), target


class RandomTransforms(object):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApplyCOCO(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApplyCOCO, self).__init__(transforms)
        self.p = p

    def __call__(self, image, target):
        if self.p < random.random():
            return image, target
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomRotationCOCO(object):
    """
    Based on original RandomRotation
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, image, target):
        # original image size
        w_orig, h_orig = image.size
        bbox = target["boxes"]

        angle = self.get_params(self.degrees)
        angle_pi = - angle * np.pi / 180
        # image
        image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center)
        # boxes
        rot_matrix = np.array([[np.cos(angle_pi), -np.sin(angle_pi)],
                               [np.sin(angle_pi), np.cos(angle_pi)]])
        if self.center is not None:
            center_x, center_y = self.center
        else:
            center_x, center_y = (w_orig / 2, h_orig / 2)

        n_boxes = bbox.size(0)

        bbox_new = np.zeros((n_boxes, 4))
        for i in range(n_boxes):
            # xy of bl and tr
            xy = bbox[i].reshape((2, 2))

            # xy of all four corners & relative to the center
            xy_all = np.concatenate((xy, np.array([[xy[0, 0], xy[1, 1]], [xy[1, 0], xy[0, 1]]]))) - \
                     np.array([center_x, center_y])

            # rotate four corners
            xy_all_new = np.transpose(np.dot(rot_matrix, np.transpose(xy_all))) + np.array([center_x, center_y])

            xy_max = np.max(xy_all_new, axis=0)
            xy_mim = np.min(xy_all_new, axis=0)
            bbox_new[i, :] = np.concatenate((xy_mim, xy_max))

        target["boxes"] = torch.from_numpy(bbox_new).float()

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class PadCOCO(object):

    def __init__(self, *args, **kwargs):

        self.image_transform = transforms.Pad(*args, **kwargs)
        self.padding = self.image_transform.padding

    def __call__(self, image, target):
        # image
        image = self.image_transform(image)

        # bboxes
        if isinstance(self.padding, tuple):
            if len(self.padding) == 2:
                (left, top) = self.padding
            elif len(self.padding) == 4:
                (left, top, _, _) = self.padding
        else:
            left = top = self.padding

        pads = np.array([left, top, left, top])
        target["boxes"] += torch.from_numpy(pads).float()

        return image, target

    def __repr__(self):
        return self.image_transform.__repr__()


class MaxCropCOCO(object):

    def __init__(self, offset=5):
        self.offset = offset

    @staticmethod
    def box_clamp(boxes, xmin, ymin, xmax, ymax):
        '''Clamp boxes.

        Args:
          boxes: (tensor) bounding boxes of (xmin, ymin, xmax, ymax), sized [N,4].
          xmin: (number) min value of x.
          ymin: (number) min value of y.
          xmax: (number) max value of x.
          ymax: (number) max value of y.

        Returns:
          (tensor) clamped boxes.
        '''
        if type(boxes) == np.ndarray:
            boxes[:, 0] = np.clip(boxes[:, 0], xmin, xmax)
            boxes[:, 1] = np.clip(boxes[:, 1], ymin, ymax)
            boxes[:, 2] = np.clip(boxes[:, 2], xmin, xmax)
            boxes[:, 3] = np.clip(boxes[:, 3], ymin, ymax)
        else:
            boxes[:, 0].clamp_(min=xmin, max=xmax)
            boxes[:, 1].clamp_(min=ymin, max=ymax)
            boxes[:, 2].clamp_(min=xmin, max=xmax)
            boxes[:, 3].clamp_(min=ymin, max=ymax)
        return boxes

    def __call__(self, image, target):
        w_orig, h_orig = image.size
        bbox = target["boxes"]

        max_x, max_y = torch.max(bbox[:, 2:], axis=0)[0]
        min_x, min_y = torch.min(bbox[:, :2], axis=0)[0]

        # update in light of offset restriction
        min_y = torch.clamp(min_y - self.offset, min=0)
        max_y = torch.clamp(max_y + self.offset, max=h_orig)
        min_x = torch.clamp(min_x - self.offset, min=0)
        max_x = torch.clamp(max_x + self.offset, max=w_orig)

        bbox = bbox - torch.tensor([min_x, min_y, min_x, min_y]).float()
        bbox = self.box_clamp(bbox, 0, 0, max_x - min_x, max_y - min_y)

        image = image.crop((min_x.item(), min_y.item(), max_x.item(), max_y.item()))
        target["boxes"] = bbox

        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
