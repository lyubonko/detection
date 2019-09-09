#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

import core.transforms as T
from utils.plot_utils import plot_single_image
from datasets.coco_to_pd import read_coco_classes

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# === pathes
pathes = {}
pathes['project_root'] = '../'
pathes['data'] = os.path.join(pathes['project_root'], 'materials/images/')

# === load image
img_path = os.path.join(pathes['data'], '2007_000720.jpg')
img = Image.open(img_path)

# === load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)

model.to(device)

# === transform image
tr_imgs = get_transform(False)
img_tensor, _ = tr_imgs(img, None)

# === run model
model.eval()
with torch.no_grad():
    prediction = model([img_tensor.to(device)])

predict_boxes = prediction[0]['boxes'].cpu().numpy()
predict_labels = prediction[0]['labels'].cpu().numpy()
predict_scores = prediction[0]['scores'].cpu().numpy()
predict_n = len(predict_labels)

# TODO: make use of mask also in plotting the results
if 'masks' in prediction[0].keys():
    predict_masks = np.squeeze(prediction[0]['masks'].mul(255).byte().cpu().numpy())

# === threshold
vis_thresh = 0.5

# === prepare results for plotting
coco_classes, _ = read_coco_classes(
    os.path.join(pathes['project_root'], 'data/coco_classes.txt'))
coco_list = [x[1] for x in sorted([(k, coco_classes[k]) for k in coco_classes], key=lambda x: x[0])]

bbox_text = []
bboxes = []
for k in range(predict_n):
    # if predict_scores[k] > vis_thresh and coco_classes[predict_labels[k]] == 'laptop':
    if predict_scores[k] > vis_thresh:
        current_box = [predict_boxes[k][0], predict_boxes[k][1],
                       predict_boxes[k][2] - predict_boxes[k][0], predict_boxes[k][3] - predict_boxes[k][1]]
        bboxes.append(current_box)
        bbox_text.append(coco_classes[predict_labels[k]] + '[{:3.2f}]'.format(predict_scores[k]))

box_info = {'bbox_text': bbox_text,
            'bbox_text_pred': [],
            'bbox_text_pred_2': [],
            'bboxes': bboxes,
            'bboxes_pred': [],
            'bboxes_pred_2': [],
            'path_image': img_path}

plot_single_image(box_info, fig_size=18, make_print=False)
plt.savefig('../materials/images/out_demo.png')
