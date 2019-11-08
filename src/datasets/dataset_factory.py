import torch
from core import utils, transforms as T
from core.coco_utils import get_coco, get_coco_kp
from datasets.dataset_penn import get_dataset_penn

from core.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
        "penn": (data_path, get_dataset_penn, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform_0(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_transform_1(train):
    transforms = [T.ToTensor()]
    if train:
        transforms = [
            # transforms on PIL.Image
            T.TransformsCOCO('ColorJitter', brightness=0.4, contrast=.4, saturation=.4),
            T.RandomCropCOCO(),
            # to tensor
            T.ToTensor(),
            # transforms on Tensor
            T.LightingCOCO(0.1),
            T.RandomHorizontalFlip(0.5)]
    return T.Compose(transforms)


def get_transform_2(train):
    transforms = [T.ToTensor()]
    if train:
        transforms = [
            # transforms on PIL.Image
            T.TransformsCOCO('ColorJitter', brightness=0.4, contrast=.4, saturation=.4),
            T.RandomCropCOCOSimple(scale=(0.6, 1.3)),
            # to tensor
            T.ToTensor(),
            # transforms on  Tensor
            T.RandomApplyCOCO([T.LightingCOCO(0.1)], 0.3),
            T.RandomHorizontalFlip(0.5)
        ]
    return T.Compose(transforms)


def get_transform_3(train):
    transforms = [T.ToTensor()]
    if train:
        transforms = [
            # transforms on PIL.Image
            T.TransformsCOCO('ColorJitter', brightness=0.4, contrast=.4, saturation=.4),
            T.RandomApplyCOCO([T.PadCOCO(100, padding_mode='edge'), T.RandomRotationCOCO(15), T.MaxCropCOCO()], 1.),
            #T.RandomApplyCOCO([T.PadCOCO(100, padding_mode='edge'), T.MaxCropCOCO()], 1.),
            T.RandomCropCOCOSimple(scale=(0.6, 1.3)),
            # to tensor
            T.ToTensor(),
            # transforms on  Tensor
            T.RandomApplyCOCO([T.LightingCOCO(0.1)], 0.3),
            T.RandomHorizontalFlip(0.5)
        ]
    return T.Compose(transforms)


def prepare_data(args):

    if args.DATASET.transform == 'transform_0':
        get_transform = get_transform_0
    elif args.DATASET.transform == 'transform_1':
        get_transform = get_transform_1
    elif args.DATASET.transform == 'transform_2':
        get_transform = get_transform_2
    elif args.DATASET.transform == 'transform_3':
        get_transform = get_transform_3

    dataset, num_classes = get_dataset(args.DATASET.name, "train", get_transform(train=True), args.DATASET.path)
    dataset_test, _ = get_dataset(args.DATASET.name, "val", get_transform(train=False), args.DATASET.path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.DATASET.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.DATASET.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.DATASET.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.DATASET.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.DATASET.batch_size_test,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    return data_loader, data_loader_test, train_sampler, num_classes
