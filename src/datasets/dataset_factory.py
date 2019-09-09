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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def prepare_data(args):

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
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.DATASET.batch_size_test,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test, train_sampler, num_classes
