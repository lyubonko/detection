import torch


def prepare_optimizer(args, model):

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.TRAIN.lr, momentum=args.TRAIN.momentum, weight_decay=args.TRAIN.weight_decay)

    # TODO: Check this stuff, do I need this (was commented in original code)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.TRAIN.lr_steps,
                                                        gamma=args.TRAIN.lr_gamma)
    return optimizer, lr_scheduler
