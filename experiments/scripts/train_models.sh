#!/bin/bash

# this series to evaluate different models
python src/train.py --param_file experiments/cfgs/params_detector.yaml \
                   --TRAIN_weight_decay 0.0005
python src/train.py --param_file experiments/cfgs/params_detector.yaml \
                   --TRAIN_weight_decay 0.0001
python src/train.py --param_file experiments/cfgs/params_detector.yaml \
                   --TRAIN_weight_decay 0.001
python src/train.py --param_file experiments/cfgs/params_detector.yaml \
                   --TRAIN_lr 0.02
python src/train.py --param_file experiments/cfgs/params_detector.yaml \
                   --TRAIN_lr 0.005