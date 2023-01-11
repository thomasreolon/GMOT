# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from typing import Iterable
import math
import gc, os
import sys
import cv2
import numpy as np

import torch

import util.multiprocess as utils
from .misc import Visualizer

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args: dict, debug:str):

    max_norm, out_d = args.clip_max_norm,  args.output_dir
    model.train()
    criterion.train()

    visualizer = Visualizer(args)

    # Set Up Logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        gc.collect(); torch.cuda.empty_cache()

        # Sequence of 5 frames from the same video (show GT for debugging)
        if debug:
            visualizer.visualize_gt(data_dict)
        
        exit()

        # Forward
        data_dict = utils.data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict, debug=debug)

        # Compute Loss
        loss_dict = criterion(outputs, data_dict)
        # weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # # Reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # loss_value = losses_reduced_scaled.item()
        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        # # Backward
        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # else:
        #     utils.get_total_grad_norm(model.parameters(), max_norm)
        # optimizer.step()

        # # Get Num Predictions 
        # dt_instances = model.module.post_process(outputs['track_instances'], data_dict['imgs'][0].shape[-2:])
        # keep = dt_instances.scores > .02
        # keep &= dt_instances.obj_idxes >= 0
        # dt_instances = dt_instances[keep]

        # # Log
        # metric_logger.update(num_det=len(dt_instances), num_gt=len(data_dict['gt_instances'][-1]))
        # metric_logger.update(loss=loss_value, **{k:v for k,v in loss_dict_reduced_scaled.items() if 'aux' not in k})
        if debug:
            for i in range(len(data_dict['imgs'])):
                visualizer.visualize_infographics(data_dict['imgs'], data_dict['gt_instances'], outputs, i, debug)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
