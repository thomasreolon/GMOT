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
import gc
import sys
import numpy as np

import torch
from models.gmot import GMOT

import util.multiprocess as utils

# TODO: batchsize
# TODO: in main: fn "change params between epochs"   t=[t1,t2,t3,t2,t1];  new_p=t[int(len(t)*epoch/n_epochs)]  ;  set_param(new_p)

def train_one_epoch(model: GMOT, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args: dict, debug:str):

    model.train()

    # Set Up Logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    visualizations = {2, 10, 30, 80, 150, 300, len(data_loader)//2, len(data_loader)*4//5}

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for d_i, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        di_debug = debug+f'b{d_i}_' if debug and d_i in visualizations else None

        # To cuda
        data_dict = utils.data_dict_to_cuda(data_dict, device)

        # Forward
        loss_dict = model(data_dict, debug=di_debug)

        # Compute Loss
        losses = sum(loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        if d_i%print_freq==0:
            loss_dict_reduced = {k: v for k, v in utils.reduce_dict(loss_dict).items()}
            loss_value = sum(loss_dict_reduced.values()).item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        # Backward
        optimizer.zero_grad()
        losses.backward()
        
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        else:
            utils.get_total_grad_norm(model.parameters(), args.clip_max_norm)
        optimizer.step()

        # # Log
        metric_logger.update(loss=loss_value, **{k:v for k,v in loss_dict_reduced.items()})
        if debug and d_i == 0:
            # Sequence of 5 frames from the same video (show GT for debugging)
            model.visualizer.visualize_gt(data_dict)

        # # empty cache
        del loss_dict, data_dict, losses
        gc.collect(); torch.cuda.empty_cache()


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
