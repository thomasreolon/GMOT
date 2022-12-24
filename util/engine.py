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


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args: dict):

    max_norm, debug, out_d = args.clip_max_norm, args.debug, args.output_dir
    model.train()
    criterion.train()

    # Set Up Logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        gc.collect(); torch.cuda.empty_cache()

        # Sequence of 5 frames from the same video (show GT for debugging)
        if debug and 'gt_visualize.jpg' not in os.listdir(out_d+'/debug'):
            os.makedirs(args.output_dir+'/debug/', exist_ok=True)
            imgs = data_dict['imgs']
            concat = torch.cat(imgs, dim=1)
            concat = np.ascontiguousarray(concat.clone().permute(1,2,0).numpy() [:,:,::-1])
            for i in range(len(imgs)):
                for box in data_dict['gt_instances'][i].boxes:
                    box = (box.view(2,2) * torch.tensor([imgs[0].shape[2], imgs[0].shape[1]]).view(1,2)).int()
                    x1,x2 = box[0,0] - box[1,0]//2, box[0,0] + box[1,0]//2
                    y1,y2 = box[0,1] - box[1,1]//2, box[0,1] + box[1,1]//2
                    y1, y2 = y1+imgs[0].shape[1]*i, y2+imgs[0].shape[1]*i
                    tmp = concat[y1:y2, x1:x2].copy()
                    concat[y1-2:y2+2, x1-2:x2+2] = 1
                    concat[y1:y2, x1:x2] = tmp
            concat = cv2.resize(concat, (400, 1300))
            cv2.imwrite(out_d+'/debug/gt_visualize.jpg', concat)

        # Forward
        data_dict = utils.data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        # Compute Loss
        loss_dict = criterion(outputs, data_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # Get Num Predictions 
        dt_instances = model.module.post_process(outputs['track_instances'], data_dict['imgs'][0].shape[-2:])
        keep = dt_instances.scores > .02
        keep &= dt_instances.obj_idxes >= 0
        dt_instances = dt_instances[keep]

        # Log
        metric_logger.update(num_det=len(dt_instances), num_gt=len(data_dict['gt_instances'][-1]))
        metric_logger.update(loss=loss_value, **{k:v for k,v in loss_dict_reduced_scaled.items() if 'aux' not in k})
        if debug and f'predict_{epoch}.jpg' not in os.listdir(out_d+'/debug'):
            wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
            areas = wh[:, 0] * wh[:, 1]
            keep = areas > 100
            dt_instances = dt_instances[keep]

            if len(dt_instances)>0:
                bbox_xyxy = dt_instances.boxes.tolist()
                identities = dt_instances.obj_idxes.tolist()

                img = data_dict['imgs'][-1].clone().cpu().permute(1,2,0).numpy()[:,:,::-1]
                for xyxy, track_id in zip(bbox_xyxy, identities):
                    if track_id < 0 or track_id is None:
                        continue
                    x1, y1, x2, y2 = [int(a) for a in xyxy]

                    tmp = img[ y1:y2, x1:x2].copy()
                    img[y1-3:y2+3, x1-3:x2+3] = (0,2.3,0)
                    img[y1:y2, x1:x2] = tmp
                cv2.imwrite(out_d+f'/debug/predict_{epoch}.jpg', concat)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
