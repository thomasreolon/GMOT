import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from ._loss import loss_fn_getter

def build_learner(args, model):
    # loss fn
    criterion = Criterion(args, model)
    model.criterion = criterion

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # scheduler
    lr_scheduler = None
    return criterion, optimizer, lr_scheduler



class Criterion(nn.Module):
    def __init__(self, args, model) -> None:
        super().__init__()
        self.args = args
        self.losses = {
            'is_object':2,    # FOCAL_LOSS: if there is an object or no 
            'position':1,     # L1_LOSS: position of bounding boxes
            'giou':1,         # IntersectionOverUnion: position of bounding boxes (generalized)

        } # TODO: select losses as args parameter

    def forward(self, outputs, gt_instances):
        dict_losses = {}
        for loss, multiplier in self.losses.items():
            loss_fn = loss_fn_getter(loss)
            dict_losses[loss] = multiplier * loss_fn(outputs, gt_instances)
        return dict_losses

    @torch.no_grad()
    def postprocess(self, output, track_instances, gt_inst):

        ##### Update Already Assigned objidx
        ii, jj = torch.where(track_instances.obj_idx[:, None] == gt_inst.obj_ids[None]) #N,1 - 1,M
        track_instances.gt_idx[ii] = jj
        assigned_gt = {int(j) for j in jj}
        assigned_pr = {int(i) for i in ii}
        debug_assignments = torch.zeros(output['is_object'].shape[2]).int()
        debug_assignments[ii] = 2

        ##### Prediction With Confidence
        num_proposals = len(track_instances)
        obj = output['is_object'][-1,0,:num_proposals,0] # TODO: 0 at dim=1 is an hack 4 batchsize=1
        track_instances.score = obj.sigmoid().clone().detach()
        debug_assignments[num_proposals:] = 1
        
        ##### Predictions Near a GT
        gt_xy = gt_inst.boxes[None,:,:2]                       # 1, M, 2
        xy = output['position'][-1,0,:num_proposals,None,:2]   # N, 1, 2

        # sort by best matchings
        dist = ((xy - gt_xy)**2).sum(dim=-1) # N,M
        n,m = dist.shape
        _,idxs = torch.sort(dist.view(-1))
        ii = idxs.div(m, rounding_mode='trunc').tolist()
        jj = (idxs % m).tolist()

        # assignments # TODO: use hungarian for matching and add other cost other than center_distance
        for i,j in zip(ii,jj):
            if j in assigned_gt or i in assigned_pr:
                continue # you have already assigned that query or that gt_object
                
            track_instances.gt_idx[i]  = j                    # index of ground truth
            track_instances.obj_idx[i] = gt_inst.obj_ids[j]   # object id of the detection in the dataset
            assigned_gt.add(j)
            assigned_pr.add(i)
            debug_assignments[i] = 3

        debug_assignments[output['is_object'][-1,0,:,0]>self.args.det_thresh] += 10
        
        output['debug'] = debug_assignments                     # to draw colored predictions
        output['matching_gt'] = track_instances.gt_idx          # for losses
        output['matching_obj'] = track_instances.obj_idx        # for losses

        track_instances.q_ref = output['position'][-1,0,:num_proposals]
        track_instances.q_emb = output['output_hs'][-1,0,:num_proposals] # TODO: find way to update them removing position information
        track_instances.drop_miss()








