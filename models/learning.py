import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

def build_learner(args, model):
    # loss fn
    criterion = Criterion()
    model.criterion = criterion

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters())

    # scheduler
    lr_scheduler = None
    return criterion, optimizer, lr_scheduler



class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, outputs, gt_instances):
        pass

    @torch.no_grad()
    def postprocess(self, output, track_instances, gt_inst):

        ##### Update Already Assigned objidx
        ii, jj = torch.where(track_instances.obj_idx[:, None] == gt_inst.obj_ids[None]) #N,1 - 1,M
        track_instances.gt_idx[ii] = jj
        track_instances.lives[ii] = 20
        assigned_gt = {int(j) for j in jj}
        assigned_pr = {int(i) for i in ii}

        ##### Prediction With Confidence
        num_proposals = len(track_instances)
        obj = output['is_object'][-1,0,:num_proposals,0] # TODO: 0 at dim=1 is an hack 4 batchsize=1
        track_instances.score = obj.sigmoid().clone().detach()
        
        ##### Predictions Near a GT
        gt_xy = gt_inst.boxes[None,:,:2]                       # 1, M, 2
        xy = output['position'][-1,0,:num_proposals,None,:2]   # N, 1, 2

        # sort by best matchings
        dist = ((xy - gt_xy)**2).sum(dim=-1) # N,M
        n,m = dist.shape
        _,idxs = torch.sort(dist.view(-1))
        ii = (idxs // m).tolist()
        jj = (idxs % m).tolist()

        # assignments
        for i,j in zip(ii,jj):
            if j in assigned_gt or i in assigned_pr:
                continue # you have already assigned that query or that gt_object
                
            track_instances.gt_idx[i]  = j                    # index of ground truth
            track_instances.obj_idx[i] = gt_inst.obj_ids[j]   # object id of the detection in the dataset
            track_instances.lives[i]   = 20                   # keep query for 20 frames
            assigned_gt.add(j)
            assigned_pr.add(i)

        track_instances.drop_miss()








