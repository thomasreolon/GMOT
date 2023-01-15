import copy
import math
import numpy as np
import torch
from torch import nn, Tensor

from util.misc import TrackInstances, Instances
from ._loss import loss_fn_getter


class Criterion(nn.Module):
    def __init__(self, args, model) -> None:
        super().__init__()
        self.args = args
        self.matcher = get_matcher(args)
        self.losses = {
            'is_object':1,    # FOCAL_LOSS: if there is an object or no 
            'boxes':1,     # L1_LOSS: position of bounding boxes
            'giou':1,        # IntersectionOverUnion: position of bounding boxes (generalized)
            # 'fake': 0,         # just for testing..

        } # TODO: select losses as args parameter
        self.required = set(self.matcher.required+sum([loss_fn_getter(loss).required for loss in self.losses], []))

    def forward(self, outputs, gt_instances):
        dict_losses = {}
        return dict_losses

    def forward_intra_frame(self, track_instances:TrackInstances, output:dict, gt_instance:Instances):
        assert any((x>0 for x in self.losses.values())), "At least one loss must be non zero"
        
        # assigns .gt_idx ans .obj_idx
        track_instances = self.matcher(track_instances, output, gt_instance)

        # compute losses
        dict_losses = {}
        for loss, multiplier in self.losses.items():
            loss_fn = loss_fn_getter(loss)
            if loss_fn.is_intra_loss:
                # computes a loss
                if multiplier > 0:
                    dict_losses['loss_'+loss] = multiplier * loss_fn(track_instances, output, gt_instance).sum()
            else:
                # return a dict of tensors that will be used later
                dict_losses.update( loss_fn(track_instances, output, gt_instance) )

        return track_instances, dict_losses

    def forward_inter_frame(self, losses_dict:dict):
        final_losses = {k:0 for k in losses_dict[0].keys() if 'loss_' in k}
        coeff_frame_loss = self._get_loss_coeff(losses_dict)
        for loss_dict, coeff in zip(losses_dict, coeff_frame_loss):
            final_losses = {k:  v1+loss_dict[k]   for k,v1  in final_losses.items()}
            # final_losses = {k:  v1+loss_dict[k]*coeff[k]   for k,v1  in final_losses.items()}
        
        for loss, multiplier in self.losses.items():
            loss_fn = loss_fn_getter(loss)
            if not loss_fn.is_intra_loss and multiplier > 0:
                final_losses['loss_'+loss] = multiplier * loss_fn(losses_dict).sum()

        return {k[5:]:v for k,v in final_losses.items()}


    @torch.no_grad()
    def _get_loss_coeff(self, losses_dict):
        """even the loss for the frames --> if frame_4 has a big loss it's also because frame 1 predsiction are not the best..."""
        l1, l2 = losses_dict[0], losses_dict[-1]
        coeff_frame_loss = [dict() for _ in range(len(losses_dict))]
        for k in l1.keys():
            if 'loss_' not in k: continue
            rateo = (l2[k] - (l1[k]+1e-8)) / (len(losses_dict)-0.98)
            base = 1 if rateo>=0 else 1+l1[k]-l2[k]
            coeffs = base + torch.arange(len(losses_dict), device=l1[k].device, dtype=l1[k].dtype)*rateo
            for i,c in enumerate(coeffs):
                coeff_frame_loss[i][k] = 1/c
    
        return coeff_frame_loss


class SimpleMatcher():
    def __init__(self) -> None:
        self.required = ['is_object', 'boxes']

    @torch.no_grad()
    def __call__(self, track_instances:TrackInstances, output:dict, gt_inst:Instances):

        # assign GT queries to their GT
        track_gt = track_instances.get_gt_queries()
        track_gt.gt_idx = torch.arange(len(track_gt), device=track_instances.gt_idx.device).int()

        # assign prediction queries to their GT
        track_to_match = track_instances.get_prevnew_queries()

        assigned_pr = {int(i) for i in torch.where(track_to_match.gt_idx>=0)[0]} # can't assign query to more than one target
        track_to_match.gt_idx[:] = -1  # deletes previous assignments

        ##### Update Already Assigned objidx
        ii, jj = torch.where(track_to_match.obj_idx[:, None] == gt_inst.obj_ids[None]) #N,1 - 1,M
        track_to_match.gt_idx[ii] = jj
        assigned_gt = {int(j) for j in jj}
        assigned_pr = assigned_pr.union({int(i) for i in ii})

        ##### Prediction With Confidence
        num_proposals = len(track_to_match)
        obj = output['is_object'][-1,0,:num_proposals,0] # TODO: 0 at dim=1 is an hack 4 batchsize=1
        track_to_match.score = obj.sigmoid().clone().detach()
        
        ##### Predictions Near a GT
        gt_xy = gt_inst.boxes[None,:,:2]                       # 1, M, 2
        xy = output['boxes'][-1,0,:num_proposals,None,:2]   # N, 1, 2

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

            track_to_match.gt_idx[i]  = j                    # index of ground truth
            track_to_match.obj_idx[i] = gt_inst.obj_ids[j].item()   # object id of the detection in the dataset
            assigned_gt.add(j)
            assigned_pr.add(i)

        return TrackInstances.cat([track_to_match, track_gt])



def build_learner(args, model):
    # optimizer
    backbone = {
        "params": [p for p in model.backbone.parameters()],
        "lr": args.lr_backbone,
    }
    fast_learn = {
        "params": [p for n,p in model.named_parameters() if n in args.lr_linear_proj_names],
        "lr": args.lr * args.lr_linear_proj_mult,
    }
    prev = set(backbone['params']).union(fast_learn['params'])
    other = {
        "params": [p for p in model.parameters() if p not in prev],
        "lr": args.lr,
    }
    optimizer = torch.optim.AdamW([backbone, fast_learn,other], lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    return optimizer, lr_scheduler


def get_matcher(args):
    if args.matcher == 'simple':
        return SimpleMatcher()
    else:
        raise NotImplementedError(args.matcher)
