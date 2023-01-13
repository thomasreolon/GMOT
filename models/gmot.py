"""
GMOT-DETR model and criterion classes.
"""
import os
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import CheckpointFunction, decompose_output, TrackInstances

from .prebackbone import GeneralPreBackbone
from .backbone import GeneralBackbone
from .mixer import GeneralMixer
from .positionembedd import GeneralPositionEmbedder
from .decoder import GeneralDecoder

def build(args):
    model = GMOT(args)
    print("Model Num Parameters: {}".format(
        sum(p.numel() for p in model.parameters())
    ))
    return model


class GMOT(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.args   = args

        # self.track_embed = track_embed        # used in post process eval
        # self.use_checkpoint = use_checkpoint  # to look into
        # self.query_denoise = query_denoise    # add noise to GT in training
        self.prebk    = GeneralPreBackbone(args)
        self.backbone = GeneralBackbone(args, self.prebk.ch_out)
        self.mixer    = GeneralMixer(args)
        self.posembed = GeneralPositionEmbedder(args)
        self.decoder  = GeneralDecoder(args)

        if self.mixer.ref_is_none:
            # learned queries when they are not provided by the mixer
            self.ref_pts = nn.Embedding(args.num_queries, 4)
            self.q_embed = nn.Embedding(args.num_queries, args.embedd_dim)


    def forward(self, data, debug=None):
        """
            data['imgs']         = list_of_CHW_tensors  --> [torch.rand(3,256,256), torch.rand(3,256,256), ...]
            data['gt_instances'] = list_of_gt_instances  --> [inst1, inst2, ...]     , with i containing (boxes, obj_ids)
            data['exemplar']     = list_of_chw_tensors  --> [torch.rand(3,64,64)]
        """

        # initially no tracks
        outputs = []
        if 'masks' not in data: data['masks'] = [None for _ in range(len(data['imgs']))]
        track_instances = TrackInstances(self.args.embedd_dim, self.args.det_thresh, self.args.keep_for).to(data['imgs'][0].device)
        exemplars = torch.stack(data['exemplar'][:1]) # TODO: support multiple exemplars?
        
        # iterate though all frames
        for frame, gt, mask in zip(data['imgs'], data['gt_instances'], data['masks']):
            noised_gt = self.noise_gt(gt) # noise the ground truth and add it as queries to help the network learn later
            # checkpointed forward pass
            output, track_instances = self._forward_frame(frame.unsqueeze(0), exemplars, noised_gt, track_instances, mask, gt)
            self.criterion.postprocess(output, track_instances, gt) #in learning.py
            outputs.append(output)
            # exemplars = exemplars... ##### TODO: maybe update
        
        return outputs


    def _forward_frame(self, frame, exemplars, noised_gt, track_instances, b_mask, gt_inst):
        """
        Harder function to read, but allows to lower the amount of used GPU ram 
        """
        args = [frame, noised_gt, exemplars, b_mask]
        params = tuple((p for p in self.parameters() if p.requires_grad))

        dict_outputs = {} # the function will write in dict outputs
        def checkpoint_fn(frame, noised_gt, exemplar, b_mask):
            """----| REAL FORWARD |----"""

            # extract multi scale features from exemplar [[B,C,h1,w1],[B,C,h2,w2],..]
            exe_features, exe_masks = self.backbone(*self.prebk(exemplar))

            # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
            frame, mask = self.prebk(frame)
            b_mask = mask if b_mask is None else b_mask[None] | mask
            img_features, img_masks = self.backbone(frame, b_mask)
            dict_outputs['img_features'] = img_features

            # share information between exemplar & input frame
            # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
            img_features, q_prop_refp, add_keys = self.mixer(img_features, exe_features, exe_masks, dict_outputs)
            dict_outputs['img_features_mix'] = img_features

            # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
            # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
            dict_outputs['n_prev'] = len(track_instances)
            q_queries, q_ref, confidence, attn_mask = \
                self.update_track_instances(img_features, q_prop_refp, track_instances, noised_gt)
            dict_outputs['q_ref'] = q_ref
            dict_outputs['n_prop'] = len(track_instances)

            img_features, q_queries, _ = self.posembed(img_features, q_queries, None, q_ref, None, confidence, dict_outputs['n_prev'])
            dict_outputs['input_hs'] = q_queries
            dict_outputs['img_features_pos'] = img_features

            # TODO: change how ref_pts are updated
            hs, isobj, coord = self.decoder(img_features, add_keys, q_queries, q_ref, attn_mask, img_masks)
            coord = self._fix_pad(coord, mask)
            dict_outputs['output_hs'] = hs
            dict_outputs['is_object'] = isobj
            dict_outputs['position']  = coord

            if not any(['loss_' in k for k in dict_outputs]):
                self.criterion([dict_outputs], [gt_inst])
            gradients = {k:v for k,v in dict_outputs.items() if 'loss_' in k}

            return decompose_output(gradients)

        CheckpointFunction.apply(checkpoint_fn, len(args), *args, *params)
        return dict_outputs, track_instances

    def _fix_pad(self, coord, mask):
        if mask.int().sum() == 0: return coord # no padding done
        _,h,w = mask.shape
        nh,nw = h-mask[0,:,w//2].int().sum(), w-mask[0,h//2].int().sum()
        padt, padl = mask[0,:h//2, w//2].int().sum(), mask[0,h//2,:w//2].int().sum()
        coord_xy = coord[...,:2] - torch.tensor([padl/w, padt/h], device=coord.device)
        coord = torch.cat((coord_xy, coord[...,2:]), dim=-1)
        coord = coord * torch.tensor([w/nw, h/nh, w/nw, h/nh], device=coord.device)
        return coord

    def update_track_instances(self, img_features, q_prop_refp, track_instances, noised_gt):
        # TODO: batchsize=1 is still mandatory (probably 4ever)

        # queries to detect new tracks
        if q_prop_refp is None:
            q_prop_emb  = self.q_embed.weight.data
            q_prop_refp = self.ref_pts.weight.data.sigmoid()
        else:
            q_prop_emb  = self.make_q_from_ref(q_prop_refp[0], img_features)

        # queries used to help learning
        if noised_gt is not None:
            q_gt_refp = noised_gt.clamp(0,0.9998)
            q_gt_emb  = self.make_q_from_ref(q_gt_refp, img_features)
            n_gt      = q_gt_refp.shape[0]
        else:
            q_gt_refp = torch.zeros((0, 4))
            q_gt_emb  = torch.zeros((0, q_prop_emb.shape[1]))
            n_gt      = 0

        
        # add queries to detect new tracks to track_instances
        track_instances.add_new(q_prop_emb, q_prop_refp)

        # final queries for the decoder
        q_queries = torch.cat((track_instances.q_emb, q_gt_emb), dim=-2)   # N,256
        q_ref_pts = torch.cat((track_instances.q_ref, q_gt_refp), dim=-2)  # N,4

        n_tot = q_queries.shape[-2]
        attn_mask = torch.zeros((n_tot, n_tot), dtype=bool, device=q_queries.device)
        attn_mask[:n_gt, n_gt:] = True

        # confidences for position embedding
        gt_conf = torch.ones(n_gt, device=q_queries.device) * self.args.keep_for # max confidence
        confidences = torch.cat((track_instances.lives, gt_conf), dim=-1)

        return q_queries.unsqueeze(0), q_ref_pts.unsqueeze(0), confidences, attn_mask

    def make_q_from_ref(self, ref_pts, img_features):
        queries = []
        for f_scale in img_features:
            _,_,h,w = f_scale.shape
            points = (ref_pts[:,:2] * torch.tensor([[w,h]],device=ref_pts.device)).long().view(-1, 2)
            q = f_scale[0, :, points[:,1], points[:,0]]
            queries.append(q.T)  # N, C
        queries = torch.stack(queries, dim=0).mean(dim=0)
        return queries

    def noise_gt(self, gt):
        boxes = gt.boxes
        return boxes + torch.rand_like(boxes)*0.08 -0.04
