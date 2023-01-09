"""
GMOT-DETR model and criterion classes.
"""
import os
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import CheckpointFunction
from util.misc.instance import TrackInstances

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
        hidden_dim  = args.embedd_dim

        # self.track_embed = track_embed        # used in post process eval
        # self.use_checkpoint = use_checkpoint  # to look into
        # self.query_denoise = query_denoise   # add noise to GT in training
        self.prebk    = GeneralPreBackbone(args)
        self.backbone = GeneralBackbone(args, self.prebk.ch_out)
        self.mixer    = GeneralMixer(args)
        self.posembed = GeneralPositionEmbedder(args)
        self.decoder  = GeneralDecoder(args)

        if self.mixer.ref_is_none:
            # learned queries when they are not provided by the mixer
            self.ref_pts = nn.Embedding(args.num_queries, 4)
            self.q_embed = nn.Embedding(args.num_queries, hidden_dim)


    def forward(self, data, debug=None):
        """
            data['imgs']         = list_of_CHW_tensors  --> [torch.rand(3,256,256), torch.rand(3,256,256), ...]
            data['gt_instances'] = list_of_gt_instances  --> [inst1, inst2, ...]     , with i containing (boxes, obj_ids)
            data['exemplar']     = list_of_chw_tensors  --> [torch.rand(3,64,64)]
        """

        # initially no tracks
        outputs = []
        track_instances = TrackInstances(self.args.embedd_dim, self.args.det_thresh, self.args.keep_for)
        exemplars = torch.stack(data['exemplar'][:1]) # TODO: support multiple exemplars?
        
        # iterate though all frames
        for frame, gt in zip(data['imgs'], data['gt_instances']):
            noised_gt = self.noise_gt(gt) # noise the ground truth and add it as queries to help the network learn later
            # checkpointed forward pass
            output, track_instances = self._forward_frame(frame.unsqueeze(0), exemplars, noised_gt, track_instances)
            self.criterion.postprocess(output, track_instances, gt) #in learning.py
            outputs.append(output)
            # exemplars = exemplars ##### TODO: maybe update

        if debug:
            with torch.no_grad():
                os.makedirs(self.args.output_dir+'/debug/'+debug.split('/')[-2], exist_ok=True)
                self.debug_infographics(data['imgs'], outputs, data['gt_instances'], 0, debug)
                self.debug_infographics(data['imgs'], outputs, data['gt_instances'], -1, debug)

        return outputs

    def debug_infographics(self, frames, outputs, gt, num, path):
        # where to save the file
        if num==-1: num = len(frames)-1
        path = path+f'f{num}_'

        # info needed on that frame
        frame = frames[num].unsqueeze(0)
        q_ref = outputs[num]['q_ref']
        coord = outputs[num]['position']
        isobj = outputs[num]['is_object']
        n_prop = q_ref.shape[1] - len(gt[num])

        # helper functions for graphics
        self.debug_qref_start(frame, q_ref, n_prop, path)
        self.debug_qref_steps(frame, q_ref, coord, isobj, n_prop, path)


    def _forward_frame(self, frame, exemplars, noised_gt, track_instances):
        """
        Harder function to read, but allows to lower the amount of used GPU ram 
        """
        args = [frame, noised_gt, exemplars]
        params = tuple((p for p in self.parameters() if p.requires_grad))

        dict_outputs = {} # the function will write in dict outputs
        def checkpoint_fn(frame, noised_gt, exemplar):
            """----| REAL FORWARD |----"""

            # extract multi scale features from exemplar [[B,C,h1,w1],[B,C,h2,w2],..]
            exe_features, exe_masks = self.backbone(*self.prebk(exemplar))

            # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
            frame, mask = self.prebk(frame)
            img_features, img_masks = self.backbone(frame, mask)

            # share information between exemplar & input frame
            # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
            img_features, q_prop_refp, add_keys = self.mixer(img_features, exe_features, exe_masks, dict_outputs)

            # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
            # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
            q_queries, q_ref, confidence, attn_mask = \
                self.update_track_instances(img_features, q_prop_refp, track_instances, noised_gt)
            img_features, q_queries, _ = self.posembed(img_features, q_queries, None, q_ref, None, confidence)
            dict_outputs['input_hs'] = q_queries
            dict_outputs['q_ref'] = q_ref

            # TODO: change how ref_pts are updated
            hs, isobj, coord = self.decoder(img_features, add_keys, q_queries, q_ref, attn_mask, img_masks)
            dict_outputs['output_hs'] = hs
            dict_outputs['is_object'] = isobj
            dict_outputs['position']  = coord

            return [v for _,v in dict_outputs.items()]

        CheckpointFunction.apply(checkpoint_fn, len(args), *args, *params)
        return dict_outputs, track_instances

    def update_track_instances(self, img_features, q_prop_refp, track_instances, noised_gt):

        # queries to detect new tracks
        b = img_features[0].shape[0]
        if q_prop_refp is None:
            q_prop_emb  = self.q_embed.weight.data
            q_prop_refp = self.ref_pts.weight.data
        else:
            q_prop_emb  = self.make_q_from_ref(q_prop_refp[0], img_features)

        # queries used to help learning
        if noised_gt is not None:
            q_gt_refp = noised_gt
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
        gt_conf = torch.ones(n_gt) * self.args.keep_for # max confidence
        confidences = torch.cat((track_instances.lives, gt_conf), dim=-1)

        return q_queries.unsqueeze(0), q_ref_pts.unsqueeze(0), confidences, attn_mask

    def make_q_from_ref(self, ref_pts, img_features):
        queries = []
        for f_scale in img_features:
            _,_,h,w = f_scale.shape
            points = (ref_pts[:,:2] * torch.tensor([[w,h]])).long().view(-1, 2)
            q = f_scale[0, :, points[:,1], points[:,0]]
            queries.append(q.T)  # N, C
        queries = torch.stack(queries, dim=0).mean(dim=0)
        return queries

    def noise_gt(self, gt):
        boxes = gt.boxes
        return boxes + torch.rand_like(boxes)*0.08 -0.04

    def _debug_frame(self, frame):
        """util to make frame to writable"""
        frame = np.ascontiguousarray(frame[0].clone().permute(1,2,0).numpy() [:,:,::-1]) /4+0.4 # frame in BGR
        frame = np.uint8(255*(frame-frame.min())/(frame.max()-frame.min()))
        h,w,_ = frame.shape
        return cv2.resize(frame, (400,int(400*h/w)))


    def _debug_qref(self, frame, q_ref, n_prop, opacity=1):
        """util to print q_refs on frame"""
        q_ref = q_ref[0,:,:2]
        H,W,_ = frame.shape
        for i, (w, h) in enumerate(q_ref):
            color = (80,250,90) if i<n_prop else (150,100,240)
            color = tuple((c*opacity for c in color))
            w = int(w*W)
            h = int(h*H)
            frame[h-2:h+3,w-2:w+3] = (frame[h-2:h+3,w-2:w+3].astype(float) * (1-opacity)*0.8).astype(np.uint8)
            frame[h-1:h+2,w-1:w+2] = ((1-opacity)*frame[h-1:h+2,w-1:w+2].astype(float) + color).astype(np.uint8)
        return frame
    
    def debug_qref_start(self, frame, q_ref, n_prop, path):
        """save image with initial reference points"""
        out_file = self.args.output_dir+f'/debug/{path}ref_start.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            frame = self._debug_qref(frame, q_ref, n_prop)
            cv2.imwrite(out_file, frame)

    def debug_qref_steps(self, frame, q_ref, later_ref, scores, n_prop, path):
        """save image with evolution of some ref points"""
        out_file = self.args.output_dir+f'/debug/{path}ref_steps.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            _,i = scores[-1, 0,:n_prop].topk(3, dim=0)
            _,i2 = scores[-1, 0,:n_prop].min(dim=0)
            idxs = i.view(-1).tolist()+i2.tolist()+[0, n_prop]
            q_refs = [q_ref[:,idxs]] 
            q_refs += [ref[:,idxs] for ref in later_ref]
            for i, q_ref in enumerate(q_refs):
                frame = self._debug_qref(frame, q_ref, 5, (i+1)/len(q_refs))
            cv2.imwrite(out_file, frame)

################### visualization of Q-REF