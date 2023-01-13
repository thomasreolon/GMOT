"""
GMOT-DETR model and criterion classes.
"""
import torch
from torch import nn, Tensor

from util.misc import CheckpointFunction, decompose_output, TrackInstances, Instances

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


    def forward(self, data:dict):
        """ (batch size forcefully == 1)

            Arguments:
                data['imgs']         = list_of_CHW_tensors  --> [torch.rand(3,256,256), torch.rand(3,256,256), ...]
                data['gt_instances'] = list_of_gt_instances  --> [inst1, inst2, ...]     , with i containing (boxes, obj_ids)
                data['exemplar']     = list_of_chw_tensors  --> [torch.rand(3,64,64)]
            
            Returns:
                ...
        """
        if 'masks'        not in data: data['masks']        = [None for _ in range(len(data['imgs']))]
        # if 'gt_instances' not in data: data['gt_instances'] = [None for _ in range(len(data['imgs']))]  # gt always required for matching

        self.args.use_checkpointing = True
        process_frame_fn = self._forward_frame_train_ckpt if self.args.use_checkpointing else self.forward_frame_train

        # initializations
        outputs, losses = [], []
        device = data['imgs'][0].device
        track_instances = TrackInstances(vars(self.args), init=True).to(device)
        exemplars = torch.stack(data['exemplar'][:1]) # TODO: support multiple exemplars?
        
        # iterate though all frames
        for frame, gt_inst, mask in zip(data['imgs'], data['gt_instances'], data['masks']):
            output, loss, track_instances = process_frame_fn(exemplars, frame.unsqueeze(0), gt_inst, mask, track_instances)
            # outputs.append(output)
            del output
            losses.append(loss)
        
        return outputs, losses


    def forward_frame_train(self, exemplars:Tensor, frame:Tensor, gt_inst:Instances, mask:Tensor, track_instances:TrackInstances):
        
        ############## MODEL INFERENCE ###############
        # add noise to gt_queries (trains network to denoise GT)
        noised_gt_boxes = self.noise_gt(gt_inst.boxes)
        
        # extract multi scale features from exemplar [[B,C,h1,w1],[B,C,h2,w2],..]
        exe_features, exe_masks = self.backbone(*self.prebk(exemplars))
        
        # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
        frame, mask2 = self.prebk(frame)
        mask = mask2 if mask is None else mask[None] | mask2
        img_features, img_masks = self.backbone(frame, mask)
        
        # share information between exemplar & input frame
        # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
        img_features_mix, q_prop_refp, add_keys = self.mixer(img_features, exe_features, exe_masks, {})
        
        # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
        # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
        # TODO: move into mixer more elegantly
        q_queries, q_ref, _, attn_mask = \
            self.update_track_instances(img_features_mix, q_prop_refp, track_instances, noised_gt_boxes)

        img_features_pos, q_queries, _ = self.posembed(img_features, q_queries, None, q_ref, None, track_instances.lives, track_instances._idxs[0])

        # TODO: change how ref_pts are updated
        hs, isobj, coord = self.decoder(img_features, add_keys, q_queries, q_ref, attn_mask, img_masks)
        coord = self._fix_pad(coord, mask)

        ##################   LOSS  ###################

        output = {
            'img_features': [i.cpu() for i in img_features],
            'img_features_mix':[i.cpu() for i in img_features_mix],
            'img_features_pos': [i.cpu() for i in img_features_pos],
            
            'input_hs': q_queries,
            'q_ref': q_ref,

            'output_hs': hs,
            'position': coord,
            'is_object': isobj,
            
            'n_prev': track_instances._idxs[0],
            'n_prop': track_instances._idxs[1],
        }

        debug, matching_gt, matching_obj = \
            self.criterion.matching(output, track_instances.get_prevnew_queries(), gt_inst, compute_losses=False) #in learning.py

        output['debug']        = debug.cpu()                # to draw colored predictions
        output['matching_gt']  = matching_gt          # for losses
        output['matching_obj'] = matching_obj         # for losses

        losses = self.criterion([output],[gt_inst])

        track_instances = track_instances.get_tracks_next_frame(output['position'], output['output_hs'])

        return output, losses, track_instances


    def _forward_frame_train_ckpt(self, exemplars, frame, gt_inst:Instances, mask:Tensor, track_instances:TrackInstances):
        """
        Wrapper to checkpoint function, which allows to use less GPU_RAM 
        """

        gt_inst_args = [gt_inst.boxes, gt_inst.obj_ids, gt_inst.labels]
        tr_inst_args = [track_instances._idxs[0], track_instances._idxs[1], *[v for v in track_instances._fields.values()]]
        tr_keys = list(track_instances._fields.keys())
        args = [exemplars, frame, mask, *gt_inst_args, *tr_inst_args]
        params = tuple((p for p in self.parameters() if p.requires_grad))

        tmp = []
        def ckpt_fn(exemplars, frame, mask, gt_box, gt_obj_ids, gt_labels, tr_id1, tr_id2, *tr_inst_args):
            assert len(tr_keys)==len(tr_inst_args)

            # re_build ground truth and track instances
            gt_instances = Instances((1,1), boxes=gt_box, obj_ids=gt_obj_ids, labels=gt_labels)
            track_instances = TrackInstances(vars(self.args), init=False, _idxs=[tr_id1,tr_id2])
            for k,v in zip(tr_keys, tr_inst_args):
                track_instances._fields[k] = v

            # forward
            output, losses, track_instances = self.forward_frame_train(exemplars, frame, gt_instances, mask, track_instances)
            tmp.append((output, losses, track_instances))

            # return tuple of tensors we are interested in calling backward on
            return decompose_output([losses, track_instances.q_ref, track_instances.q_emb])
        
        CheckpointFunction.apply(ckpt_fn, len(args), *args, *params)
        return tmp[0]

    def _fix_pad(self, coord, mask):
        if mask.int().sum() == 0: return coord # no padding done
        _,h,w = mask.shape
        nh,nw = h-mask[0,:,w//2].int().sum(), w-mask[0,h//2].int().sum()
        padt, padl = mask[0,:h//2, w//2].int().sum(), mask[0,h//2,:w//2].int().sum()
        coord_xy = coord[...,:2] - torch.tensor([padl/w, padt/h], device=coord.device)
        coord = torch.cat((coord_xy, coord[...,2:]), dim=-1)
        coord = coord * torch.tensor([w/nw, h/nh, w/nw, h/nh], device=coord.device)
        return coord

    def update_track_instances(self, img_features, q_prop_refp, track_instances:TrackInstances, noised_gt:Tensor):
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
        track_instances.add_new(q_gt_emb, q_gt_refp, is_gt=True)

        # final queries for the decoder
        q_queries = track_instances.q_emb
        q_ref_pts = track_instances.q_ref

        n_tot = q_queries.shape[-2]
        attn_mask = torch.zeros((n_tot, n_tot), dtype=bool, device=q_queries.device)
        attn_mask[:n_gt, n_gt:] = True

        return q_queries.unsqueeze(0), q_ref_pts.unsqueeze(0), track_instances.lives, attn_mask

    def make_q_from_ref(self, ref_pts, img_features):
        queries = []
        for f_scale in img_features:
            _,_,h,w = f_scale.shape
            points = (ref_pts[:,:2] * torch.tensor([[w,h]],device=ref_pts.device)).long().view(-1, 2)
            q = f_scale[0, :, points[:,1], points[:,0]]
            queries.append(q.T)  # N, C
        queries = torch.stack(queries, dim=0).mean(dim=0)
        return queries

    def noise_gt(self, boxes):
        return boxes + torch.rand_like(boxes)*0.08 -0.04
