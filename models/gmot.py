"""
GMOT-DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util.misc import CheckpointFunction

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate, get_rank,
#                        is_dist_avail_and_initialized, inverse_sigmoid, box_ops, checkpoint)

# from util.misc import Instances, Boxes, pairwise_iou, matched_boxlist_iou




class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False, query_denoise=0):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.query_denoise = query_denoise
        self.position = nn.Embedding(num_queries, 4)
        self.yolox_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, 4:], d_model) + self.yolox_embed.weight])
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, gtboxes=None):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if gtboxes is not None:
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embed, ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank, mem_bank_pad_mask=track_instances.mem_padding_mask, attn_mask=attn_mask)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = frame_res['pred_logits'][:, n_ins:]
            ps_boxes = frame_res['pred_boxes'][:, n_ins:]
            frame_res['hs'] = frame_res['hs'][:, :n_ins]
            frame_res['pred_logits'] = frame_res['pred_logits'][:, :n_ins]
            frame_res['pred_boxes'] = frame_res['pred_boxes'][:, :n_ins]
            ps_outputs = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]
            for aux_outputs in frame_res['aux_outputs']:
                ps_outputs.append({
                    'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                    'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                })
                aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]
            frame_res['ps_outputs'] = ps_outputs

        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        tmp = {}
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None, proposals=None, exemplar=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        data['proposals'] = torch.zeros(len(frames),0,5, device=frames[0].device)
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        for frame_index, (frame, gt, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])): 
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1

            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, gtboxes)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame, gtboxes] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'hs': tmp[2],
                    'aux_outputs': [{
                        'pred_logits': tmp[3+i],
                        'pred_boxes': tmp[3+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)
            frame_res = self._post_process_single_image(frame_res, track_instances, False) #is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

            if False:     # if true will show detections for each image (debugging)
                import cv2
                dt_instances = self.post_process(track_instances, data['imgs'][0].shape[-2:])

                keep = dt_instances.scores > .02
                keep &= dt_instances.obj_idxes >= 0
                dt_instances = dt_instances[keep]

                wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
                areas = wh[:, 0] * wh[:, 1]
                keep = areas > 100
                dt_instances = dt_instances[keep]

                if len(dt_instances)==0:
                    print('nothing found')
                else:
                    print('ok')
                    bbox_xyxy = dt_instances.boxes.tolist()
                    identities = dt_instances.obj_idxes.tolist()

                    img = data['imgs'][frame_index].clone().cpu().permute(1,2,0).numpy()[:,:,::-1]
                    for xyxy, track_id in zip(bbox_xyxy, identities):
                        if track_id < 0 or track_id is None:
                            continue
                        x1, y1, x2, y2 = [int(a) for a in xyxy]
                        color = tuple([(((5+track_id*3)*4909 % p)%256) /110 for p in (3001, 1109, 2027)])

                        tmp = img[ y1:y2, x1:x2].copy()
                        img[y1-3:y2+3, x1-3:x2+3] = color
                        img[y1:y2, x1:x2] = tmp
                    cv2.imshow('preds', img/4+.4)
                    cv2.waitKey()


        outputs['track_instances'] = track_instances
        if self.training:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


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
        self.prebk    = PreBackbone(args)
        self.backbone = Backbone(args)
        self.mixer    = Mixer(args)
        self.posembed = PositionEmbedd(args)
        self.decoder  = Decoder(args)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 5)  # [ is_obj, x, y, w, h ]
        )


    def forward(self, data):
        assert data['imgs'][0].shape[0] == 1, "only BatchSize=1 is supported now.."
        
        # initially no tracks
        outputs = []
        track_instances = None
        exemplars = data['exemplar']
        
        # noise the ground truth and add it as queries to help the network learn later
        if 'gt_instances' in data and data['gt_instances'] is not None:
            noised_gt = self.noise_gt(data['gt_instances'])

        for frame in data['imgs']:
            # checkpointed forward pass
            output, track_instances = self._forward_frame(frame, exemplars, noised_gt, track_instances)
            outputs.append(output)
            exemplars = exemplars ##### maybe update

        return outputs

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
            exe_features, exe_masks = self.backbone(self.prebk(exemplar, True))

            # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
            img_features, img_masks = self.backbone(self.prebk(frame))

            # share information between exemplar & input frame
            # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
            img_features, q_prop_refp = self.mixer(img_features, exe_features, dict_outputs, exe_masks)

            # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
            # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
            q_queries, q_refpoints, attn_mask = \
                self.update_track_instances(img_features, q_prop_refp, track_instances, noised_gt)

            img_features, q_queries = self.posembed(img_features, q_queries, q_prop_refp, noised_gt)
            dict_outputs['input_hs'] = q_queries

            # TODO: change how ref_pts are updated
            hs, _, _, _, _ = \
                self.decoder(img_features, img_masks, None, q_queries, ref_pts=q_refpoints, attn_mask=attn_mask)
            dict_outputs['output_hs'] = hs

            predictions = self.head(hs)
            dict_outputs['is_object'] = predictions[:,0]
            dict_outputs['position']  = predictions[:,1:5]

            return [v for _,v in dict_outputs.items()]

        CheckpointFunction.apply(checkpoint_fn, len(args), *args, *params)
        return dict_outputs, track_instances


