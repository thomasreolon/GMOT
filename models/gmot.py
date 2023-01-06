"""
GMOT-DETR model and criterion classes.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import CheckpointFunction

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate, get_rank,
#                        is_dist_avail_and_initialized, inverse_sigmoid, box_ops, checkpoint)

# from util.misc import Instances, Boxes, pairwise_iou, matched_boxlist_iou

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
        noised_gt = None
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
            exe_features, exe_masks = self.backbone(*self.prebk(exemplar))

            # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
            img_features, img_masks = self.backbone(*self.prebk(frame))

            # share information between exemplar & input frame
            # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
            img_features, q_prop_refp, add_keys = self.mixer(img_features, exe_features, exe_masks, dict_outputs)

            # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
            # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
            q_queries, q_ref, attn_mask = \
                self.update_track_instances(img_features, q_prop_refp, track_instances, noised_gt)

            img_features, q_queries, _ = self.posembed(img_features, q_queries, None, q_ref, None)
            dict_outputs['input_hs'] = q_queries

            # TODO: change how ref_pts are updated
            hs, refs = self.decoder(img_features, add_keys, q_queries, q_ref, attn_mask)
            dict_outputs['output_hs'] = hs

            predictions = self.head(hs)
            dict_outputs['is_object'] = predictions[:,0]
            dict_outputs['position']  = predictions[:,1:5]

            return [v for _,v in dict_outputs.items()]

        CheckpointFunction.apply(checkpoint_fn, len(args), *args, *params)
        return dict_outputs, track_instances

    def update_track_instances(self, img_features, q_prop_refp, track_instances, noised_gt):
        # TODO: to support multi batch size ->
        ##          add padding queries
        ##          set attn_mask = true for padding queries

        # queries to detect new tracks
        b = img_features[0].shape[0]
        if q_prop_refp is None:
            q_prop_emb  = self.q_embed.weight.data.unsqueeze(0).expand(b,-1,-1)
            q_prop_refp = self.ref_pts.weight.data.unsqueeze(0).expand(b,-1,-1)
        else:
            q_prop_emb  = self.make_q_from_ref(q_prop_refp, img_features)

        # queries used to help learning
        if noised_gt is not None:
            q_gt_refp = noised_gt
            q_gt_emb  = self.make_q_from_ref(q_gt_refp, img_features)
            n_gt      = q_gt_refp.shape[-2]
        else:
            q_gt_refp = torch.zeros((b, 0, 4))
            q_gt_emb  = torch.zeros((b, 0, q_prop_emb.shape[2]))
            n_gt      = 0

        
        # add queries to detect new tracks to track_instances
        # track_instances.add_new(q_prop_emb, q_prop_refp)

        # final queries for the decoder
        # q_queries = torch.cat((track_instances.q_emb, q_gt_emb), dim=-2)   # B,N,256
        # q_ref_pts = torch.cat((track_instances.q_ref, q_gt_refp), dim=-2)  # B,N,4
        q_queries = torch.cat((q_prop_emb, q_gt_emb), dim=-2)   # B,N,256 ##DEBUG
        q_ref_pts = torch.cat((q_prop_refp, q_gt_refp), dim=-2)  # B,N,4
        n_tot = q_queries.shape[-2]
        attn_mask = torch.zeros((n_tot, n_tot), dtype=bool, device=q_queries.device)
        attn_mask[:n_gt, n_gt:] = True

        return q_queries, q_ref_pts, attn_mask

    def make_q_from_ref(self, ref_pts, img_features):
        b_idxs = sum([[i]*ref_pts.shape[1] for i in range(ref_pts.shape[0])], [])
        queries = []
        for f_scale in img_features:
            b,c,h,w = f_scale.shape
            points = (ref_pts[:,:,:2] * torch.tensor([[[w,h]]])).int().view(-1, 2)
            q = img_features[b_idxs, :, points[:,1], points[:,0]]
            queries.append(q.view(b,ref_pts.shape[1], c))
        queries = torch.stack(queries, dim=0).mean(dim=0)
        return queries



