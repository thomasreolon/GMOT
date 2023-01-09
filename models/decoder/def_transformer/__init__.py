import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_transformer_plus import DeformableTransformer
from util.misc import inverse_sigmoid

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRHead(nn.Module):
    def __init__(self, hidden_dim=256, dec_layers=6, with_box_refine=True) -> None:
        super().__init__()

        self.is_obj = nn.Linear(hidden_dim, 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if with_box_refine:
            self.is_obj = _get_clones(self.is_obj, dec_layers)
            self.bbox_embed = _get_clones(self.bbox_embed, dec_layers)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.is_obj = nn.ModuleList([self.is_obj for _ in range(dec_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(dec_layers)])
    
    def forward(self, hs, init_reference, inter_references):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.is_obj[lvl](hs[lvl])
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

        return outputs_class, outputs_coord



def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.embedd_dim,
        nhead=args.nheads,
        num_encoder_layers=6,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=args.num_queries,
        decoder_self_cross=True,
        sigmoid_attn=False,
        extra_track_attn=False,
        memory_bank=False
    )


class OriginalDefDetr(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.encoderdecoder = build_deforamble_transformer(args)
        self.head = DETRHead(args.embedd_dim, args.dec_layers)

    def forward(self, img_features, add_src, q_emb, q_ref, attn_mask, srcs_mask=None):
        assert add_src is None, "Original DefDetr do no support concat of keys"

        # attend to all input
        if srcs_mask is None:
            srcs_mask = [torch.zeros_like(sr) for sr in img_features]
            srcs_mask = [sr[:,0].bool() for sr in img_features]

        hs, init_reference, inter_references, _, _ = \
            self.encoderdecoder(
                srcs=img_features,
                masks=srcs_mask, 
                pos_embeds=None, 
                query_embed=q_emb, 
                ref_pts=q_ref, 
                attn_mask=attn_mask
            )
                        
        isobj, position = self.head(hs, init_reference, inter_references)
        return hs, isobj, position
