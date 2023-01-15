# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_, eye_
from util.misc import inverse_sigmoid

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        self.im2col_step = 64
        self.proj_min = 4
        self.sigmoid_attn = sigmoid_attn

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.ModuleList([
            nn.Linear(d_model,d_model, bias=False) for _ in range(n_heads * (n_levels+1))
        ])
        self.value_proj = nn.ModuleList([ nn.Linear(d_model, d_model) for _ in range(n_heads * 2) ])
        self.head_mixer = nn.Linear(self.n_heads+1, self.d_model, bias=False)

        self._reset_parameters()


    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, add_keys, input_level_start_index,input_padding_mask=None):
        """input_padding_mask
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :param add_keys                    (N, Length_{x}, C)

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        assert N==1, "thought for batch size=1 only"


        # value = self.value_proj(input_flatten)
        # if input_padding_mask is not None:
        #     value.masked_fill_(input_padding_mask[..., None], float(0))
        # value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / input_spatial_shapes[None, None, None, :, None, (1, 0)]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # attention
        keys = self.get_keys(sampling_locations, input_flatten, input_spatial_shapes, input_level_start_index).view(
            N*Len_q, self.n_heads, self.n_levels, self.n_points, self.d_model
        )
        keys = keys.permute(1,2,3,0,4).contiguous()
        head_w = self.head_mixer.weight.T.softmax(dim=0) # 3, 256
        result = query.view(-1, self.d_model) * head_w[None,-1]
        for h in range(self.n_heads): # (Q@W@K.T) (W@K).T
            # similarity queries/pixels
            attn_matrix = []
            for lvl in range(self.n_levels+1):
                simil = self.attention_weights[h*self.n_levels+lvl].weight
                if lvl != self.n_levels:
                    ki = simil @ keys[h, lvl].view(-1,self.d_model).T
                    ki = ki.view(self.n_points, N*Len_q, self.d_model)
                    attn = (ki * query.view(-1,self.d_model)[None]).sum(dim=-1) #n_points,N
                elif add_keys is not None:
                    ki = add_keys.view(-1,self.d_model) @ simil.T
                    attn = ki @ query.view(-1,self.d_model).T  # X,N
                attn_matrix.append(attn)
            attn_matrix = torch.cat(attn_matrix, dim=0).softmax(dim=0)  #lvls*n_points+X, N
            
            # output as linear proj of pixels
            values = self.value_proj[h*2]  (keys[h].view(-1,self.d_model))   # lvls*n_points, dim
            values = values.view(self.n_levels*self.n_points, N*Len_q, self.d_model)
            if add_keys is not None:
                v2 = self.value_proj[1+h*2](add_keys.view(-1, self.d_model)) # X, dim
                v2 = v2[:,None].expand(-1,N*Len_q,-1)
                values = torch.cat((values,v2), dim=0)                 # lvls*n_points+X, N, dim
            
            # values weighted to attn_matrix
            result = result + (attn_matrix[:,:,None] * values).sum(dim=0) * head_w[None, h]
        return result[None]

    def get_keys(self, sampling_locations, input_flatten, iss, input_level_start_index):
        keys = []
        # N   Len_q   n_heads   n_levels   n_points
        sampling_locations = torch.clamp(sampling_locations, min=0, max=0.999)

        idx = (sampling_locations*iss.view(1,1,1,-1,1,2)).int()     # h*H, w*W
        idx[:,:,:,:,:,1] *= iss[:,0].view(1,1,1,-1,1)               #int(h*H)*W
        idx = idx.sum(dim=-1) + input_level_start_index.view(1,1,1,-1,1)  # int(h*H)*W + w*W + index_lvl

        keys = input_flatten[0, idx.view(-1), :] # BS=1

        return keys

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        [eye_(w.weight.data) for w in self.attention_weights]
        [eye_(w.weight.data) for w in self.value_proj]
        constant_(self.head_mixer.weight.data, 0.)


class DefAddkeysTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.new_frame_adaptor = None
        self.d_model = args.embedd_dim
        self.nhead = args.nheads
        self.two_stage = False

        decoder_layer = DeformableTransformerDecoderLayer(args.embedd_dim, args.dim_feedforward,
                                                          args.dropout, 
                                                          'gelu',
                                                          args.num_feature_levels, 
                                                          args.nheads, 
                                                          args.dec_n_points, 
                                                          False,
                                                          sigmoid_attn=False, 
                                                          extra_track_attn=False,
                                                          memory_bank=False)
        self.decoder = DeformableTransformerDecoder(decoder_layer, args.dec_layers, True)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, add_keys, query_embed, ref_pts, attn_mask, masks):

        # prepare input for decoder
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            h, w = src.shape[-2:]
            spatial_shapes.append((w,h))
            src = src.flatten(2).transpose(1, 2)  # B, HW, C
            src_flatten.append(src)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        
        src_flatten = torch.cat(src_flatten, 1) #  B, HW+HW2... , C
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        
        # decoder
        hs, isobj, coord = self.decoder(query_embed, ref_pts, src_flatten,
                                            spatial_shapes, level_start_index,
                                            valid_ratios, mask_flatten,
                                            None, None, attn_mask, add_keys)

        return hs, isobj, coord




class ReLUDropout(torch.nn.Dropout):
    def forward(self, input):
        return relu_dropout(input, p=self.p, training=self.training, inplace=self.inplace)

def relu_dropout(x, p=0, inplace=False, training=False):
    if not training or p == 0:
        return x.clamp_(min=0) if inplace else x.clamp(min=0)

    mask = (x < 0) | (torch.rand_like(x) > 1 - p)
    return x.masked_fill_(mask, 0).div_(1 - p) if inplace else x.masked_fill(mask, 0).div(1 - p)

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

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_cross=True, sigmoid_attn=False,
                 extra_track_attn=False, memory_bank=False):
        super().__init__()

        self.self_cross = self_cross
        self.num_head = n_heads
        self.memory_bank = memory_bank

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, sigmoid_attn=sigmoid_attn)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # next ref & pro
        self.head = MLP(d_model*2, d_ffn, 5, 2)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout_relu = ReLUDropout(dropout, True)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout_relu(self.linear1(tgt)))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _forward_self_attn(self, tgt, query_pos, attn_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                                  attn_mask=attn_mask)[0].transpose(0, 1)
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)


    def _forward_self_cross(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, add_keys=None):

        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # cross attention

        tgt2 = self.cross_attn(tgt, reference_points,
                               src, src_spatial_shapes, add_keys, level_start_index, src_padding_mask)
        objcoord = self.head(torch.cat(tgt, tgt2), dim=-1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, objcoord

    def _forward_cross_self(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                            src_padding_mask=None, attn_mask=None, add_keys=None):
        # cross attention
        tgt2 = self.cross_attn(tgt, reference_points,
                               src, src_spatial_shapes, add_keys, level_start_index, src_padding_mask)

        objcoord = self.head(torch.cat((tgt, tgt2), dim=-1))
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # self attention
        tgt = self._forward_self_attn(tgt, query_pos, attn_mask)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, objcoord

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None, add_keys=None):
        if self.self_cross:
            return self._forward_self_cross(tgt, query_pos, reference_points, src, src_spatial_shapes,
                                            level_start_index, src_padding_mask, attn_mask, add_keys)
        return self._forward_cross_self(tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                                        src_padding_mask, attn_mask, add_keys)



class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, mem_bank=None, mem_bank_pad_mask=None, attn_mask=None, add_keys=None):
        output = tgt
        ref_pts = reference_points

        intermediate = []
        intermediate_reference_points = []
        intermediate_score = []
        for layer in self.layers:
            if ref_pts.shape[-1] == 4:
                reference_points_input = ref_pts[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert ref_pts.shape[-1] == 2
                reference_points_input = ref_pts[:, :, None] * src_valid_ratios[:, None]

            output, objcoord = layer(output, None, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, src_padding_mask, mem_bank, mem_bank_pad_mask, attn_mask, add_keys)
            is_obj = objcoord[...,:1]

            ref_pts = (inverse_sigmoid(ref_pts) + objcoord[...,1:]/10).sigmoid()


            intermediate.append(output)
            intermediate_score.append(is_obj)
            intermediate_reference_points.append(ref_pts)

        return torch.stack(intermediate), torch.stack(intermediate_score), torch.stack(intermediate_reference_points)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
