import torch
from .deformable_transformer_plus import DeformableTransformer

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

    def forward(self, srcs, cat_keys, q, q_ref, attn_mask, srcs_mask=None):
        assert cat_keys is None, "Original DefDetr do no support concat of keys"

        # attend to all input
        if srcs_mask is None:
            srcs_mask = [torch.zeros_like(sr) for sr in srcs]
            srcs_mask = [sr[:,0].bool() for sr in srcs]

        hs, init_reference, inter_references, _, _ = \
            self.encoderdecoder(srcs, srcs_mask, None, q, ref_pts=q_ref,
                                mem_bank=None, mem_bank_pad_mask=None, attn_mask=attn_mask)
        refs = None # fn(init, inter_ref)
        return hs, refs
