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
                        
        refs = None # fn(init, inter_ref)
        return hs, refs
