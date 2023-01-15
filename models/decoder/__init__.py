import torch.nn as nn
import torch
from .def_transformer import OriginalDefDetr
from .addkeys import DefAddkeysTransformer

def get_main_decoder(name, args):
    
    if name == 'base':
        return OriginalDefDetr(args)
    if name == 'addkeys':
        return DefAddkeysTransformer(args)
    else:
        raise NotImplementedError()


class GeneralDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder = get_main_decoder(args.dec_name, args)

    def forward(self, img_features, add_keys, track_instances_pos, attn_mask, img_masks):
        q_emb = track_instances_pos.q_emb[None]
        q_ref = track_instances_pos.q_ref[None]
        return self.decoder(img_features, add_keys, q_emb, q_ref, attn_mask, img_masks)
