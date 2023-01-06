import torch.nn as nn
from .gaussian_embedd import GaussianEmbedder
from .sine_embedd import SinCosEmbedder
from .embedders import ConcatPos, SumPos

def get_position_type(name, size):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if 'sin' in name:
        return SinCosEmbedder(size)
    elif 'gauss' in name:
        return GaussianEmbedder(size)
    elif 'sinv2' in name:
        return None
    else:
        raise NotImplementedError()

def get_fuser(name, emb_all, emb_pos):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if 'cat' in name:
        return ConcatPos(emb_all, emb_pos)
    if 'sum' in name:
        return SumPos()
    else:
        raise NotImplementedError()


class GeneralPositionEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        pos_embedd_size = args.embedd_dim if 'sum' in args.position_embedding else args.embedd_dim//16 *2
        # option to add x,y directly in concat
        # option to remove embedd from query
        ## ? cat_sin_sinv2 ?

        self.embedder = get_position_type(args.position_embedding, pos_embedd_size)
        self.fuser = get_fuser(args.position_embedding, args.embedd_dim, pos_embedd_size)

    def forward(self, multiscale_img_feats, q1, q2, q1_ref, q2_ref):
        
        q1 = self.make_queries(q1, q1_ref)
        q2 = self.make_queries(q2, q2_ref)
        multiscale_img_feats = self.make_features(multiscale_img_feats)

        return multiscale_img_feats, q1, q2

    def make_queries(self, q, q_ref):
        if q is None: return None

        q_pos = self.embedder.get_q_pos(q_ref)
        return self.fuser(q, q_pos)

    def make_features(self, msf):
        poss = [self.embedder.get_fmap_pos(img) for img in msf]
        msf = [self.fuser(f, f_pos) for f, f_pos in zip(msf, poss)]
        return msf
        
