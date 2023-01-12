import torch.nn as nn
from .gaussian_embedd import GaussianEmbedder
from .sine_embedd import SinCosEmbedder
from .embedders import ConcatPos, SumPos

def get_position_type(name, size, keep_for):
    "type of position embedding to be used embedding (sine, others)"

    if 'sin' in name:
        return SinCosEmbedder(size)
    elif 'gauss' in name:
        return GaussianEmbedder(size, keep_for)
    elif 'sinv2' in name:
        return None
    else:
        raise NotImplementedError()

def get_fuser(name, emb_all, emb_pos):
    "how to use position ambeddings (sum/concat)"

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
        # TODO: option to add x,y directly in concat
        # option to remove embedd from query
        ## ? cat_sin_sinv2 ?

        self.embedder = get_position_type(args.position_embedding, pos_embedd_size, args.keep_for)
        self.fuser = get_fuser(args.position_embedding, args.embedd_dim, pos_embedd_size)

    def forward(self, multiscale_img_feats, q1, q2, q1_ref, q2_ref, confidences, n_prev):
        
        img_shape=multiscale_img_feats[0].shape[2:]
        q1 = self.make_queries(q1, q1_ref, confidences, img_shape, n_prev)
        q2 = self.make_queries(q2, q2_ref, None, None, None)
        multiscale_img_feats = self.make_features(multiscale_img_feats)

        return multiscale_img_feats, q1, q2

    def make_queries(self, q, q_ref, confidences,img_shape, n_prev):
        if q is None: return None

        b,n,c = q.shape
        q_pos = self.embedder.get_q_pos(q_ref.view(b*n, 4), confidences=confidences, img_shape=img_shape)
        q = self.fuser(q.view(b*n, -1), q_pos, n_prev)
        return q.view(b,n,c)

    def make_features(self, msf):
        poss = [self.embedder.get_fmap_pos(img) for img in msf]
        msf = [self.fuser(f, f_pos,0) for f, f_pos in zip(msf, poss)]
        return msf
        
