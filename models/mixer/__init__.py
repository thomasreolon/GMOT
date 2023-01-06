import torch
import torch.nn as nn

from .nothing import LearnedQueries

def get_main_mixer(name, num_queries, embedd_dim):
    """module that mix information from the frame and the exemplar, each module can implement its own method as long it returns a list of multiscale image features (keys for decoder); queries for the decoder; possible additional keys"""

    if name == 'motr':
        return LearnedQueries()
    if name == '__-':
        return None
    else:
        raise NotImplementedError()


class GeneralMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mixer = get_main_mixer(args.mix_arch, args.num_queries, args.embedd_dim)

        # tells other modules if the mixer is proposing queries (if not they will be learned)
        _, q_ref, _ = self.mixer([torch.rand(1,args.embedd_dim,256,256)], [torch.rand(1,args.embedd_dim,64,64)], torch.zeros(1,256,256), {})
        self.ref_is_none = q_ref is None

    def forward(self, multiscale_img_feats, multiscale_exe_features, exe_masks, dict_outputs):
        multiscale_img_feats, q_ref_pts, other_keys = \
            self.mixer(multiscale_img_feats, multiscale_exe_features, exe_masks, dict_outputs)
    
        return multiscale_img_feats, q_ref_pts, other_keys
