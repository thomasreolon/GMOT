import torch.nn as nn

from .nothing import LearnedQueries

def get_main_mixer(name, num_queries, embedd_dim):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if name == 'motr':
        return LearnedQueries(num_queries, embedd_dim)
    if name == '__-':
        return None
    else:
        raise NotImplementedError()


class GeneralMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mixer = get_main_mixer(args.mix_arch, args.num_queries, args.embedd_dim)

    def forward(self, multiscale_img_feats, multiscale_exe_features):
        return self.mixer(multiscale_img_feats, multiscale_exe_features)
