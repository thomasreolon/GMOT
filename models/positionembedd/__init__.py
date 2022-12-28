import torch.nn as nn


def get_position_type(name, size):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if name == 'sin':
        return None
    else:
        raise NotImplementedError()

def get_embedder(name, size):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if name == 'concat':
        return None
    if name == 'sum':
        return None
    else:
        raise NotImplementedError()



class GeneralMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.embedder = get_position_type(args.mix_arch, args.num_queries, args.embedd_dim)

    def forward(self, multiscale_img_feats, q1, q2):

        pass
