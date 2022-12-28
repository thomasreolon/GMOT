import torch
from .padder import Padder
from .nothing import MinSize


def get_main_prebackbone(type):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if type == 'padder':
        return Padder()
    if type == 'nothing':
        return MinSize()
    else:
        raise NotImplementedError()


class GeneralPreBackbone(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.body = get_main_prebackbone(args.img_prep)

        # forward to undesrtand the shape of the output (done in YOLO too to compute stride)
        out = self.body(torch.rand(1,3,530,730))
        self.ch_out = int(out.shape[1])
        del out

    def forward(self, x):
        return self.body(x)
