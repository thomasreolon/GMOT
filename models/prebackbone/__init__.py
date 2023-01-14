import torch
from .padder import Padder
from .nothing import MinSize


def get_main_prebackbone(name):
    "This is the type of preprocessing done before the real backbone, can: resize image, add padding to make img divisible by 32, extract some channels in advance, ..."

    if name == 'padding':
        return Padder()
    if name == 'nothing':
        return MinSize()
    else:
        raise NotImplementedError()


class GeneralPreBackbone(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.body = get_main_prebackbone(args.img_prep)

        # forward to undesrtand the shape of the output (done in YOLO too to compute stride)
        out, _ = self.body(torch.rand(1,3,530,730))
        self.ch_out = int(out.shape[1])
        del out

    def forward(self, x):
        return self.body(x)

    def _fix_pad(self, coord, mask):
        """the prediction are relative to the padded input tensor, this accounts for the translation wrt the original image"""
        if mask.int().sum() == 0: return coord # no padding done
        _,h,w = mask.shape
        nh,nw = h-mask[0,:,w//2].int().sum(), w-mask[0,h//2].int().sum()
        padt, padl = mask[0,:h//2, w//2].int().sum(), mask[0,h//2,:w//2].int().sum()
        coord_xy = coord[...,:2] - torch.tensor([padl/w, padt/h], device=coord.device)
        coord = torch.cat((coord_xy, coord[...,2:]), dim=-1)
        coord = coord * torch.tensor([w/nw, h/nh, w/nw, h/nh], device=coord.device)
        return coord