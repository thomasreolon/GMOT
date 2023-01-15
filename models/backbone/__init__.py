import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet
from .revbifpn import Revbifpn


def get_main_backbone(name, n_levels, ch_in):
    "select the type of backbone for feature multiple extraction from an image"

    # TODO: in general backbone, if ch_in!=3: substitute the first conv (or in the specific net code)
    assert ch_in==3, 'for now it is mandatory to have 3 input channels'

    if name == 'resnet50':
        return ResNet('resnet50', True, True, False)
    if name == 'revbifpn':
        return Revbifpn(n_levels)
    else:
        raise NotImplementedError()


class GeneralBackbone(nn.Module):
    def __init__(self, args, ch_in):
        super().__init__()
        self.args = args
        self.body = get_main_backbone(args.backbone, args.num_feature_levels, ch_in)

        # forward to undesrtand the shape of the output (done in YOLO too, to compute stride)
        out = self.body(torch.rand(1,3,256,256))
        ch_out = [int(o.shape[1]) for o in out]
        del out

        # creates more downsampling layers if num_feature_levels>len(ch_out)
        self.downsampling = nn.ModuleList()
        for _ in range(args.num_feature_levels-len(ch_out)):
            new_ch = ch_out[-1]   # +16
            h_dim = (new_ch*2+7) //8 *8
            self.downsampling.append(nn.Sequential(
                nn.BatchNorm2d(ch_out[-1]),
                nn.Conv2d(ch_out[-1], h_dim, kernel_size=1),                                              # linear proj of featues
                nn.GELU(),
                nn.Conv2d(h_dim, new_ch, kernel_size=3, stride=2, dilation=3, padding=3, groups=new_ch),  # longer distance dependencies
                nn.GELU(),
            ))   # maybe other types of block would be smarter 
            ch_out.append(new_ch)
        
        
        # gets to embed_dim number of channels
        self.linear_proj = nn.ModuleList()
        for ch in ch_out:
            n_groups = max([n-(ch%n)*9999 for n in (32, 16, 8,7,5,4,3,2,1)])
            self.linear_proj.append(nn.Sequential(
                nn.GroupNorm(n_groups, ch),
                nn.Conv2d(ch, args.embedd_dim, kernel_size=1),
                nn.GELU(),
            ))

    def forward(self, x, mask, outputs_dict=None):
        # get featuremaps from core backbone
        feature_maps = list(self.body(x))

        # add other f_maps if not enough
        for layer in self.downsampling:
            y = layer(feature_maps[-1])
            feature_maps.append(y)

        # standardize num_out_ch to embedd_dim
        for i in range(len(feature_maps)):
            feature_maps[i] = self.linear_proj[i]( feature_maps[i] )
        
        # select only num_feature_levels
        feature_maps = feature_maps[-self.args.num_feature_levels:]

        # make masks of multi sizes
        masks = []
        for f in feature_maps:
            tmp = F.interpolate(mask.unsqueeze(1), size=f.shape[-2:])
            masks.append(tmp.to(torch.bool).squeeze(1))

        return feature_maps, masks
