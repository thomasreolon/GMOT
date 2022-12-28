import torch
import torchvision.transforms.functional as T


class MinSize(torch.nn.Module):

    def __init__(self, min_size=16):
        """Does nothing for preprocessing. Only checks if image is at least {minsizeXminsize}pixels (useful for exemplar)"""
        super().__init__()
        self.min_size=min_size
    
    def forward(self, x):
        _, _, h, w = x.shape
        if min(w,h)<self.min_size:
            pad_w = max(0, self.min_size-w)
            pad_h = max(0, self.min_size-h)
            x = T.pad(x, (pad_w//2, pad_h//2))
        return x

