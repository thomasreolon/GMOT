import torch
import torchvision.transforms.functional as T


class MinSize(torch.nn.Module):

    def __init__(self, min_size=16):
        """Does nothing for preprocessing. Only checks if image is at least {minsizeXminsize}pixels (useful for exemplar)"""
        super().__init__()
        self.min_size=min_size
    
    def forward(self, x, mask=None):
        b, _, h, w = x.shape
        assert mask is None or mask.shape[1]==h and mask.shape[2]==w, "mask has a different shape from x"
        
        if mask is None:
            mask = torch.zeros((b,h,w), device=x.device)
        if min(w,h)<self.min_size:
            # will be true only for exemplar hopefully
            pad_w = max(0, self.min_size-w)
            pad_h = max(0, self.min_size-h)
            x = T.pad(x, (pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2))
            mask = T.pad(mask, (pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2), fill=1)
        return x, mask

