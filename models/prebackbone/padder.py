import torch
import torchvision.transforms.functional as T


class Padder(torch.nn.Module):

    def __init__(self, padding=32):
        """Assures that the image will be disible by {padding}"""
        super().__init__()
        self.padding=padding
    
    def forward(self, x, mask=None):
        b, _, h, w = x.shape
        assert mask is None or mask.shape[1]==h and mask.shape[2]==w, "mask has a different shape from x"
        
        if mask is None:
            mask = torch.zeros((b,h,w), device=x.device)

        pad = self.padding
        new_h = (h+pad-1) //pad *pad
        new_w = (w+pad-1) //pad *pad
        x = T.pad(x, (0,0,new_w-w,new_h-h),padding_mode="reflect")
        mask = T.pad(mask, (0,0,new_w-w,new_h-h), fill=1)
        return x, mask
