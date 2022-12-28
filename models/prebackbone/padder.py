import torch
import torchvision.transforms.functional as T


class Padder(torch.nn.Module):

    def __init__(self, padding=32):
        """Assures that the image will be disible by {padding}"""
        super().__init__()
        self.padding=padding
    
    def forward(self, x):
        _, _, h, w = x.shape
        pad = self.padding
        new_h = (h+pad-1) //pad *pad
        new_w = (w+pad-1) //pad *pad
        x = T.pad(x, (0,0,new_w-w,new_h-h),padding_mode="reflect")
        return x
