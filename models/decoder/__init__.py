import torch.nn as nn

from .def_transformer import OriginalDefDetr

def make_decoder(name, args):
    
    
    return OriginalDefDetr(args)



class GeneralDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder = make_decoder(None, args)

    def forward(self, *a, **b):
        return self.decoder(*a, **b)
