import torch.nn as nn

from .def_transformer import OriginalDefDetr

def get_main_decoder(name, args):
    
    if name == 'base':
        return OriginalDefDetr(args)
    else:
        raise NotImplementedError()


class GeneralDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder = get_main_decoder(args.dec_name, args)

    def forward(self, *a, **b):
        return self.decoder(*a, **b)
