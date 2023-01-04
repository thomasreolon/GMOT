import torch



class ConcatPos(torch.nn.Module):

    def __init__(self, embedd_size, pos_embedd_size=32):
        super().__init__()
        self.linear = torch.nn.Linear(embedd_size, embedd_size-pos_embedd_size)
    
    def forward(self, x, pos):
        x = x.linear(x)
        return torch.cat((x,pos), dim=-1)



class SumPos(torch.nn.Module):
    def __init__(self, embedd_size, pos_embedd_size=32):
        super().__init__()
    
    def forward(self, x, pos):
        return x if pos is None else x+pos



