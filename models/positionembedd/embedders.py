import torch



class ConcatPos(torch.nn.Module):

    def __init__(self, embedd_size, pos_embedd_size=32):
        super().__init__()
        # self.linear = torch.nn.Linear(embedd_size, embedd_size-pos_embedd_size)
        self.linear = torch.nn.Conv2d(embedd_size, embedd_size-pos_embedd_size, 1, bias=False)

    def forward(self, x, pos):
        if len(x.shape) != 2:
            # case  [B,size,H,W]
            x = self.linear(x)
            x = torch.cat((x,pos), dim=1)
        else:
            # case [N,size]
            W = self.linear.weight.flatten(1).T
            x = x @ W
            x = torch.cat((x,pos), dim=-1)
        return x


class SumPos():
    def __call__(self, x, pos):
        return x if pos is None else x+pos



