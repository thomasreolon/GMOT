import torch



class ConcatPos(torch.nn.Module):

    def __init__(self, embedd_size, pos_embedd_size=32):
        super().__init__()
        self.size = pos_embedd_size
        self.linear = torch.nn.Conv2d(embedd_size, embedd_size-pos_embedd_size, 1, bias=False)

    def forward(self, x, pos, n_prev):
        if len(x.shape) != 2:
            # case  [B,size,H,W]
            x = self.linear(x)
            x = torch.cat((x,pos), dim=1)
        else:
            # case [N,size]
            W = self.linear.weight.flatten(1).T
            if n_prev>0:
                x2 = x[n_prev:] @ W
                x1 = x[:n_prev, :-self.size]
                x = torch.cat((x1, x2), dim=0)
            else:
                x = x @ W
            x = torch.cat((x,pos), dim=-1)
        return x


class SumPos():
    def __call__(self, x, pos, n_prev):
        if n_prev>0:
            return torch.cat((x[:n_prev], x[n_prev:]+pos[n_prev:]), dim=0)
        return x if pos is None else x+pos



