import torch

"""

def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb
"""

class SinCosEmbedder(torch.nn.Module):

    def __init__(self, size, temperature=10000, fixed_size=True):
        super().__init__()
        self.size = size
        self.temperature = temperature
        self.fixed_size = fixed_size

    def get_fmap_pos(self, feat_map):
        B,_,H,W = feat_map.shape
        size = self.size //2

        positions = torch.ones(B,H,W).bool()

        y_embed = positions.cumsum(1, dtype=torch.float32)
        x_embed = positions.cumsum(2, dtype=torch.float32)

        if self.fixed_size:
            y_embed = 200*(y_embed-0.5) / y_embed.sum()
            x_embed = 200*(x_embed-0.5) / x_embed.sum()

        dim_t = torch.arange(size, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="trunc") / size)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2).to(feat_map.device)
        return pos

    @torch.no_grad()
    def get_q_pos(self, ref_pts, confidences=None):
        if len(ref_pts.shape)==3: ref_pts=ref_pts[0]
        size = self.size //2

        conf = torch.stack((confidences, ((ref_pts-1)**2).sum(-1)), dim=-1) # needed to find a way to use confidence..

        dim_t = torch.arange(size, dtype=torch.float32, device=ref_pts.device)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="trunc") / size)

        y_embed = ref_pts[:,1]
        x_embed = ref_pts[:,0]
        if self.fixed_size:
            y_embed = 200 * y_embed
            x_embed = 200 * x_embed

        N = ref_pts.shape[0]
        pos_x = x_embed.view(N, 1) / dim_t.view(1,-1)
        pos_y = y_embed.view(N, 1) / dim_t.view(1,-1)
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x[:,:-1], pos_y[:,:-1], conf), dim=1)
        return pos




