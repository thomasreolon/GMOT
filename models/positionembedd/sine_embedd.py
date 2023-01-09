import torch



class SinCosEmbedder(torch.nn.Module):

    def __init__(self, size, temperature=10000):
        super().__init__()
        self.size = size
        self.temperature = temperature


    def get_fmap_pos(self, feat_map):
        B,_,H,W = feat_map.shape
        size = self.size //2

        positions = torch.ones(B,H,W).bool()

        y_embed = positions.cumsum(1, dtype=torch.float32)
        x_embed = positions.cumsum(2, dtype=torch.float32)

        dim_t = torch.arange(size, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="trunc") / size)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        return pos

        # return B,C=size,H,W

    def get_q_pos(self, ref_pts, confidences, img_shape=(800,1200)):
        size = self.size //2
        
        dim_t = torch.arange(size, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t.div(2, rounding_mode="trunc") / size)

        y_embed = ref_pts[:,1]*img_shape[0] +1
        x_embed = ref_pts[:,0]*img_shape[1] +1
        N = ref_pts.shape[0]

        pos_x = x_embed.view(N, 1, 1, 1) / dim_t.view(1,1,1,-1)
        pos_y = y_embed.view(N, 1, 1, 1) / dim_t.view(1,1,1,-1)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        return pos




