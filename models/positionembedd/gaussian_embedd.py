import torch



class GaussianEmbedder(torch.nn.Module):

    def __init__(self, size, keep_for=20, strength=50):
        super().__init__()
        self.size = size
        self.strength = strength
        self.keep_for = keep_for

    @torch.no_grad()
    def get_fmap_pos(self, feat_map):
        B,_,H,W = feat_map.shape
        size = self.size //2

        size_range = torch.arange(size).view(1,1,size).expand(H,W,-1).clone() / size +0.5/size
        h_embedd = torch.arange(H).view(H,1,1).expand(-1,W,size).clone() / H +0.5/H
        h_embedd = 0.01 + torch.exp(-(size_range-h_embedd)**2 *self.strength)
        
        w_embedd = torch.arange(W).view(1,W,1).expand(H,-1,size).clone() / W +0.5/W
        w_embedd = 0.01 + torch.exp(-(size_range-w_embedd)**2 *self.strength)

        embedd = torch.cat((w_embedd, h_embedd), dim=2).to(feat_map.device)
        return embedd.permute(2,0,1).unsqueeze(0).expand(B,-1,-1,-1)

    @torch.no_grad()
    def get_q_pos(self, ref_pts, confidences=None, img_shape=None):
        if confidences is None: confidences = torch.zeros(ref_pts.shape[0])
        size = self.size //2
        strength = self.strength*(1+0.5*confidences/self.keep_for)

        h_embedd = (torch.arange(size, device=ref_pts.device)/ size +0.5/size).view(1,-1)
        h_embedd = h_embedd - ref_pts[:,1].unsqueeze(1)
        h_embedd = 0.01 + torch.exp(-h_embedd**2 *strength[:,None])
        h_embedd = h_embedd/h_embedd.sum(dim=-1)[:,None]
        
        w_embedd = (torch.arange(size, device=ref_pts.device)/ size +0.5/size).view(1,-1) # 1,S
        w_embedd = w_embedd - ref_pts[:,0].unsqueeze(1)            # N,1
        w_embedd = 0.01 + torch.exp(-w_embedd**2 *strength[:,None])
        w_embedd = w_embedd/w_embedd.sum(dim=-1)[:,None]

        return torch.cat((w_embedd, h_embedd), dim=1).to(ref_pts.device)
