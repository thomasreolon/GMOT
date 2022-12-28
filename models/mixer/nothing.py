import torch.nn as nn



class LearnedQueries(nn.Module):
    """
    This is the approach for MOT where you do not need an exemplar  
    (if you add a TransformerEncoder to process multiscale_img_feats is like MOTR)
    """
    def __init__(self, num_queries, embedd_dim):
        super().__init__()
        self.ref_pts = nn.Embedding(num_queries, 4)
        self.q_embed = nn.Embedding(num_queries, embedd_dim)

    
    def forward(self, multiscale_img_feats, multiscale_exe_feats):
        learned_queries = (self.ref_pts.weight.data, self.q_embed.weight.data)

        # new img features,  queries to use in decoder,   keys to concat to decoder
        return multiscale_img_feats,     learned_queries,            None


