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
        learned_queries = self.q_embed.weight.data, self.ref_pts.weight.data

        #      new img features,       queries to use in decoder,          keys to concat to decoder
        return multiscale_img_feats,  learned_queries[0],learned_queries[1],    None,None



class LearnedQueries():
    """
    This is the approach for MOT where you do not need an exemplar  
    (if you add a TransformerEncoder to process multiscale_img_feats is like MOTR)
    """
    def __call__(self, multiscale_img_feats, multiscale_exe_feats, multiscale_exe_masks, dict_outputs):
        #      new img features,       queries to use in decoder:None=learned,     keys to concat to decoder
        return multiscale_img_feats,            None,                              None


