import numpy as np
import torch.nn as nn
import torch

class AddKeys():
    """
    This is the approach for MOT where you do not need an exemplar  
    (if you add a TransformerEncoder to process multiscale_img_feats is like MOTR)
    """
    def __call__(self, multiscale_img_feats, multiscale_exe_feats, multiscale_exe_masks, dict_outputs):
        #      new img features,       queries to use in decoder:None=learned,     keys to concat to decoder

        add_keys = self.get_keys(multiscale_exe_feats, multiscale_exe_masks)
        return multiscale_img_feats,            None,                              add_keys

    def get_keys(self, esrcs, masks):
        """the number of queries is always squarable (1,4,9,16)"""
        # set exemplar as first pixel
        queries = []
        m = masks[0][0]
        rw = 1-(m[m.shape[0]//2].sum() / m.shape[1]).item()
        rh = 1-(m[:,m.shape[1]//2].sum() / m.shape[0]).item()
        for lvl, src in enumerate(esrcs):
            _,_,H,W = src.shape

            # always get central pixel     # +1 
            queries.append( src[:,:,H//2,W//2] )
            # always get global pooling     # +1 
            queries.append( src.mean(dim=(2,3)) )

            # pixels in a oval
            num_pix = 2*(len(esrcs)-lvl-1) # 4, 2, 0
            if num_pix:
                for h,w in self.get_points_hw((rh*H/4, rw*W/4), (H//2,W//2), num_pix):
                    queries.append( src[:,:,h,w] )
        return torch.cat(queries, dim=0)[None]  # 1,N,C

    def get_points_hw(self, r_hw, center, n_points):
        points = []
        step = 6.28318 / n_points
        for i in range(n_points):
            p = [center[0]-np.cos(i*step)*r_hw[0], center[1]-np.sin(i*step)*r_hw[1]]
            points.append(np.round(p).astype(int).tolist())
        return points