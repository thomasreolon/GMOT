import torch.nn as nn


class LearnedQueries():
    """
    This is the approach for MOT where you do not need an exemplar  
    (if you add a TransformerEncoder to process multiscale_img_feats is like MOTR)
    """
    def __call__(self, multiscale_img_feats, multiscale_exe_feats, multiscale_exe_masks, dict_outputs):
        #      new img features,       queries to use in decoder:None=learned,     keys to concat to decoder
        return multiscale_img_feats,            None,                              None


