import torch

def multiplier_decoder_level(loss) -> torch.Tensor:
    """shape NL,1,NQ,X  --> 1,NQ,X"""
    NL,B,NQ,X = loss.shape
    scale = torch.arange(NL, dtype=loss.dtype, device=loss.device) / (NL*(NL+1)/2)
    loss = scale.view(1,-1) @ loss.view(NL,-1)
    return loss.view(B,NQ,X)

def matching_preds_gt(match, prediction, target=None):
    """select predictions matched with GT and the GT"""
    # TODO: implement cache? store in output the selection result?
    prediction = prediction[:,:,:len(match)]
    to_select = match>=0
    prediction = prediction[to_select]
    if target is not None:
        idxs = match[to_select]
        target = target[idxs]
    return prediction, target
