import torch
from ._loss_utils import multiplier_decoder_level, matching_preds_gt


def lossfn_is_object(output, target, outputs, targets, i):
    """focal loss, probability there is an object"""
    n_prop = len(output['matching_gt'])
    prob = output['is_object'].sigmoid() # TODO: maybe other activation is better for gradient propagation (sigmoid + softmax over queries?)
    nois_gt = prob[:,:,n_prop:]

    pos, _ = matching_preds_gt(output['matching_gt'], prob)   # should predict isobj=1
    neg, _ = matching_preds_gt(output['matching_gt'], prob)   # should predict isobj=0
    loss = isobj(pos, True) + isobj(neg, False)      # loss
    if nois_gt.numel() > 0:
        loss = loss + isobj(nois_gt, True) # "teaching" loss should predict isobj=1

    return multiplier_decoder_level(loss).mean()

        
def isobj(pred, positive, alpha=0.3, gamma=2.0):
    if positive:
        res = alpha * ((1 - pred) ** gamma) * (-(pred + 1e-8).log())
    else:
        res = alpha * ((1 - pred) ** gamma) * (-(pred + 1e-8).log())
    return res

    
# def lossfn_is_object(output, target, outputs, targets, i):
#     """focal loss, probability there is an object"""
#     alpha, gamma = 0.3, 2.0

#     n_prop = output['n_prop']  # == len(output['matching_gt'])
#     is_object = output['is_object']
#     positive = torch.cat((output['matching_gt']>=0, torch.ones(n_prop-is_object.shape[2], device=is_object.device, dtype=bool)))  # assigned predictions cat ground truth

#     pos = is_object[positive]
#     pos = alpha * ((1 - pos) ** gamma) * (-(pos + 1e-8).log())

#     neg = is_object[~positive]
#     neg = alpha * ((1 - neg) ** gamma) * (-(neg + 1e-8).log())

#     loss = multiplier_decoder_level(pos + neg).mean()

#     return loss
