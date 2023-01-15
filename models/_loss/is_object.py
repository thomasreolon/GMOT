import torch
from ._loss_utils import multiplier_decoder_level, matching_preds_gt


def lossfn_is_object(track_instances, output, gt_instance):
    """focal loss, probability there is an object"""

    pos, _ = matching_preds_gt(track_instances.gt_idx, output['is_object'])   # should predict isobj=1
    neg, _ = matching_preds_gt(~track_instances.gt_idx, output['is_object'])   # should predict isobj=0

    loss = isobj(pos.sigmoid(), True) + isobj(neg.sigmoid(), False)      # loss

    # stronger gradient
    if pos.numel()>0 and neg.numel()>0:
        pos, neg = pos.mean(-2).unsqueeze(-2).exp(), neg.mean(-2).unsqueeze(-2).exp()
        tot = pos+neg
        pos, neg = pos/tot, neg/tot
        loss = loss + isobj(pos, True) + isobj(neg, False)

    return multiplier_decoder_level(loss).mean()

lossfn_is_object.is_intra_loss = True
lossfn_is_object.required = ['is_object']

def isobj(pred, positive, alpha=0.8, gamma=2.0):
    if pred.numel()==0: return torch.zeros(*pred.shape[:-2],1,1,device=pred.device)
    elif positive:
        res = alpha * ((1 - pred) ** gamma) * (-(pred + 1e-8).log())
    else:
        res = (1 - alpha) * (pred ** gamma) * (-(1 - pred + 1e-8).log())
    return res[...,0].mean(-1)  * 30



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
