import torch
from ._loss_utils import multiplier_decoder_level, matching_preds_gt


def lossfn_fake(output, target, outputs, targets, i):
    """focal loss, probability there is an object"""
    prob = output['is_object']

    return ((prob.sigmoid()-0)**2).mean()




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
