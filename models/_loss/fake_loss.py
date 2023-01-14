import torch
from ._loss_utils import multiplier_decoder_level, matching_preds_gt


def lossfn_fake(track_instances, output, gt_instance):
    """focal loss, probability there is an object"""
    prob = output['is_object']
    return ((prob.sigmoid()-0)**2).mean()

lossfn_fake.is_intra_loss = True
lossfn_fake.required = ['is_object']