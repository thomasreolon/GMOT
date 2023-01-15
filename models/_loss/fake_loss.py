import torch
from ._loss_utils import multiplier_decoder_level, matching_preds_gt

from .is_object import lossfn_is_object
from .giou import lossfn_giou
from .position import lossfn_position

gt_box = torch.rand(10, 4)

def lossfn_fake(track_instances, output, gt_instance):
    """ fixed predictions : test memory / learning capabilities """
    ngt = min(10, len(gt_instance.boxes))
    track_instances.gt_idx = torch.cat((torch.arange(ngt), -torch.ones(len(track_instances.gt_idx)-ngt))).long().to(track_instances.gt_idx.device)
    gt_instance.boxes = torch.cat( (gt_box.to(gt_instance.boxes.device)[:ngt],gt_instance.boxes[ngt:]) )

    l1 = lossfn_is_object(track_instances, output, gt_instance)
    l2 = lossfn_giou(track_instances, output, gt_instance)
    l3 = lossfn_position(track_instances, output, gt_instance)

    return l1+l2+l3




lossfn_fake.is_intra_loss = True
lossfn_fake.required = ['is_object', 'boxes']