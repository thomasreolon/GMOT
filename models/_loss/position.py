import torch
from ._loss_utils import matching_preds_gt, multiplier_decoder_level
from util.misc import box_cxcywh_to_xyxy

def lossfn_position(track_instances, output, gt_instance):
    """L1 loss between predicted and real"""

    pred, sorted_target = matching_preds_gt(track_instances.gt_idx, output['boxes'], gt_instance)

    loss = position(pred, sorted_target.boxes)      # loss
    return multiplier_decoder_level(loss).mean()

lossfn_position.is_intra_loss = True
lossfn_position.required = ['boxes']



def position(pred_box, tgt_box):
    """in_shapes pred[6,1,N,4] tgt[N,4]"""
    if pred_box.numel()==0: return torch.zeros(*pred_box.shape[:-2],1,1,device=pred_box.device)
    return (pred_box-tgt_box[None,None]).abs().mean(-2)
