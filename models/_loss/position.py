import torch
from ._loss_utils import matching_preds_gt, multiplier_decoder_level
from util.misc import box_cxcywh_to_xyxy

def lossfn_position(output, target, outputs, targets, i):
    """L1 loss between predicted and real"""
    n_prop = len(output['matching_gt'])
    nois_gt = output['position'][:,:,n_prop:]

    pred, sorted_target = matching_preds_gt(output['matching_gt'], output['position'], target)

    loss = position(pred, sorted_target.boxes)      # loss
    if nois_gt.numel() > 0:
        loss = loss + position(nois_gt, target.boxes) # "teaching" loss

    return multiplier_decoder_level(loss).mean()



def position(pred_box, tgt_box):
    """in_shapes pred[6,1,N,4] tgt[N,4]"""
    
    return (pred_box-tgt_box[None,None]).abs()
