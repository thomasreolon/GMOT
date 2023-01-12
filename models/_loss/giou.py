import torch
from ._loss_utils import matching_preds_gt, multiplier_decoder_level
from util.misc import box_cxcywh_to_xyxy


def lossfn_giou(output, target, outputs, targets, i):
    n_prop = len(output['matching_gt'])
    nois_gt = output['position'][:,:,n_prop:]

    pred, sorted_target = matching_preds_gt(output['matching_gt'], output['position'], target)

    loss = giou(pred, sorted_target.boxes)      # loss
    if nois_gt.numel() > 0:
        loss = loss + giou(nois_gt, target.boxes) # "teaching" loss

    return multiplier_decoder_level(loss).mean()



def giou(pred_box, tgt_box):
    if pred_box.numel()==0: return torch.zeros(*pred_box.shape[:-2],1,1,device=pred_box.device)
    pred_box = box_cxcywh_to_xyxy(pred_box) # 6,1,N,4
    tgt_box = box_cxcywh_to_xyxy(tgt_box)   # N,4

    # each box area
    area_pred = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1]) # 6,1,N
    area_tgt  = (tgt_box[..., 2] - tgt_box[..., 0]) * (tgt_box[..., 3] - tgt_box[..., 1])

    # internal points
    lt = torch.max(pred_box[..., :2], tgt_box[None, None, :, :2])  # 6,1,N,2
    rb = torch.min(pred_box[..., 2:], tgt_box[None, None, :, 2:])  # 6,1,N,2

    # intersection area
    wh = (rb - lt).clamp(min=0)  # [6,1,N,2]
    inter = wh[..., 0] * wh[..., 1]  # [6,1,N]
    union = area_pred + area_tgt - inter
    iou = inter / union

    # external points
    lt = torch.min(pred_box[..., :2], tgt_box[:, :2])
    rb = torch.max(pred_box[..., 2:], tgt_box[:, 2:])

    # whole area
    wh = (rb - lt) # [6,1,N,2]
    area = wh[..., 0] * wh[..., 1]

    return 1-(iou - (area - union) / area).mean(-1) # 6,1
