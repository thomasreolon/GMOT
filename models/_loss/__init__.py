from .is_object import lossfn_is_object
from .giou import lossfn_giou
from .position import lossfn_position
from .fake_loss import lossfn_fake


def loss_fn_getter(name):
    if name=='is_object':
        return Wrapper(lossfn_is_object, ['is_object', 'matching_gt'])
    if name=='giou':
        return Wrapper(lossfn_giou, ['position', 'matching_gt'])
    if name=='fake':
        return Wrapper(lossfn_fake, ['is_object'])
    if name=='position':
        return Wrapper(lossfn_position, ['position', 'matching_gt'])
    else:
        raise NotImplementedError(name)


def Wrapper(loss_fn, required=[]):
    """execute loss_fn for every output/target in the sequence, skips if not all required inputs in output_dict, get pre_computed value if already in output_dict"""
    def list_loss_fn(outputs, gt_instances):
        loss= 0
        if all(k in outputs[0] for k in required):
            for i, (output, target) in enumerate(zip(outputs, gt_instances)):
                loss = loss + loss_fn(output, target, outputs, gt_instances, i).view(1)
        return loss / len(outputs)
    return list_loss_fn


