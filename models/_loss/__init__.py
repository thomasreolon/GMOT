from .is_object import lossfn_is_object
from .giou import lossfn_giou
from .position import lossfn_position
from .fake_loss import lossfn_fake


def loss_fn_getter(name):
    if name=='is_object':
        return lossfn_is_object
    if name=='giou':
        return lossfn_giou
    if name=='fake':
        return lossfn_fake
    if name=='boxes':
        return lossfn_position
    else:
        raise NotImplementedError(name)

