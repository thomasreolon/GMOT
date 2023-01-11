from .is_object import lossfn_is_object


def loss_fn_getter(name):
    name = 'loss_'+name
    if name=='is_object':
        lossfn_is_object.name = name
        return Wrapper(lossfn_is_object, [])
    else:
        raise NotImplementedError()


def Wrapper(loss_fn, required=[]):
    """execute loss_fn for every output/target in the sequence, skips if not all required inputs in output_dict, get pre_computed value if already in output_dict"""
    def list_loss_fn(outputs, gt_instances):
        loss, name = 0, loss_fn.name
        if all(k in outputs[0] for k in required):
            for output, target in zip(outputs, gt_instances):
                if name not in output:
                    output[name] = loss_fn(output, target)
                loss = loss + output[name]
        return loss / len(outputs)
    return list_loss_fn


