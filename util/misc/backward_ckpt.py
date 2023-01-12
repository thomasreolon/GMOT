import torch

def decompose_output(element):
    # return all the tensor in the dict
    # require_true will be set to each tensor, in this way backward will reach the Checkpoint

    if isinstance(element, torch.Tensor):
        return [element]
    elif isinstance(element, list):
        return sum((decompose_output(e) for e in element), [])
    elif isinstance(element, dict):
        return sum((decompose_output(e) for e in element.values()), [])
    else:
        return []

class CheckpointFunction(torch.autograd.Function):
    """Cool Function to compute gradient at steps by re-forward-passing during the backward"""
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            # forward without grad
            output_tensors = ctx.run_function(*ctx.input_tensors)
        for out in output_tensors:
            out.requires_grad = True
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            # detach gradient from input tensors
            if check_require_grad(temp):
                ctx.input_tensors[i] = temp.detach()
                ctx.input_tensors[i].requires_grad = temp.requires_grad
        to_autograd = list(filter(check_require_grad, ctx.input_tensors))
        with torch.enable_grad():
            # forward with grad
            output_tensors = ctx.run_function(*ctx.input_tensors)
        output_tensors, output_grads = zip(*filter(lambda t: t[0].requires_grad, zip(output_tensors, output_grads)))
        # gradientof_input_and_params = grad(output, start_grad_jacobian)
        input_grads = torch.autograd.grad(output_tensors, to_autograd + ctx.input_params, output_grads, allow_unused=True)
        input_grads = list(input_grads)
        for i in range(len(ctx.input_tensors)):
            if not check_require_grad(ctx.input_tensors[i]):
                input_grads.insert(i, None)
        return (None, None) + tuple(input_grads)


def check_require_grad(t):
    return isinstance(t, torch.Tensor) and t.requires_grad

