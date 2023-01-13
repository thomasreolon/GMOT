import torch

def decompose_output(element, first=True):
    # return all the tensor in the dict
    # require_true will be set to each tensor, in this way backward will reach the Checkpoint

    if isinstance(element, torch.Tensor):
        res = [element]
    elif isinstance(element, list):
        res = sum((decompose_output(e,False) for e in element), [])
    elif isinstance(element, dict):
        res = sum((decompose_output(e,False) for e in element.values()), [])
    else:
        res = []
    
    return tuple(res) if first else res

class CheckpointFunction(torch.autograd.Function):
    """Cool Function to compute gradient at steps by re-forward-passing during the backward"""
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.asd = torch.rand(1).item()
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            # forward without grad
            output_tensors = ctx.run_function(*ctx.input_tensors)
        print('HI-------------------',ctx.asd)
        print('-', output_tensors)
        print('------------------------------')
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
        
        print('BK-------------------',len(output_tensors),ctx.asd)
        print(output_tensors)
        print(output_grads)

        output_tensors, output_grads = zip(*filter(lambda t: t[0].requires_grad, zip(output_tensors, output_grads)))
        print(len(output_tensors))
        print('------------------------------')
        # gradientof_input_and_params = grad(output, start_grad_jacobian)
        input_grads = torch.autograd.grad(output_tensors, to_autograd + ctx.input_params, output_grads, allow_unused=True)
        input_grads = list(input_grads)
        for i in range(len(ctx.input_tensors)):
            if not check_require_grad(ctx.input_tensors[i]):
                input_grads.insert(i, None)
        return (None, None) + tuple(input_grads)


def check_require_grad(t):
    return isinstance(t, torch.Tensor) and t.requires_grad


###########################################################
import torch.nn as nn
model = nn.Sequential(nn.Linear(4,2048),nn.ReLU(),nn.Linear(2048,8192),nn.ReLU(),nn.Linear(8192,4)).cuda().train()
input = torch.rand(3, 4).cuda()


outputs = []
tmp = input
for k in range(100):

    output = {}
    def ckpt_fn(tmp, i):
        out = model(tmp)
        output['out'] = out
        output['loss'] = - (out[...,0]-i)**2 
        output['i'] = i

        return decompose_output(output, True)

    CheckpointFunction.apply(ckpt_fn, 2, tmp, k, *list(model.parameters()))
    outputs.append(output)
    tmp = output['out']


loss = sum(( -v['loss'].sum() for v in outputs))

for o in outputs:
    out=o['out']
    loss = loss + ((out[...,1]-9)**2).mean()

loss.backward()
print(list(model.parameters())[0].grad.sum())
