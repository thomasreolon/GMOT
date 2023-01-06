# ------------------------------------------------------------------------
# Copyright (c) 2022 RIKEN. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


from typing import  List
import datetime
import time

import torch
from torch import Tensor 

def mot_collate_fn(batch: List[dict]) -> dict:
    ret_dict = {}
    for key in list(batch[0].keys()):
        assert not isinstance(batch[0][key], Tensor)
        ret_dict[key] = [img_info[key] for img_info in batch]
        if len(ret_dict[key]) == 1:
            ret_dict[key] = ret_dict[key][0]
    return ret_dict

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def print_time_from(train_start, epoch_start):
    total_time = time.time() - train_start
    epoch_time = time.time() - epoch_start
    print('>> Epoch took {}. Total time {}'.format(
        str(datetime.timedelta(seconds=int(epoch_time))),
        str(datetime.timedelta(seconds=int(total_time)))
    ))

def set_seed(seed):
    import numpy, random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)



# def _max_by_axis(the_list):
#     # type: (List[List[int]]) -> List[int]
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes


# def nested_tensor_from_tensor_list(tensor_list: List[Tensor], size_divisibility: int = 0):
#     # TODO make this more general
#     if tensor_list[0].ndim == 3:
#         # TODO make it support different-sized images

#         max_size = _max_by_axis([list(img.shape) for img in tensor_list])
#         if size_divisibility > 0:
#             stride = size_divisibility
#             # the last two dims are H,W, both subject to divisibility requirement
#             max_size[-1] = (max_size[-1] + (stride - 1)) // stride * stride
#             max_size[-2] = (max_size[-2] + (stride - 1)) // stride * stride

#         # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
#         batch_shape = [len(tensor_list)] + max_size
#         b, c, h, w = batch_shape
#         dtype = tensor_list[0].dtype
#         device = tensor_list[0].device
#         tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
#         mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
#         for img, pad_img, m in zip(tensor_list, tensor, mask):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
#             m[: img.shape[1], :img.shape[2]] = False
#     else:
#         raise ValueError('not supported')
#     return NestedTensor(tensor, mask)
