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

class smartdict(dict):
    def __init__(self, required:set, detach=False, *a, **kw):
        self._required = required
        self._detach = detach
        super().__init__(*a, **kw)
    def update(self, new_dict:dict):
        if self._detach: new_dict = {k:(v.cpu().detach() if isinstance(v,Tensor) else v) for k,v in new_dict.items()}
        super().update({k:v for k,v in new_dict.items() if k in self._required})

def get_info(model):
    import gc, json
    for name, param in model.named_parameters():
        if param.requires_grad:
            print( name, param.data.view(-1)[[0,-1]])
    objs = {'cpu':0, 'params':0}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                if str(obj.device) == 'cpu':
                    objs['cpu'] += 1
                else:
                    if isinstance(obj, torch.nn.parameter.Parameter):
                        objs['params'] += 1
                    elif str(obj.shape) not in objs: objs[str(obj.shape)] = 1
                    else: objs[str(obj.shape)] +=1
        except:
            pass
    print(json.dumps(objs, indent=1))
    import GPUtil; GPUtil.showUtilization()
    input()