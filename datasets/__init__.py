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

from .fscd import build_fscd
from .gmot import build as build_gmot


def build_dataset(image_set, args):
    # TODO: get more than 1 dataset at the time
    args.dataset_file=args.dataset_file[0]

    if args.dataset_file == 'e2e_fscd':
        return build_fscd(image_set, args)
    if args.dataset_file == 'e2e_gmot':
        return build_gmot(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
