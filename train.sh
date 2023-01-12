#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


args=''
num_nodes=1
for var in "$@"
do
    # get additional arguments
    if [ -a $var ]
    then
        tmp=$(<"$var")
        args="${args} ${tmp}"
    fi

    # if number set it as number of GPUs
    if [[ $var =~ ^[0-9]+$ ]]
    then
        num_nodes=$var
    fi
done

# run code
if [ $num_nodes -gt "1" ]
then
    python3 -m torch.distributed.launch --nproc_per_node=$num_nodes --use_env main.py $args
else
    CUDA_LAUNCH_BLOCKING=1 python3 main.py ${args}
fi

