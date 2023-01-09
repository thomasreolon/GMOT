import time
import os

import torch

from configs.defaults import get_args_parser
import util.misc as utils
import util.multiprocess as distrib
from util.engine import train_one_epoch

from datasets import build_dataset
from models.gmot import build as build_model
from models.learning import build_learner



def main(args):
    # Initialize Run
    distrib.init_distributed_mode(args)                 # multi GPU
    utils.set_seed(args.seed + distrib.get_rank())      # reproducibility
    device = torch.device(args.device)

    # Build Model
    model      = build_model(args)                      # transformer tracker
    criterion, optimizer, lr_scheduler \
               = build_learner(args, model)             # loss function: models_outputs, ground_truth --> scalar
    tr_dataset = build_dataset('train', args)                    # dict(imgs, gt_instances, exemplar) 

    # Distribute Model & Dataset (if args.distributed==True)
    sampler, data_loader_train, model_without_ddp, model = distrib.make_distributed(args, model, tr_dataset)

    # Load Weights
    utils.load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler)

    # Training Loop
    start_time = time.time()
    debug_epochs = {args.start_epoch, (args.epochs+args.start_epoch)//2, args.epochs-1} # when save info images
    for epoch in range(args.start_epoch, args.epochs):
        start_e_time = time.time()
        if args.distributed:
            sampler.set_epoch(epoch)

        debug = None
        if args.debug and epoch in debug_epochs:
            debug = f'epoch_{epoch}/img{distrib.get_rank()}_'

        # Train
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args, debug)
        lr_scheduler.step()
        tr_dataset.step_epoch()

        # Save
        if args.output_dir:
            checkpoint_path = args.output_dir + '/checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        utils.print_time_from(start_time, start_e_time)


if __name__=='__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir+'/debug', exist_ok=True)
        if args.debug: os.system(f'rm -r "{args.output_dir}/debug/*"')
    main(args)