import torch
import os

def load_checkpoint(args, model, optimizer=None, lr_scheduler=None):
    # load only weights
    if args.pretrained is not None and args.resume is None:
        load_model(model, args.pretrained)

    # special case: load last checkpoint
    if args.resume=='__last__':
        tmp = args.output_dir + '/checkpoint.pth'
        args.resume = tmp if os.path.exists(tmp) else None

    # load weights and more
    args.start_epoch = 0
    if args.resume is not None:
        print(f'resuming ...')
        checkpoint = load_model(model, args.resume)

        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and \
        optimizer is not None    and lr_scheduler is not None:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step()
            args.start_epoch = checkpoint['epoch'] + 1


def load_model(model, model_path):
    print(f'loading pretrained {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                if 'class_embed' in k:
                    print("load class_embed: {} shape={}".format(k, state_dict[k].shape))
                    if model_state_dict[k].shape[0] == 1:
                        state_dict[k] = state_dict[k][1:2]
                    elif model_state_dict[k].shape[0] == 2:
                        state_dict[k] = state_dict[k][1:3]
                    elif model_state_dict[k].shape[0] == 3:
                        state_dict[k] = state_dict[k][1:4]
                    else:
                        raise NotImplementedError('invalid shape: {}'.format(model_state_dict[k].shape))
                    continue
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return checkpoint

def load_for_eval(args):
    ARCHITECTURE = ['img_prep', 'backbone', 'num_feature_levels', 'mix_arch', 'use_learned', 'position_embedding', 'num_queries', 'dec_name', 'dec_layers', 'dec_n_points', 'dim_feedforward', 'nheads', 'embedd_dim']
    model_path = args.resume or args.pretrained or args.output_dir + '/checkpoint.pth'

    print("loading... ", model_path)
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'args' in checkpoint:
        old_args = checkpoint['args']
        for k in ARCHITECTURE:
            args.__dict__[k] = old_args.__getattribute__(k)

    from models.gmot import build as build_model
    model = build_model(args)
    load_model(model, model_path)
    return model.eval()