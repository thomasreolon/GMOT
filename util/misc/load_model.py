import torch


def load_checkpoint(args, model, optimizer, lr_scheduler):
    # load only weights
    if args.pretrained is not None and args.resume is None:
        load_model(model, args.pretrained)

    # special case: load last checkpoint
    if args.resume=='':
        args.resume = args.output_dir + '/checkpoint.pth'

    # load weights and more
    args.start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.step_size = args.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(f'loaded {model_path}')
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

