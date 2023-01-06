###########################
# tests code's components #
###########################
import numpy as np
import cv2
import torch


if __name__=='__main__':
    import torch

    from configs.defaults import get_args_parser
    from models.prebackbone import GeneralPreBackbone
    from models.backbone import GeneralBackbone
    from models.mixer import GeneralMixer
    from models.positionembedd import GeneralPositionEmbedder
    from models.decoder import GeneralDecoder

    print('> Testing Modules...')
    args = get_args_parser().parse_args()
    pre = GeneralPreBackbone(args)
    bk = GeneralBackbone(args, pre.ch_out)
    mx = GeneralMixer(args)
    emb = GeneralPositionEmbedder(args)
    dec = GeneralDecoder(args)

    print('>> modules built')

    img = torch.rand(1,3,512,728)
    exe = torch.rand(1,3,32,64)
    print('- start shape', img.shape)
    out = pre(img)
    print('- after prebk', out.shape)
    out = bk(out)
    print('- after bk', [o.shape for o in out])
    exe = bk(pre(exe))
    out,q1,q1_ref,q2,q2_ref = mx(out, exe)
    print('- after mixer',len(out), q1.shape, q2 and q2.shape)
    out, q1, q2 = emb(out,q1,q2, q1_ref,q2_ref)
    print('- after embedd',out[0].shape, q1.shape)
    hs, refs = dec(out, q2, q1, q1_ref, None)
    print('- after decoder', hs.shape)


    ### remember that:     if solution = 0.6,0.8 ---> real: 0.6 /orig_size *(origsize+PREpad)

    print()

    # print('>> testing gaussin embedding')
    # from models.positionembedd.gaussian_embedd import GaussianEmbedder as Embedder
    # from models.positionembedd.sine_embedd import SinCosEmbedder as Embedder
    # cl = Embedder(64)

    # fmap=cl.get_fmap_pos(torch.rand(1,2,100,100))[0].permute(1,2,0)
    # queries = cl.get_q_pos(torch.tensor([[0.9,0.95], [0.7,0.9]]), (100,100))  # w,h

    # img = fmap @ queries.sum(0).view(-1,1)
    # img = (img-img.min()) / (img.max()-img.min())
    # cv2.imshow('similarity of queries wrt image', img.numpy())
    # cv2.waitKey()

