###########################
# tests code's components #
###########################
import numpy as np
import cv2
import torch



def test_model_forward():
    from configs.defaults import get_args_parser
    from models.gmot import build

    print('\n\n> Testing GMOT')
    args = get_args_parser().parse_args()
    model = build(args)

    model({'imgs':torch.rand(5,1,3,128,128), 'exemplar':torch.rand(1,3,56,56)})

def test_position_embedds():
    print('\n\n> testing pos embedding')
    from models.positionembedd.sine_embedd import SinCosEmbedder
    from models.positionembedd.gaussian_embedd import GaussianEmbedder

    queries=[[0.9,0.95], [0.5,0.2]]
    print('>> queries positions (w,h):', queries)

    for Embedder in [SinCosEmbedder, GaussianEmbedder]:
        cl = Embedder(64)
        name = str(Embedder).split('.')[-1].split('>')[0][:-1]

        fmap=cl.get_fmap_pos(torch.rand(1,2,100,100))[0].permute(1,2,0)
        queries = cl.get_q_pos(torch.tensor(queries), (100,100))  # w,h

        img = fmap @ queries.sum(0).view(-1,1)
        img = (img-img.min()) / (img.max()-img.min())
        cv2.imshow(f'[{name}] similarity of queries wrt image', img.numpy())
    cv2.waitKey()


def test_single_modules():
    from configs.defaults import get_args_parser
    from models.prebackbone import GeneralPreBackbone
    from models.backbone import GeneralBackbone
    from models.mixer import GeneralMixer
    from models.positionembedd import GeneralPositionEmbedder
    from models.decoder import GeneralDecoder

    print('\n\n> Testing Modules...')
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
    print('- after prebk', out[0].shape)
    out, _ = bk(*out)
    print('- after bk', [o.shape for o in out])
    exe, mk = bk(*pre(exe))
    print('- exe masks', [o.shape for o in mk])
    out,q1,q1_ref,q2,q2_ref = mx(out, exe)
    print('- after mixer',len(out), q1.shape, q2 and q2.shape)
    out, q1, q2 = emb(out,q1,q2, q1_ref,q2_ref)
    print('- after embedd',out[0].shape, q1.shape)
    hs, _ = dec(out, q2, q1, q1_ref, None)
    print('- after decoder', hs.shape)


if __name__=='__main__':
    # test_position_embedds()
    # test_single_modules()
    test_model_forward()
