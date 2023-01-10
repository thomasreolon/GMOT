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

    outs = model({'imgs':torch.rand(5,1,3,128,128), 'exemplar':torch.rand(1,3,56,56)})
    print([(k,v.shape) for k,v in outs[0].items()])

def test_position_embedds():
    print('\n\n> testing pos embedding')
    from models.positionembedd.sine_embedd import SinCosEmbedder
    from models.positionembedd.gaussian_embedd import GaussianEmbedder

    ref_q=torch.tensor([[0.9,0.95], [0.5,0.2]])
    print('>> queries positions (w,h):', ref_q)

    for Embedder in [SinCosEmbedder, GaussianEmbedder]:
        for sizes in [(100,100), (20,36), (80,96)]:
            cl = Embedder(64)
            name = str(Embedder).split('.')[-1].split('>')[0][:-1]

            fmap=cl.get_fmap_pos(torch.rand(1,2,*sizes))[0].permute(1,2,0)
            queries = cl.get_q_pos(ref_q)

            img = fmap @ queries.sum(0).view(-1,1)
            img = (img-img.min()) / (img.max()-img.min())
            cv2.imshow(f'[{name}-{sizes}] similarity of queries wrt image', img.numpy())
    cv2.waitKey()

def test_full_embedder():
    from configs.defaults import get_args_parser
    from models.prebackbone import GeneralPreBackbone
    from models.backbone import GeneralBackbone
    from models.mixer import GeneralMixer
    from models.positionembedd import GeneralPositionEmbedder
    from models.decoder import GeneralDecoder

    args = get_args_parser().parse_args()
    args.position_embedding = 'gauss_sum'
    pre = GeneralPreBackbone(args)
    bk = GeneralBackbone(args, pre.ch_out)
    mx = GeneralMixer(args)
    emb = GeneralPositionEmbedder(args)


    img = torch.rand(1,3,512,728)
    exe = torch.rand(1,3,32,64)

    out, mask = bk(*pre(img))
    exe, mk = bk(*pre(exe))
    out,q1_ref,q2 = mx(out, exe, mk, {})
    if q1_ref is None: q1_ref = torch.tensor([[[.8,.2,.1,.1]]])
    q1 = make_q_from_ref(q1_ref, out)

    # compute pos embedd
    out, q1, _ = emb(out,q1,None, q1_ref,None,None)
    sim = (q1 @ out[0][0].flatten(1)).view(out[0].shape[2], out[0].shape[3]).sigmoid()
    sim = (sim-sim.min()) / (sim.max()-sim.min()) 
    cv2.imshow(f'(1)sim to: {q1_ref[0,0,:2].tolist()}', sim.numpy())


    # copy pos embedd
    f_scale = out[0]
    _,_,h,w = f_scale.shape
    points = (q1_ref[0,:,:2] * torch.tensor([[w,h]])).long().view(-1, 2)
    q1 = f_scale[0, :, points[:,1], points[:,0]].T.unsqueeze(0)


    sim = (q1 @ out[0][0].flatten(1)).view(out[0].shape[2], out[0].shape[3]).sigmoid()
    sim = (sim-sim.min()) / (sim.max()-sim.min()) 
    cv2.imshow(f'(2)sim to: {q1_ref[0,0,:2].tolist()}', sim.numpy())
    cv2.waitKey()

def make_q_from_ref(ref_pts, img_features):
    queries = []
    for f_scale in img_features:
        _,_,h,w = f_scale.shape
        points = (ref_pts[:,:2] * torch.tensor([[w,h]])).long().view(-1, 2)
        q = f_scale[0, :, points[:,1], points[:,0]]
        queries.append(q.T)  # N, C
    queries = torch.stack(queries, dim=0).mean(dim=0)
    return queries

def test_simple_embedder():
    from configs.defaults import get_args_parser
    from models.positionembedd import GeneralPositionEmbedder
    from util.misc import Visualizer
    args = get_args_parser().parse_args()
    args.position_embedding = 'sin_cat'
    args.num_feature_levels = 3

    embedder = GeneralPositionEmbedder(args)
    q1_ref = torch.tensor([[[.8,.2,0,0],[.0,.0,0,0],[.5,.5,0,0]]])

    mif = get_img_feats(args)
    mif += [torch.zeros(1,256,30,45)]

    q1 = make_q_from_ref(q1_ref[0], mif)[None]
    mif,q1,_ = embedder(mif, q1, None, q1_ref, None, None)

    vis = Visualizer(args)
    import os; os.path.exists(args.output_dir+'/debug/similarity.jpg') and os.remove(args.output_dir+'/debug/similarity.jpg')
    vis.debug_q_similarity(q1, mif, q1_ref,1,'')

    # sim = (q1[:,:1] @ mif[0][0].flatten(1)).view(mif[0].shape[2], mif[0].shape[3])
    # sim = sim.view(-1).softmax(dim=0).view(sim.shape[0],-1)

    # sim = (sim-sim.min()) / (sim.max()-sim.min()) 
    # cv2.imshow(f'(A)sim to: {q1_ref[0,0,:2].tolist()}', sim.numpy())
    # cv2.waitKey()

def get_img_feats(args):
    from models.prebackbone import GeneralPreBackbone
    from models.backbone import GeneralBackbone

    pre = GeneralPreBackbone(args)
    bk = GeneralBackbone(args, pre.ch_out)

    img = torch.rand(1,3,800,800)
    out = pre(img)
    out, mask = bk(*out)
    return out

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
    out, mask = bk(*out)
    print('- after bk', [o.shape for o in out])
    exe, mk = bk(*pre(exe))
    print('- exe masks', [o.shape for o in mk])
    out,q1_ref,q2 = mx(out, exe, mk, {})
    if q1_ref is None: q1_ref = torch.rand(1,20,4).sigmoid()
    q1 = make_q_from_ref(q1_ref[0], out)
    print('- after mixer: q1, q_ref', q1_ref.shape, q1.shape)

    out, q1, q2 = emb(out,q1,None, q1_ref,None,torch.tensor([0 for i in range(q1_ref.shape[1])]))
    print('- after embedd',out[0].shape, q1.shape)
    hs, _, _ = dec(out, q2, q1, q1_ref, None, mask)
    print('- after decoder', hs.shape)


if __name__=='__main__':
    with torch.no_grad():
        # test_single_modules()
        # test_model_forward()
        # test_position_embedds()

        # test_full_embedder()
        test_simple_embedder()
