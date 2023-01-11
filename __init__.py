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

def test_dataset_gt():
    from datasets.fscd import build_fscd
    from configs.defaults import get_args_parser
    args = get_args_parser().parse_args()
    from tqdm import tqdm

    dataset = build_fscd('train', args)
    print('> testing', len(dataset), 'fscd dataset images')
    tot=0; size=0
    out = [torch.tensor([0.])]
    out2= [torch.tensor([1.])]
    for d in tqdm(dataset):
        for gt in d['gt_instances']:
            gt = gt.boxes
            tot += len(gt)
            size += (gt[:,2:]>=1).int().sum()+(gt[:,2:]<0).int().sum()

            gt = gt[:,:2]
            out.append(gt[gt <0].view(-1))
            out2.append(gt[gt >=1].view(-1))
    out = torch.cat(out, dim=0)
    out2 = torch.cat(out2, dim=0)

    # 380749bbox,  5000 out,   _ bb_wrong
    print(f"""BOXES={tot}
        smaller 0 = {len(out)}     [mean={out.mean()}   std={out.std()}   min={out.topk(min(len(out),3), largest=False)[0]}]
        bigger 1 = {len(out2)}     [mean={out2.mean()}   std={out2.std()}   min={out2.topk(min(len(out2),3))[0]}]
        bb_size = {size}"""
    )



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
    args.position_embedding = 'gauss_cat'
    args.num_feature_levels = 3
    args.embedd_dim = 256

    embedder = GeneralPositionEmbedder(args)
    q1_ref = torch.tensor([[[.8,.2,0,0],[.0,.0,0,0],[.5,.5,0,0]]])

    mif = get_img_feats(args)
    mif += [torch.zeros_like(mif[0])]

    q1 = make_q_from_ref(q1_ref[0], mif)[None]
    mif,q1,_ = embedder(mif, q1, None, q1_ref, None, None)
    q1[0,2,:-32] = 0  ### if clean embedd max is always precise

    vis = Visualizer(args)
    import os; os.path.exists(args.output_dir+'/debug/similarity.jpg') and os.remove(args.output_dir+'/debug/similarity.jpg')
    vis.debug_q_similarity(q1, mif, q1_ref,1,'')

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

        test_dataset_gt()
        # test_simple_embedder()
