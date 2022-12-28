###########################
# tests code's components #
###########################
import numpy as np
import cv2
import math
import torch



# #### get pos 1
# def sine1(H=2, W=2, num_feat=8, smooth=50):

#     feat_range = torch.arange(num_feat).view(1,1,num_feat).expand(H,W,-1).clone() / num_feat +0.5/num_feat
#     feat_w = torch.arange(W).view(1,W,1).expand(H,-1,num_feat).clone() / H +0.5/H
#     feat_h = torch.arange(H).view(H,1,1).expand(-1,W,num_feat).clone() / W +0.5/W
    
#     w_similarity = 0.01 + torch.exp(-(feat_range-feat_w)**2 *smooth)
#     h_similarity = 0.01 + torch.exp(-(feat_range-feat_h)**2 *smooth)

#     features = torch.cat((w_similarity*1,  h_similarity*1), dim=2)

#     print(features.max(), features.std(), features.mean(), features.min() )

#     return features.numpy()
#     # a = torch.arange(num_feat).view(1,-1) / num_feat
#     # h = torch.arange(H).view(-1,1) / H
#     # w = torch.arange(W).view(-1,1) / W
#     # h = torch.exp(- ((a-h) **2))
#     # w = torch.exp(- ((a-w) **2))



#     # return emb.numpy()

# def sineold(img_size=500, num_feat=64):
#     mask = torch.zeros(1,img_size,img_size).bool()
#     not_mask = ~mask

#     y_embed = not_mask.cumsum(1, dtype=torch.float32)
#     x_embed = not_mask.cumsum(2, dtype=torch.float32)

#     dim_t = torch.arange(num_feat, dtype=torch.float32)
#     dim_t = 10000 ** (2 * (dim_t // 2) / num_feat)

#     pos_x = x_embed[:, :, :, None] / dim_t
#     pos_y = y_embed[:, :, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#     pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#     pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#     return pos[0].permute(1,2,0).numpy()


# images = [
#     ('old', sineold(4,8)),
#     # ('s1', sine1(500,500,16)),
#     # ('s2', sine1(66,66,64)),
#     # ('s1_5', sine1(speed=5)),
#     # ('s1_9', sine1(strength=.4)),
#     # ('s1_9', sine1(speed=5, strength=.4)),
# ]

# for name, img in images:

#     h,w,_ = img.shape
#     tri_right = (img[h//2,w//2] + img[h*2//3,w*2//3] + img[h//3,w*2//3]) / 3

#     tri_left = (img[h//4,w//4] + img[h*3//4,w//4])
#     topright = (img[0,0])
#     center = img[h//2,w//2]
#     toprow = (img[h*2//3,w//4]+ img[h//4,w*3//4])

#     for R, tit in zip([center, tri_right, tri_left, topright, toprow], ('cent', 'right', 'left', 'topleft', 'diag')):
#         tmp=(img[:,:,None,:] @ R.reshape(1,1,-1,1))[:,:,0]
#         # tmp = np.exp(tmp)
#         tmp = (tmp-tmp.min()) / (tmp.max()-tmp.min())

#         cv2.imshow(name+tit, tmp)
# cv2.waitKey()

if __name__=='__main__':
    import torch

    from configs.defaults import get_args_parser
    from models.prebackbone import GeneralPreBackbone
    from models.backbone import GeneralBackbone
    from models.mixer import GeneralMixer

    print('> Testing Modules...')
    args = get_args_parser().parse_args()
    pre = GeneralPreBackbone(args)
    bk = GeneralBackbone(args, pre.ch_out)
    mx = GeneralMixer(args)

    print('>> modules built')

    img = torch.rand(1,3,512,728)
    exe = torch.rand(1,3,32,64)
    print('- start shape', img.shape)
    out = pre(img)
    print('- after prebk', out.shape)
    out = bk(out)
    print('- after bk', [o.shape for o in out])
    exe = bk(pre(exe))
    i,q1,q2 = mx(out, exe)
    print('- after mixer',len(i), q1[0].shape, q2 and q2[0].shape)

    ### remember that:     if solution = 0.6,0.8 ---> real: 0.6 /orig_size *(origsize+PREpad)

    print()

    print('>> testing gaussin embedding')
    from models.positionembedd.gaussian_embedd import GaussianEmbedder
    cl = GaussianEmbedder(32)

    fmap=cl.get_fmap_pos(torch.rand(1,2,100,100))[0].permute(1,2,0)
    queries = cl.get_q_pos(torch.tensor([[0.9,0.99],[0.2,0.8]]))

    img = fmap @ queries.sum(0).view(-1,1)
    img = (img-img.min()) / (img.max()-img.min())
    cv2.imshow('similarity of queries wrt image', img.numpy())
    cv2.waitKey()

