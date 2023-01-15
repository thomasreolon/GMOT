from typing import List, Tuple
import torch
from torch import nn, Tensor

from .nothing import LearnedQueries
from .addkeys import AddKeys
from util.misc import TrackInstances, inverse_sigmoid

def get_main_mixer(name, num_queries, embedd_dim):
    """module that mix information from the frame and the exemplar, each module can implement its own method as long it returns a list of multiscale image features (keys for decoder); queries for the decoder; possible additional keys"""

    if name == 'motr':
        return LearnedQueries()
    if name == 'addkeys':
        return AddKeys()
    else:
        raise NotImplementedError()


class GeneralMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mixer = get_main_mixer(args.mix_arch, args.num_queries, args.embedd_dim)

        # if the mixer is not proposing queries they will be learned
        with torch.no_grad():
            _, q_ref, _ = self.mixer([torch.rand(1,args.embedd_dim,256,256)], [torch.rand(1,args.embedd_dim,64,64)], torch.zeros(1,1,256,256), {})
        
        self.q_embed, self.ref_pts = None, None
        if q_ref is None:
            # learned queries when they are not provided by the mixer
            self.ref_pts = nn.Embedding(args.num_queries, 4)
        if args.use_learned:
            self.q_embed = nn.Embedding(args.num_queries, args.embedd_dim) #TODO: if args.learned
            self.q_embed_gt = nn.Embedding(args.num_queries, args.embedd_dim) #TODO: if args.learned
    
    @torch.no_grad()
    def _init_weights(self):
        if self.ref_pts is not None:
            self.ref_pts.weight.copy_(inverse_sigmoid(1e-2+torch.rand_like(self.ref_pts.weight)*0.98))
        if self.q_embed is not None:
            torch.nn.init.xavier_normal_(self.q_embed.weight)

    def forward(
        self, 
        multiscale_img_feats:List[Tensor], 
        multiscale_exe_features:List[Tensor],
        exe_masks:List[Tensor], 
        track_instances:TrackInstances, 
        noised_gt_boxes:Tensor, 
        dict_outputs:dict
    ) -> Tuple[List[Tensor], TrackInstances, Tensor, Tensor]:

        # uses img_features and exemplar to get initial queries positions
        multiscale_img_feats, q_ref_pts, add_keys = \
            self.mixer(multiscale_img_feats, multiscale_exe_features, exe_masks, dict_outputs)

        # make tracking queries [proposed+previous+GT] or [learned+previous+GT]
        # make input tensors for decorer:   eg.   q_embedd = cat(track.embedd, gt)
        attn_mask = self.update_track_instances(multiscale_img_feats, q_ref_pts, track_instances, noised_gt_boxes)

        return multiscale_img_feats, add_keys, attn_mask


    def update_track_instances(self, img_features:List[Tensor], q_prop_refp:Tensor, track_instances:TrackInstances, noised_gt:Tensor):
        dev = track_instances.q_emb.device
        # queries to detect new tracks
        if q_prop_refp is None:
            q_prop_refp = self.ref_pts.weight.sigmoid()
        
        if self.args.use_learned: # learned embeddings (MOTR LIKE)
            q_prop_emb  = self.q_embed.weight
        else:
            q_prop_emb  = self.make_q_from_ref(q_prop_refp, img_features)

        # queries used to help learning
        if noised_gt is None:
            q_gt_refp = torch.zeros((0, 4), device=dev)
            q_gt_emb  = torch.zeros((0, q_prop_emb.shape[1]),device=dev)
        else:
            q_gt_refp = noised_gt.clamp(0,0.9998)
            if self.args.use_learned:
                q_gt_emb  = torch.zeros((0, q_prop_emb.shape[1]),device=dev)
                l = len(noised_gt)
                while l>0:
                    q_gt_emb  = torch.cat((q_gt_emb,self.q_embed_gt.weight[:l]),dim=0)
                    l -=   q_gt_emb.shape[0]
            else:
                q_gt_emb  = self.make_q_from_ref(q_gt_refp, img_features)

        # add queries to detect new tracks to track_instances
        track_instances.add_new(q_prop_emb, q_prop_refp)
        track_instances.add_new(q_gt_emb, q_gt_refp, is_gt=True)

        # attn_mask
        n_pre, n_q, n_tot = *track_instances._idxs, len(track_instances)
        attn_mask = torch.zeros((n_tot, n_tot), dtype=bool, device=dev)
        attn_mask[:n_q, n_q:] = True     # proposed cannot see ground truth
        attn_mask[n_q:, :n_pre] = True     # ground truth cannot see queries we know are good

        return attn_mask


    def make_q_from_ref(self, ref_pts, img_features):
        queries = []
        for f_scale in img_features:
            _,_,h,w = f_scale.shape
            points = (ref_pts[:,:2] * torch.tensor([[w,h]],device=ref_pts.device)).long().view(-1, 2)
            q = f_scale[0, :, points[:,1], points[:,0]]
            queries.append(q.T)  # N, C
        queries = torch.stack(queries, dim=0).mean(dim=0)
        return queries

