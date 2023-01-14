"""
GMOT-DETR model and criterion classes.
"""
import gc
import torch
from torch import Tensor

from util.misc import CheckpointFunction, decompose_output, TrackInstances, Instances, smartdict, Visualizer

from .prebackbone import GeneralPreBackbone
from .backbone import GeneralBackbone
from .mixer import GeneralMixer
from .positionembedd import GeneralPositionEmbedder
from .decoder import GeneralDecoder
from .learning import Criterion

def build(args):
    model = GMOT(args)
    print("Model Num Parameters: {}".format(
        sum(p.numel() for p in model.parameters())
    ))
    return model


class GMOT(torch.nn.Module):
    """
    Core Module made of 5 components:
        prebackbone: pad images                     f(tensor.shape[1,3,H,W])                    -> tensor.shape[1,n,H,W]
        backbone: extracts features                 f(tensor.shape[1,n,H,W])                    -> [tensor.shape[1,C,h1,w1],tensor.shape[1,C,h2,w2],...]
        mixer: uses info from exemplar              f(img_features, exemplar_feats)              -> newimg_features, query_positions, additional_decoder_keys
        posembed: transformer positional embeddings f(newimg_features, queries, query_positions) -> newimg_features, queries
        decoder: extracts features                  f(newimg_features, queries, query_positions,additional_decoder_keys) -> predictions
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args   = args
        args.use_checkpointing = True  # runs out of memory otherwise

        # self.track_embed = track_embed        # used in post process eval
        self.prebk    = GeneralPreBackbone(args)
        self.backbone = GeneralBackbone(args, self.prebk.ch_out)
        self.mixer    = GeneralMixer(args)
        self.posembed = GeneralPositionEmbedder(args)
        self.decoder  = GeneralDecoder(args)

        self.criterion = Criterion(args, self)
        self.visualizer=Visualizer(args)


    def forward(self, data:dict, debug:str=None):
        """ (batch size forcefully == 1)

            Arguments:
                data['imgs']         = list_of_CHW_tensors  --> [torch.size(3,256,256), torch.size(3,256,256), ...]
                data['gt_instances'] = list_of_gt_instances  --> [inst1, inst2, ...]     , with i containing (boxes, obj_ids)
                data['exemplar']     = list_of_chw_tensors  --> [torch.size(3,64,64)]
            
            Returns:
                loss_dicts: List[Dict[str,Tensor]]    (losses are computed in the forward to maximize checkpointing usefulness)
        """
        if 'masks'        not in data: data['masks']        = [None for _ in range(len(data['imgs']))]

        process_frame_fn = self._forward_frame_train_ckpt if self.args.use_checkpointing else self.forward_frame_train

        # initializations
        self.debug = debug
        losses = []
        device = data['imgs'][0].device
        track_instances = TrackInstances(vars(self.args), init=True).to(device)
        exemplars = torch.stack(data['exemplar'][:1]) # TODO: support multiple exemplars?
        
        # iterate though all frames
        for i,(frame, gt_inst, mask) in enumerate(zip(data['imgs'], data['gt_instances'], data['masks'])):
            self.debug_i = i #TODO: prettier way
            loss_dict, track_instances = process_frame_fn(exemplars, frame.unsqueeze(0), gt_inst, mask, track_instances)
            losses.append(loss_dict)
            gc.collect(); torch.cuda.empty_cache()

        loss_dict = self.criterion.forward_inter_frame(losses)
        
        return loss_dict


    def forward_frame_train(self, exemplars:Tensor, frame:Tensor, gt_inst:Instances, mask:Tensor, track_instances:TrackInstances):
        output = smartdict(self.criterion.required) # stores only logits required by the losses (if they are not needed the reference is dropped and the can be freed from the memory)
        out_debug = smartdict((self.visualizer.required if self.args.debug else []), True)

        ############## MODEL INFERENCE ###############
        
        # extract multi scale features from exemplar [[B,C,h1,w1],[B,C,h2,w2],..]
        exe_features, exe_masks = self.backbone(*self.prebk(exemplars))
        
        # extract multi scale features from image [[B,C,H1,W1],[B,C,H2,W2],..]
        frame, mask2 = self.prebk(frame)
        mask = mask2 if mask is None else mask[None] | mask2
        img_features, img_masks = self.backbone(frame, mask)
        output.update({'img_features':img_features, 'exe_features':exe_features}) # TODO: move inside each component?
        out_debug.update({'frame':frame, 'gtboxes':gt_inst.boxes})
        
        # share information between exemplar & input frame
        # returns img_feat[[B,C,H1,W1],[B,C,H2,W2],..]   and    queries positions [[xywh],[xywh],[xywh],...]
        img_features, add_keys, attn_mask = self.mixer(img_features, exe_features, exe_masks, track_instances, self.noise_gt(gt_inst.boxes), output) # add noise to gt_queries (trains network to denoise GT)
        output.update({'img_features_mix':img_features, 'q_emb_mix':track_instances})

        # add positional embeddings to queries & keys
        img_features, track_instances = self.posembed(img_features, track_instances)
        output.update({'img_features_pos':img_features})
        out_debug.update({'img_features_pos':img_features})

        # transformer for tracking
        hs, isobj, coord = self.decoder(img_features, add_keys, track_instances, attn_mask, img_masks)
        coord = self.prebk._fix_pad(coord, mask)
        output.update({'output_hs':hs, 'boxes':coord, 'is_object':isobj})
        out_debug.update({'q_ref':track_instances.q_ref, 'boxes':coord, 'is_object':isobj, 'input_hs':track_instances.q_emb})

        ##################   LOSS  ###################

        # macthing & loss computation
        track_instances, loss_dict = self.criterion.forward_intra_frame(track_instances, output, gt_inst)
        out_debug.update({'matching_gt':track_instances.gt_idx})

        # makes visualizations
        if self.debug:
            self.visualizer.visualize_infographics(out_debug, path=self.debug+str(self.debug_i))

        track_instances = track_instances.get_tracks_next_frame(coord, hs, isobj)

        return loss_dict, track_instances

    def noise_gt(self, boxes):
        rand_coeff = max(0.05, (100-len(boxes)) / 100) * 0.06
        return boxes + torch.rand_like(boxes)*rand_coeff -rand_coeff/2

    def _forward_frame_train_ckpt(self, exemplars, frame, gt_inst:Instances, mask:Tensor, track_instances:TrackInstances):
        """
        Wrapper to checkpoint function, which allows to use less GPU_RAM 
        """

        # parameters needed in the function --> unrolled
        gt_inst_args = [gt_inst.boxes, gt_inst.obj_ids, gt_inst.labels]
        tr_inst_args = track_instances._fields.values()
        tr_keys = list(track_instances._fields.keys())
        args = [exemplars, frame, mask, *gt_inst_args, *tr_inst_args]
        params = tuple((p for p in self.parameters() if p.requires_grad))

        tmp = []
        def ckpt_fn(exemplars, frame, mask, gt_box, gt_obj_ids, gt_labels, *tr_inst_args):
            assert len(tr_keys)==len(tr_inst_args)

            # re_build ground truth and track instances
            gt_instances = Instances((1,1), boxes=gt_box, obj_ids=gt_obj_ids, labels=gt_labels)
            track_instances2 = TrackInstances(vars(self.args), init=False, _idxs=track_instances._idxs)
            for k,v in zip(tr_keys, tr_inst_args):
                track_instances2._fields[k] = v

            # forward
            loss_dict, track_instances2 = self.forward_frame_train(exemplars, frame, gt_instances, mask, track_instances2)
            tmp.append((loss_dict, track_instances2))

            # return tuple of tensors we are interested in calling backward on
            asd = decompose_output([loss_dict, track_instances2.q_ref, track_instances2.q_emb])
            return asd

        CheckpointFunction.apply(ckpt_fn, len(args), *args, *params)
        return tmp[0]






































    # def _forward_frame_train_ckpt(self, exemplars, frame, gt_inst:Instances, mask:Tensor, track_instances:TrackInstances):
    #     """ Step 1.
    #     Wrapper to checkpoint function, which allows to use less GPU_RAM 
    #     """

    #     # Unroll Tensors Used In Forward
    #     gt_inst_args = [gt_inst.boxes, gt_inst.obj_ids, gt_inst.labels]
    #     tr_inst_args = track_instances._fields.values()
    #     tr_keys = list(track_instances._fields.keys())
    #     args = [exemplars, frame, mask, *gt_inst_args, *tr_inst_args]
    #     params = tuple((p for p in self.parameters() if p.requires_grad))

    #     # Checkpointed Forward
    #     ckpt_fn = self._get_ckpt_fn(tr_keys, track_instances._idxs)
    #     res = CheckpointFunction.apply(ckpt_fn, len(args), *args, *params)
    #     assert len(self.criterion.l_names) + len(tr_keys)== len(res) # rebuilbing errors

    #     # Rebuild Losses
    #     num_losses = len(self.criterion.l_names)
    #     loss_dict = {k:v for k,v in zip(self.criterion.l_names, res[:num_losses])}
    #     # Rebuild TrackInstances
    #     track_instances = TrackInstances(vars(self.args))
    #     track_instances._fields = {k:v for k,v in zip(tr_keys, res[num_losses:])}
    #     track_instances._idxs = [res[num_losses].shape[0], 0]

    #     print(track_instances)

    #     return loss_dict, track_instances

    # def _get_ckpt_fn(self, tr_keys, _idxs):
    #     """Step 2.
    #     We create the ckpt function in a context where there are no tensors"""

    #     def ckpt_fn(exemplars, frame, mask, gt_box, gt_obj_ids, gt_labels, *tr_inst_args):
    #         assert len(tr_keys)==len(tr_inst_args)

    #         # re_build ground truth and track instances
    #         gt_instances = Instances((1,1), boxes=gt_box, obj_ids=gt_obj_ids, labels=gt_labels)
    #         track_instances = TrackInstances(vars(self.args), init=False, _idxs=_idxs)
    #         for k,v in zip(tr_keys, tr_inst_args):
    #             track_instances._fields[k] = v

    #         # forward
    #         loss_dict, track_instances = self.forward_frame_train(exemplars, frame, gt_instances, mask, track_instances)
    #         self.criterion.l_names = loss_dict.keys()

    #         # return tuple of tensors we are interested in calling backward on
    #         asd = decompose_output([loss_dict, track_instances._fields])
    #         return asd
    #     return ckpt_fn
