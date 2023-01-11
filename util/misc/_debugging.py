import torch
import cv2
import numpy as np
import os

class Visualizer():
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_dir+'/debug/', exist_ok=True)

    def _debug_frame(self, frame, out_w=400):
        """util to make frame to writable"""
        if len(frame.shape) == 4: frame = frame[0]
        frame = np.ascontiguousarray(frame.clone().cpu().permute(1,2,0).numpy() [:,:,::-1]) /4+0.4 # frame in BGR
        frame = np.uint8(255*(frame-frame.min())/(frame.max()-frame.min()))
        h,w,_ = frame.shape
        return cv2.resize(frame, (out_w,int(out_w*h/w)))

    @torch.no_grad()
    def visualize_gt(self, data_dict):
        out_d = self.args.output_dir
        if os.path.exists(out_d+'/debug/gt_visualize.jpg'): return        
        # image shape
        num_imgs = len(data_dict['imgs']) + 1
        num_rows = int(np.sqrt(num_imgs))
        whites_to_add = 1 + num_imgs - num_rows*(num_rows+1)

        # write bboxes on images
        imgs = []
        for img, gt in zip(data_dict['imgs'], data_dict['gt_instances']):
            img = self._debug_frame(img, 600)
            H,W,_ = img.shape
            def clean(x,X): return int(max(0,min(x, X-1)))
            for box in gt.boxes:
                box = (box.view(2,2) * torch.tensor([W, H], device=box.device).view(1,2)).int()
                x1,x2 = box[0,0] - box[1,0].div(2,rounding_mode='trunc'), box[0,0] + box[1,0].div(2,rounding_mode='trunc')
                y1,y2 = box[0,1] - box[1,1].div(2,rounding_mode='trunc'), box[0,1] + box[1,1].div(2,rounding_mode='trunc')
                x1,x2,y1,y2 = clean(x1,W),clean(x2,W),clean(y1,H),clean(y2,H)
                tmp = img[y1:y2, x1:x2].copy()
                img[y1-2:y2+2, x1-2:x2+2] = (255,0,255)
                img[y1:y2, x1:x2] = tmp
            imgs.append(img)
        imgs += [200*np.ones_like(img) for _ in range(whites_to_add)]

        # add exemplar
        exe_scale = int(data_dict['exemplar'][0].shape[2] * 600 / data_dict['imgs'][0].shape[2])
        exemplar = self._debug_frame(data_dict['exemplar'][0], exe_scale)
        h1,h2 = H//3, H//3 +exemplar.shape[0]
        w1,w2 = W//3, W//3 +exemplar.shape[1]
        imgs[-1][h1:h2, w1:w2] = exemplar

        # unique image
        imgs = np.stack(imgs).reshape(num_rows, num_rows+1, H,W,3)
        imgs = np.concatenate([i for i in imgs], axis=1)
        imgs = np.concatenate([i for i in imgs], axis=1)

        cv2.imwrite(out_d+'/debug/gt_visualize.jpg', imgs)

    @torch.no_grad()
    def visualize_infographics(self, frames, gt, outputs, num, path):
        os.makedirs(self.args.output_dir+'/debug/'+path.split('/')[-2], exist_ok=True)
        # where to save the file
        if num==-1: num = len(frames)-1
        path = path+f'f{num}_'

        # info needed on that frame
        frame = frames[num].unsqueeze(0)
        q_ref = outputs[num]['q_ref'].cpu()
        coord = outputs[num]['position'].cpu()
        isobj = outputs[num]['is_object'].cpu()
        n_prop = q_ref.shape[1] - len(gt[num])

        assignments = outputs[num]['debug'].cpu()
        matching = outputs[num]['matching'].cpu()
        gt_boxes = gt[num].boxes

        queries = outputs[num]['input_hs'].cpu()
        img_features = [i.cpu() for i in outputs[num]['img_features_pos']]

        # helper functions for graphics
        self.debug_q_similarity(queries, img_features, q_ref, n_prop, path)
        self.debug_matching(frame, coord[-1], gt_boxes, matching, n_prop, path)
        self.debug_qref_steps(frame, q_ref, coord, isobj, n_prop, path)
        self.debug_predictions(frame, coord[-1], assignments, path)
        self.debug_qref_start(frame, q_ref, n_prop, path)
    
    def debug_q_similarity(self, queries, img_features, q_ref, n_prop, path):
        out_file = self.args.output_dir+f'/debug/{path}similarity.jpg'
        idxs = [0,n_prop,n_prop+1]

        if not os.path.exists(out_file):
            imgs = []
            q_emb = queries[0, idxs] # 3,256
            q_ref = (q_ref[:, idxs] + torch.tensor([[0.,0,0,0],[0,1,0,0],[0,2,0,0]], device=q_ref.device)) * torch.tensor([1,0.333,1,1], device=q_ref.device)
            w, h = 128, 80*3

            for img in img_features: # 1,256,h,w
                similarity = (q_emb @ img[0].transpose(0,1)).transpose(0,1).contiguous()  # 3,h,w
                idx = similarity.flatten(1).max(dim=(1))[1]
                i,j = torch.div(idx,similarity.shape[-1], rounding_mode='trunc').long(), (idx%similarity.shape[-1]).long()

                # similarity = similarity.sigmoid()
                similarity = similarity.view(3,-1).softmax(dim=1).view(3,similarity.shape[1],-1)

                similarity = (similarity-similarity.min()) / (similarity.max()-similarity.min())
                for d1,d2 in [(0,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    similarity[[0,1,2], (i+d1).clamp(0,similarity.shape[1]-1), (j+d2).clamp(0,similarity.shape[2]-1)] = 1 
                similarity = torch.cat([i for i in similarity], dim=0)[:,:,None].expand(-1,-1,3)
                similarity = np.uint8(similarity.cpu().numpy()*255)     # h*3, w, 1
                similarity = cv2.resize(similarity, (w, h))
                similarity[:,-1:] = similarity[:,-1:].clip(120)         # grey border
                similarity[[h//3,h*2//3],:] = similarity[[h//3,h*2//3],:].clip(120)   # grey border
                similarity = self._debug_qref(similarity, q_ref, 3, 0, 5)
                imgs.append(similarity)
            frame = np.concatenate(imgs, axis=1)
            cv2.imwrite(out_file, frame)


    def debug_matching(self, frame, q_ref, gt_boxes, matching, n_prop, path):
        out_file = self.args.output_dir+f'/debug/{path}matching.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            # all queries
            q_ref = q_ref[:,:n_prop]
            frame = self._debug_qref(frame, q_ref, n_prop, 1/5)

            # init matched ref
            start = q_ref[0, matching>=0].unsqueeze(0)
            frame = self._debug_qref(frame, start, 0, 2/3)

            # gt_boxes matched ref
            end = matching[matching>=0]
            end = gt_boxes[end].unsqueeze(0)
            frame = self._debug_qref(frame, end, 0)

            # lines
            frame = self._debug_lines(frame, start, end)

            cv2.imwrite(out_file, frame)



    def debug_qref_steps(self, frame, q_ref, later_ref, scores, n_prop, path):
        """save image with evolution of some ref points"""
        out_file = self.args.output_dir+f'/debug/{path}ref_steps.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            _,i = scores[-1, 0,:n_prop].topk(3, dim=0)
            _,i2 = scores[-1, 0,:n_prop].min(dim=0)
            idxs = i.view(-1).tolist()+i2.tolist()+[0, n_prop]
            q_refs = [q_ref[:,idxs]] + [ref[:,idxs] for ref in later_ref]
            for i, q_ref in enumerate(q_refs):
                frame = self._debug_qref(frame, q_ref, 5, (i+1)/len(q_refs))
            for i in range(len(q_refs)-1):
                frame = self._debug_lines(frame, q_refs[i], q_refs[i+1])
            cv2.imwrite(out_file, frame)

    def debug_predictions(self, frame, coord, assign, path):
        """print predicted boundingboxes"""
        # white=confident; green=gt; purple=old; orange=new;    half_black=not_confident
        out_file = self.args.output_dir+f'/debug/{path}pred.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            for i,box in enumerate(coord[0]):
                if assign[i] == 0: continue
                frame = self._draw_box(frame, box, assign[i])
            cv2.imwrite(out_file, frame)

    def debug_qref_start(self, frame, q_ref, n_prop, path):
        """save image with initial reference points"""
        out_file = self.args.output_dir+f'/debug/{path}ref_start.jpg'
        if not os.path.exists(out_file):
            frame = self._debug_frame(frame)
            frame = self._debug_qref(frame, q_ref, n_prop)
            cv2.imwrite(out_file, frame)


    def _debug_lines(self, frame, ref1, ref2):
        """utils to print movements of q_refs"""
        ref1 = ref1[0,:,:2]
        ref2 = ref2[0,:,:2]
        H,W,_ = frame.shape
        def clean_pt(x,X): return int(max(0,min(x*X, X-1)))
        for (w1,h1), (w2, h2) in zip(ref1, ref2):
            w1 = clean_pt(w1, W); h1 = clean_pt(h1, H)
            w2 = clean_pt(w2, W); h2 = clean_pt(h2, H)
            
            frame = cv2.line(frame, (w1,h1), (w2,h2), (0,0,0), 1) 
        return frame

    def _debug_qref(self, frame, q_ref, n_prop=500, opacity=1, s=2):
        """util to write q_refs on frame"""
        q_ref = q_ref[0,:,:2]
        H,W,_ = frame.shape
        c2 = np.array([255,100,80])  # blue border
        def clean(x,X): return int(max(0, min(X-1, x)))
        for i, (w, h) in enumerate(q_ref):
            color = np.array([80,250,90] if i<n_prop else [250,80,255]) # green prop;  purple gt
            w = int(w*W)
            h = int(h*H)
            tmp = frame[max(0,h-s)+1:h+s,max(0,w-s)+1:w+s].copy()
            frame[max(h-s,0):h+s+1,max(0,w-s):w+s+1] = ((0.75-opacity/4)*frame[max(h-s,0):h+s+1,max(0,w-s):w+s+1].astype(float) + c2*(.75+opacity/4)).astype(np.uint8)
            frame[max(h-s,0)+1:h+s,max(0,w-s)+1:w+s] = ((1-opacity)*tmp.astype(float) + color*opacity).astype(np.uint8)
        return frame


    def _draw_box(self, frame, box, color_id, border=2):
        # white=confident; green=gt; purple=old; orange=new;    half_black=not_confident
        ## --> eval predictions: white+full_green+full_orange
        color = [255, (0,255,128),(255,0,255),(0,128,255)] [color_id%10]
        h,w,_ = frame.shape

        # get position
        x1,x2 = int(w*(box[0]-box[2]/2)), int(w*(box[0]+box[2]/2))
        y1,y2 = int(h*(box[1]-box[3]/2)), int(h*(box[1]+box[3]/2))
        x1=max(x1,border);x2=min(x2,w-border);y1=max(y1,border);y2=min(y2,h-border)

        # draw box
        tmp = frame[y1:y2, x1:x2].copy()
        frame[y1-border:y2+border, x1-border:x2+border] = color
        if color_id<10: frame[y1-border:y2+border, x1:x2] = 0
        frame[y1:y2, x1:x2] = tmp
        return frame




