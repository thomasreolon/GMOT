
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import cv2
import os

import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import torch

from util.misc import load_for_eval, Instances, box_cxcywh_to_xyxy
from configs.defaults import get_args_parser
from datasets.fscd import build_fscd


def main():

    # Info about code execution
    args = get_args_parser().parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build Model
    model = load_for_eval(args).to(args.device)

    # Load dataset
    dataset = load_svdataset(args.t_dataset_file, 'val', args)

    # rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    # ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    # dataset = dataset[rank::ws]

    # Track
    det = Detector(args, model, dataset)
    for vid in range(len(dataset)):
        det.detect(args.det_thresh, vis=args.debug, video=vid)

def load_svdataset(datasetname, split, args):
    assert datasetname in {'e2e_gmot', 'e2e_fscd'}, f'invalid dataset "{datasetname}"'
    assert split in {'train', 'val', 'test'}, f'invalid dataset "{split}"'
    
    if datasetname=='e2e_gmot':
        return load_gmot(split, args)
    elif datasetname=='e2e_fscd':
        return load_fscd(split, args)

def load_gmot(split, args):
    base_path = args.gmot_dir+'/GenericMOT_JPEG_Sequence/'

    list_dataset = []
    videos = os.listdir(base_path)

    for video in videos:
        # get 1st BB
        gt = args.gmot_dir+f'/track_label/{video}.txt'
        with open(gt, 'r') as fin:
            for _ in range(300):
                line = fin.readline()
                if line[0] == '0': break
        line = [int(l) for l in line.split(',')]
        bb = line[2], line[3], line[2]+line[4], line[3]+line[5], 

        # get images
        imgs = sorted(os.listdir(f'{base_path}{video}/img1'))

        list_dataset.append((f'{base_path}{video}/img1/', imgs, bb))  # none should be the exemplar_bb xyxy

    return list_dataset


def load_fscd(split, args):
    args.sampler_lengths[0] = 20
    args.small_ds = True
    ds = build_fscd(split, args)

    list_dataset = []
    for vid in range(min(len(ds), 3)):
        data = ds[vid]
        images = [img.permute(1,2,0).numpy() for img in data['imgs']]
        exemplar = data['exemplar'][0].permute(1,2,0).numpy()

        list_dataset.append([None, images, exemplar])
    return list_dataset


class ListImgDataset(Dataset):
    def __init__(self, base_path, img_list, exemplar_bb) -> None:
        super().__init__()
        self.base_path = base_path
        self.img_list = img_list
        self.exemplar = exemplar_bb
        self.e = None

        '''
        common settings
        '''
        self.img_height = 704   # 800
        self.img_width = 1216   # 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, fpath_or_ndarray):
        if isinstance(fpath_or_ndarray, str):
            # bb as a box coordinates array [x1,y1,x2,y2]
            bb = self.exemplar
            cur_img = cv2.imread(os.path.join(self.base_path, fpath_or_ndarray))
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            if self.e is None:
                self.e = cur_img[bb[1]:bb[3], bb[0]:bb[2]]
        else:
            # bb as a Tensor
            cur_img = np.array(fpath_or_ndarray)
            cur_img = cur_img/4+.45      # de normalize
            cur_img = cur_img-cur_img.min() / (cur_img.max()-cur_img.min())
            if self.e is None:
                self.e = np.array(self.exemplar)/4+.45
                self.e = self.e-self.e.min() / (self.e.max()-self.e.min())
        assert cur_img is not None
        return cur_img, self.e

    def init_img(self, img, exemplar):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        exem_wh = 1+int(exemplar.shape[1]*scale), 1+int(exemplar.shape[0]*scale)
        if len(exemplar)==3: exemplar = np.transpose(exemplar, (1,2,0))
        exemplar = cv2.resize(exemplar, exem_wh)
        exemplar = F.normalize(F.to_tensor(exemplar), self.mean, self.std)
        return img, ori_img, exemplar

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, exemplar = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, exemplar)


class Detector(object):
    def __init__(self, args, model, dataset):
        self.args = args
        self.gmot = model
        self.dataset = dataset  # list of tuples: (/path/to/MOT/vidname, )

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.6, area_threshold=30, vis=True, video=0):
        total_dts = 0
        total_occlusion_dts = 0

        # for img, ori_img, exemplar in ListImgDataset(*self.dataset[video]):
        #     img = img.squeeze(0).permute(1,2,0).numpy()
        #     exemplar = exemplar.squeeze(0).permute(1,2,0).numpy()
        #     cv2.imshow('img', img/4+.4)
        #     cv2.imshow('exemplar', exemplar/4+.4)
        #     cv2.imshow('ori_img', ori_img)
        #     cv2.waitKey()


        loader = DataLoader(ListImgDataset(*self.dataset[video]), 1, num_workers=2)  
        lines = []
        track_instances = None
        for i, (img, ori_img, exemplar) in enumerate(tqdm(loader)):
            # predict
            img, exemplar = img.to(self.args.device), exemplar.to(self.args.device)
            track_instances = self.gmot.forward_frame_eval(img, track_instances, exemplar) 

            # filter
            i = torch.where(track_instances.score>prob_threshold)
            identities = track_instances.obj_idx[i].clone().cpu().numpy()
            bbox = track_instances.q_ref[i]
            bbox_xyxy = box_cxcywh_to_xyxy(bbox).clone().cpu().numpy()
            
            # reshape
            ori_img = ori_img.squeeze(0)
            seq_h, seq_w, _ = ori_img.shape
            bbox_xyxy *= (seq_w, seq_h, seq_w, seq_h)
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))

                if vis:
                    color = tuple([(((5+track_id*3)*4909 % p)%256) /256 for p in (3001, 1109, 2027)])
                    x1, y1, x2, y2 = [int(a*800/1080) for a in xyxy]
                    tmp = ori_img[ y1:y2, x1:x2].copy()
                    ori_img[y1-3:y2+3, x1-3:x2+3] = color
                    ori_img[y1:y2, x1:x2] = tmp
            if vis:
                # ori_img = cv2.resize(ori_img, (600,350))
                cv2.imshow('preds', ori_img)
                cv2.waitKey(40)
            

        # with open(os.path.join(self.predict_path, f'{self.seq_num}.txt'), 'w') as f:
        #     f.writelines(lines)
        print("totally {} dts {} occlusion dts".format(total_dts, total_occlusion_dts))

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.02, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1



if __name__=='__main__':
    main()


