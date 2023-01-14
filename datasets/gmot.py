import os
import random
import pathlib
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
from datasets.transforms import make_viddataset_transforms
from util.misc import Instances


# load tracking labels (MOT format)
def load_labels(txt_path):
    gt = defaultdict(lambda: [])
    data = np.genfromtxt(txt_path, delimiter=',', dtype=np.int)

    for line in data:
        gt[line[0]].append( [line[2], line[3], line[2]+line[4], line[3]+line[5], line[1]] )
    return gt

def get_images(video_name, d):
    base_path = f"{d['sequences']}/{video_name}/img1/"
    images = [] # list of tuple(path to file, frame number)
    for img in os.listdir(base_path):
        fr_n = int(img.split('.')[0])  #  000231.jpg --> 231
        path = base_path + img
        images.append( (path, fr_n) )
    return images


class GMOTDataset(Dataset):

    def __init__(self, args, split='all', transform=None):
        self._cache = {}        # store precomputed inputs (images, patches, ...)
        self.args = args
        self.current_epoch = 0
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.seqnames = []
        self.transform = transform

        # check files
        self.dataset_path = args.gmot_dir

        # select files for the split
        self.data = {}
        self.indices = []
        self.vid_tmax = {}
        self.video_dict = {}
        self.load_files(split, self.dataset_path)

        self.sampler_steps: list = []
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

    def load_files(self, split, dataset_path):
        videos = os.listdir(f'{dataset_path}/GenericMOT_JPEG_Sequence')
        if split == 'train':
            videos = [v for v in videos if ('0' in v) or ('2' in v)]
        if split == 'val':
            videos = [v for v in videos if ('3' in v)]
        if split == 'test':
            videos = [v for v in videos if ('1' in v)]

        for vid in videos:
            txt_path = f'{dataset_path}/track_label/{vid}.txt'
            self.data[vid] = load_labels(txt_path)

            if self.args.small_ds:
                self.data[vid] = {k:v for k,v in self.data[vid].items() if k<40}

            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.data[vid].keys())
            t_max = max(self.data[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = random.randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def __getitem__(self, idx):
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        images, targets = self.pre_continuous_frames(vid, indices)

        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances = []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)

        exemplar = self.get_exemplar(images[0], targets[0])

        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'exemplar': exemplar,
        }

    def __len__(self):
        return len(self.indices)

    def get_exemplar(self,img,target):
        bb = target['boxes'][0].clone()
        bb = (bb.view(2,2) * torch.tensor([img.shape[2],img.shape[1]]).view(1,2)).flatten()  # coords in img
        bb = torch.cat((bb[:2]-bb[2:]/2, bb[:2]+bb[2:]/2)).int()               # x1y1x2y2
        patch = img[:, bb[1]:bb[3], bb[0]:bb[2]]
        return [patch]


    def _pre_single_frame(self, vid, idx):
        if (vid,idx) not in self._cache:
            xywhi = self.data[vid][idx]
            img_path = f'{self.dataset_path}/GenericMOT_JPEG_Sequence/{vid}/img1/{idx:06d}.jpg'
            img = Image.open(img_path)

            obj_idx_offset = self.video_dict[vid] * 100000
            target = {
                'boxes': [box[:4] for box in xywhi],
                'labels': [0 for _ in range(len(xywhi))],
                'obj_ids': [obj_idx_offset+id[-1] for id in xywhi],

                # could be removed if fix transformation
                'iscrowd': torch.tensor([0 for _ in range(len(xywhi))]),
                'scores': torch.tensor([0 for _ in range(len(xywhi))]),
            }

            target['labels'] = torch.tensor(target['labels'])
            target['obj_ids'] = torch.tensor(target['obj_ids'], dtype=torch.float64)
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
            
            if len(self._cache) < 100:
                self._cache[(vid,idx)] = img, target
        else:
            img, target = self._cache[(vid,idx)]
        
        return  img, target
        
    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        self.set_epoch(self.current_epoch + 1)

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range


def build(image_set, args):
    root = pathlib.Path(args.gmot_dir)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = make_viddataset_transforms(args, image_set)
    
    dataset = GMOTDataset(args, image_set, transform)

    return dataset