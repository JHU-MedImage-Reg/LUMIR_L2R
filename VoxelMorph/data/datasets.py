import os, glob
import torch, json
from torch.utils.data import Dataset
import random
import numpy as np
import nibabel as nib
import sys
class L2RLUMIRDataset(Dataset):
    def __init__(self, data_path, stage='train'):
        self.paths = glob.glob(data_path + '*.nii.gz')
        self.stage = stage
        self.pairs = None
        self.stage = stage
        if stage != 'train':
            self.pairs = self.set_pairs(data_path)

    def set_pairs(self, data_path):
        file_names = glob.glob(data_path + '*.nii.gz')
        mv = file_names[0:-1]
        fx = file_names[1:]
        return mv, fx

    def __getitem__(self, index):
        if self.stage == 'train':
            path = self.paths[index]
            tar_list = self.paths.copy()
            tar_list.remove(path)
            random.shuffle(tar_list)
            tar_file = tar_list[0]
        else:
            path = self.pairs[0][index]
            tar_file = self.pairs[1][index]
        x = nib.load(path)
        x = x.get_fdata()/255.
        y = nib.load(tar_file)
        y = y.get_fdata()/255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.paths)

class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path, stage='train'):
        with open(json_path) as f:
            d = json.load(f)
        if stage.lower()=='train':
            self.imgs = d['training']
        elif stage.lower()=='validation':
            self.imgs = d['validation']
        else:
            raise 'Not implemented!'
        self.base_dir = base_dir
        self.stage = stage

    def __getitem__(self, index):
        if self.stage == 'train':
            mov_dict = self.imgs[index]
            fix_dicts = self.imgs.copy()
            fix_dicts.remove(mov_dict)
            random.shuffle(fix_dicts)
            fix_dict = fix_dicts[0]
            x = nib.load(self.base_dir+mov_dict['image'])
            y = nib.load(self.base_dir+fix_dict['image'])

        else:
            img_dict = self.imgs[index]
            mov_path = img_dict['moving']
            fix_path = img_dict['fixed']
            x = nib.load(self.base_dir + mov_path)
            y = nib.load(self.base_dir + fix_path)
        x = x.get_fdata() / 255.
        y = y.get_fdata() / 255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)  # [channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)