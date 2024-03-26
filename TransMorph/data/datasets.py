import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import nibabel as nib

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
