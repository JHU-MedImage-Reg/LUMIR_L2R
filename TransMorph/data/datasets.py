import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import nibabel as nib

class OpenBHBBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = glob.glob(data_path + '*.nii.gz')
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x = nib.load(path)
        x = x.get_fdata()/255.
        y = nib.load(tar_file)
        y = y.get_fdata()/255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.paths)