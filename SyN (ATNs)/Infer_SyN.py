'''
SyN (ATNs) for LUMIR at Learn2Reg 2024
Author: Junyu Chen
        Johns Hopkins University
        jchen245@jhmi.edu
Date: 05/31/2024
'''
import os, random, glob, sys
import numpy as np
from data import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import nibabel as nib
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '6'
import ants

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    val_dir = '/scratch/jchen/DATA/LUMIR/'
    if not os.path.exists('LUMIR_outputs/'):
        os.makedirs('LUMIR_outputs/')

    val_set = datasets.L2RLUMIRJSONDataset(base_dir=val_dir, json_path=val_dir + 'LUMIR_dataset.json',
                                           stage='validation')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    val_files = val_set.imgs

    '''
    Validation
    '''
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            mv_id = val_files[i]['moving'].split('_')[-2]
            fx_id = val_files[i]['fixed'].split('_')[-2]
            x_image = data[0]
            y_image = data[1]
            x = x_image.squeeze(0).squeeze(0).detach().cpu().numpy()
            y = y_image.squeeze(0).squeeze(0).detach().cpu().numpy()
            
            x_ants = ants.from_numpy(x)
            y_ants = ants.from_numpy(y)
            print('start registration: {}'.format(i))
            reg12 = ants.registration(y_ants, x_ants, 'antsRegistrationSyN[so]', syn_metric='CC', reg_iterations=(200, 200, 200), verbose=True)
            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:, :, :, 0, :].transpose(3, 0, 1, 2)
            save_nii(flow, 'LUMIR_outputs/' + 'disp_{}_{}'.format(fx_id, mv_id))
            print('disp_{}_{}.nii.gz saved to {}'.format(fx_id, mv_id, 'LUMIR_outputs/'))


def seedBasic(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    DEFAULT_RANDOM_SEED = 12
    seedBasic(DEFAULT_RANDOM_SEED)
    seedTorch(DEFAULT_RANDOM_SEED)
    main()
