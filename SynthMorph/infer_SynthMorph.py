'''
SynthMorph for LUMIR at Learn2Reg 2024
Author: Junyu Chen
        Johns Hopkins University
        jchen245@jhmi.edu
Date: 05/20/2024
'''
import os, random, glob, sys
import numpy as np
from data import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import nibabel as nib


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')
            
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
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
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
            print('start registration: {}'.format(i))
            x_nib = nib.Nifti1Image(x, np.eye(4))
            x_nib.header.get_xyzt_units()
            x_nib.header['pixdim'][1:4] = [1., 1., 1.]
            x_nib.to_filename('x.nii.gz')

            y_nib = nib.Nifti1Image(y, np.eye(4))
            y_nib.header.get_xyzt_units()
            y_nib.header['pixdim'][1:4] = [1., 1., 1.]
            y_nib.to_filename('y.nii.gz')

            
            #print(def_.shape)
            os.system('/scratch/jchen/python_projects/synthmorph/synthmorph -m deform -t def.nii.gz x.nii.gz y.nii.gz')
            flow = nib_load('def.nii.gz')
            
            #np.savez('LUMIR_outputs/' + 'disp_{}_{}.npz'.format(fx_id, mv_id), flow)
            save_nii(flow, 'LUMIR_outputs/' + ptrain_wts_dir + 'disp_{}_{}')
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
