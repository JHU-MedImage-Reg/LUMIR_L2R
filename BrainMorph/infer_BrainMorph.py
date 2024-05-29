'''
BrainMorph for LUMIR at Learn2Reg 2024
Author: Junyu Chen
        Johns Hopkins University
        jchen245@jhmi.edu
Date: 05/28/2024
'''
import os, random, glob, sys

import matplotlib.pyplot as plt
import numpy as np
from data import datasets
from torch.utils.data import DataLoader
import torch, shutil
import torch.nn.functional as F
import nibabel as nib

def standardize_flow(flow):
    flow = torch.from_numpy(flow[None, ])
    flow = flow[...,[2,1,0]]
    flow = flow.permute(0, 4, 1, 2, 3)  # Bring channels to second dimension
    shape = flow.shape[2:]

    # Scale normalized flow to pixel indices
    for i in range(3):
        flow[:, i, ...] = (flow[:, i, ...] + 1) / 2 * (shape[i] - 1)

    # Create an image grid for the target size
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids, dim=0).unsqueeze(0).to(flow.device, dtype=torch.float32)

    # Calculate displacements from the image grid
    disp = flow - grid
    return disp.cpu().detach().numpy()[0]

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
    val_dir = 'F:/Junyu/DATA/LUMIR/'
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
            if os.path.exists('./register_output'):
                shutil.rmtree('./register_output')
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

            os.system('python scripts/register.py --num_keypoints 512 --variant H --weights_dir ./weights/ --moving x.nii.gz --fixed y.nii.gz --list_of_aligns tps_0 --list_of_metrics mse --save_eval_to_disk --download')

            flow = np.load('register_output/eval_numkey512_variantH/0_0_fixed_moving/grid_0-fixed_0-moving-rot0-tps_0.npy')
            flow = standardize_flow(flow)
            flow = flow[:, (256-160)//2:(256-160)//2+160, (256-224)//2:(256-224)//2+224, (256-192)//2:(256-192)//2+192]

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
