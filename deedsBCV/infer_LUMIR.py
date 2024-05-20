'''
deedsBCV for LUMIR at Learn2Reg 2024
Author: Junyu Chen
        Johns Hopkins University
        jchen245@jhmi.edu
Date: 05/20/2024
'''
import os, random, glob, sys
import numpy as np
from data_LUMIR import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import nibabel as nib

def load_flow(flow_path: str):
    '''
    Load displacement field from deedsBCV
    :param flow_path: Path of the .nii.gz files
    :return: a displacement tensor in PyTorch format, 1x3xHxWxD
    '''
    disp_ux = nib_load(flow_path+'_ux.nii.gz')[None,]
    disp_vx = nib_load(flow_path+'_vx.nii.gz')[None,]
    disp_wx = nib_load(flow_path+'_wx.nii.gz')[None,]
    disp_arr = np.concatenate([disp_vx, disp_ux, disp_wx], axis=0)
    disp_tensor = torch.from_numpy(disp_arr).float()
    return disp_tensor.unsqueeze(0)

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

def main():
    val_dir = '/mnt/g/DATA/LUMIR/'
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

            os.system('/mnt/d/pythonProject/deedsBCV_reg/deedsBCV/deedsBCV -F y.nii.gz -M x.nii.gz -O output -G 6x5x4x3x2 -L 6x5x4x3x2 -Q 5x4x3x2x1')
            flow = load_flow('dense_disp')

            flow = flow.cpu().detach().numpy()[0]
            np.savez('LUMIR_outputs/' + 'disp_{}_{}.npz'.format(fx_id, mv_id), flow)
            print('disp_{}_{}.npz saved to {}'.format(fx_id, mv_id, 'LUMIR_outputs/'))


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
