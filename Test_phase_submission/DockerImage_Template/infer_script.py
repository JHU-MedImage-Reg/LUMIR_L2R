import os
from torch.utils.data import DataLoader
import numpy as np
import torch, glob
import torch.nn.functional as F
from natsort import natsorted
import nibabel as nib
import json
from torch.utils.data import Dataset
from argparse import ArgumentParser

class L2RLUMIRJSONDataset(Dataset):
    def __init__(self, base_dir, json_path):
        with open(json_path) as f:
            d = json.load(f)
        self.imgs = d['validation']
        self.base_dir = base_dir

    def __getitem__(self, index):
        img_dict = self.imgs[index]
        mov_path = img_dict['moving']
        fix_path = img_dict['fixed']
        x = nib.load(self.base_dir + mov_path)
        y = nib.load(self.base_dir + fix_path)
        x = x.get_fdata() / 255.
        y = y.get_fdata() / 255.
        x, y = x[None, ...], y[None, ...]
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.float(), y.float()

    def __len__(self):
        return len(self.imgs)

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    input_dir = "./input/"
    output_dir = "./output/"
    wts_dir = "./pretrained_weights/"
    json_dir = "./LUMIR_dataset.json"
 
    '''
    Initialize model
    '''
    model = #Your model
    pretrained = torch.load(wts_dir + natsorted(os.listdir(wts_dir))[0], map_location=torch.device('cpu'))
    model.load_state_dict(pretrained['state_dict'])
    print('model: {} loaded!'.format(natsorted(os.listdir(wts_dir))[0]))

    '''
    Initialize dataset
    '''
    val_set = L2RLUMIRJSONDataset(base_dir=input_dir, json_path=json_dir)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    val_files = val_set.imgs
    
    '''
    inference
    '''
    print('Inference begins!\n')
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            mv_id = val_files[i]['moving'].split('_')[-2]
            fx_id = val_files[i]['fixed'].split('_')[-2]
            model.eval()
            x = data[0]
            y = data[1]
            flow = model((x, y)) # obtain displacement field
            flow = flow.cpu().detach().numpy()[0]
            save_nii(flow, output_dir + 'disp_{}_{}'.format(fx_id, mv_id))
            print('disp_{}_{}.nii.gz saved to {}'.format(fx_id, mv_id, output_dir))

if __name__ == '__main__':
    torch.manual_seed(0)
    main()