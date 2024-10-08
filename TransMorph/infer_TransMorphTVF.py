import os
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn.functional as F
from natsort import natsorted
import nibabel as nib

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def save_nii(img, file_name, pix_dim=[1., 1., 1.]):
    x_nib = nib.Nifti1Image(img, np.eye(4))
    x_nib.header.get_xyzt_units()
    x_nib.header['pixdim'][1:4] = pix_dim
    x_nib.to_filename('{}.nii.gz'.format(file_name))

def main():
    weights = [1, 1] # loss weights
    val_dir = 'G:/DATA/LUMIR/'
    ptrain_wts_dir = 'TransMorphTVF_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('outputs/'+ptrain_wts_dir):
        os.makedirs('outputs/'+ptrain_wts_dir)

    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 64, W // 64, D // 64)
    config.out_chan = 3
    model = TransMorph.TransMorphTVF(config, time_steps=7)
    pretrained = torch.load('experiments/' + ptrain_wts_dir + natsorted(os.listdir('experiments/' + ptrain_wts_dir))[0])
    model.load_state_dict(pretrained['state_dict'])
    print('model: {} loaded!'.format(natsorted(os.listdir('experiments/' + ptrain_wts_dir))[0]))
    model.cuda()

    '''
    Initialize dataset
    '''
    val_set = datasets.L2RLUMIRJSONDataset(base_dir=val_dir, json_path=val_dir+'LUMIR_dataset.json', stage='validation')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    val_files = val_set.imgs
    '''
    Validation
    '''
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            mv_id = val_files[i]['moving'].split('_')[-2]
            fx_id = val_files[i]['fixed'].split('_')[-2]
            model.eval()
            x = data[0].cuda()
            y = data[1].cuda()
            x_half = F.avg_pool3d(x, 2).cuda()
            y_half = F.avg_pool3d(y, 2).cuda()
            flow = model((x_half, y_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            flow = flow.cpu().detach().numpy()[0]
            #np.savez('outputs/' + ptrain_wts_dir + 'disp_{}_{}.npz'.format(fx_id, mv_id), flow)
            save_nii(flow, 'outputs/' + ptrain_wts_dir + 'disp_{}_{}'.format(fx_id, mv_id))
            print('disp_{}_{}.nii.gz saved to {}'.format(fx_id, mv_id, 'outputs/' + ptrain_wts_dir))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()
