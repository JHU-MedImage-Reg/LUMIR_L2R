from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn.functional as F

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    weights = [1, 1] # loss weights
    train_dir = 'G:/DATA/LUMIR/'
    save_dir = 'TransMorphTVF_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 #learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    H, W, D = 160, 224, 192
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H//2, W//2, D//2)
    config.window_size = (H // 64, W // 64, D // 64)
    config.out_chan = 3
    model = TransMorph.TransMorphTVF(config, time_steps=7)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model((H, W, D), 'nearest')
    reg_model.cuda()
    spatial_trans = TransMorph.SpatialTransformer((H, W, D)).cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 85
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[0]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_set = datasets.L2RLUMIRJSONDataset(base_dir=train_dir, json_path=train_dir+'LUMIR_dataset.json', stage='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if cont_training:
        opt_state = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['optimizer']
        optimizer.load_state_dict(opt_state)
        print('Optimizer loaded!')

    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            with torch.no_grad():
                x = data[0].cuda()
                y = data[1].cuda()
                x_half = F.avg_pool3d(x, 2).cuda()
                y_half = F.avg_pool3d(y, 2).cuda()
            flow = model((x_half, y_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = spatial_trans(x, flow)
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow, y) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow = model((y_half, x_half))
            flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear', align_corners=False) * 2
            output = spatial_trans(y, flow)
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_reg = criterion_reg(flow, x) * weights[1]
            loss = loss_ncc + loss_reg
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_ncc.item(),
                                                                                            loss_reg.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='transmorph_wts.pth.tar')

        grid_img = mk_grid_img(8, 1)
        def_grid = spatial_trans(grid_img.float(), flow.cuda())
        plt.switch_backend('agg')
        pred_fig = comput_fig(output)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('moving', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('fixed', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('deformed', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 224, 192)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
