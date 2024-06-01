"""
uniGradICON for LUMIR at Learn2Reg 2024
Author: Lin Tian
        University of North Carolina at Chapel Hill
        lintian@cs.unc.edu
Tested by: 
        Junyu Chen
        Johns Hopkins University
        jchen245@jhmi.edu
Date: 05/31/2024
"""

import argparse
import json
import os

import footsteps
import icon_registration as icon
import itk
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unigradicon import make_network

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--io_steps", type=int, default=0, help="Steps for IO")
parser.add_argument("--device", type=int, default=0, help="GPU ID.")
parser.add_argument("--exp", type=str, default="", help="Experiment name.")


origin_shape = [1, 1, 192, 224, 160]
input_shape = [1, 1, 175, 175, 175]

args = parser.parse_args()
weights_path = args.weights_path
device = torch.device(f'cuda:{args.device}')
torch.cuda.set_device(device)

if args.exp == "":
    footsteps.initialize(output_root="evaluation_results/")
else:
    footsteps.initialize(output_root="evaluation_results/", run_name=f"{args.exp}/L2R_LUMIR")

os.makedirs(f"{footsteps.output_dir}/submission/LUMIR", exist_ok=True)

def preprocess(img):
    im_min, im_max = torch.min(img), torch.quantile(img.view(-1), 0.99)
    img = torch.clip(img, im_min, im_max)
    img = (img-im_min) / (im_max-im_min)
    return img

def finetune_execute(model, image_A, image_B, steps, lr=2e-5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.eval()
    return loss

net = make_network(input_shape, include_last_step=True)

net.regis_net.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
net.to(device)
net.eval()


with open(f"{args.data_folder}/LUMIR_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["validation"]]

spacing = 1.0 / (np.array(origin_shape[2::]) - 1)
identity = torch.from_numpy(icon.mermaidlite.identity_map_multiN(origin_shape, spacing)).to(device)

dices = []
dices_origin = []
flips = []
back_flips = []

original_state_dict = net.state_dict()
for (fixed_path, moving_path) in tqdm(test_cases):
    # Restore net weight in case we ran IO
    net.load_state_dict(original_state_dict)

    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))

    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = preprocess(fixed)
    fixed_in_net = F.interpolate(fixed, input_shape[2:], mode='trilinear', align_corners=False)
    
    moving = torch.Tensor(np.array(moving)).unsqueeze(0).unsqueeze(0)
    moving = preprocess(moving)
    moving_in_net = F.interpolate(moving, input_shape[2:], mode='trilinear', align_corners=False)

    if args.io_steps > 0:
        loss = finetune_execute(net, moving_in_net.to(device), fixed_in_net.to(device), args.io_steps)

    with torch.no_grad():
        net(moving_in_net.to(device), fixed_in_net.to(device))

        # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
        # maps computed by the model
        phi_AB = net.phi_AB(identity)

        flips.append(icon.losses.flips(phi_AB, in_percentage=True).item())

        # Transform to displacement format that l2r evaluation script accepts
        disp = (phi_AB - identity)[0].cpu()

        network_shape_list = list(identity.shape[2:])

        dimension = len(network_shape_list)

        # We convert the displacement field into an itk Vector Image.
        scale = torch.Tensor(network_shape_list)

        for _ in network_shape_list:
            scale = scale[:, None]
        disp *= scale

        disp_itk_format = (
            disp.float()
            .numpy()[list(reversed(range(dimension)))]
            .transpose([3,2,1,0])
        )

    # Save to output folders
    disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
    nib.save(disp_itk_format, f"{footsteps.output_dir}/submission/LUMIR/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.nii.gz")

import subprocess

subprocess.call("zip -r submission.zip ./*", shell=True, cwd=f"{footsteps.output_dir}/submission/")  
print(f"Mean folds percentage: {np.mean(flips)}")
