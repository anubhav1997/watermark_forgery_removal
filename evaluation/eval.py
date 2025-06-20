import math
import numpy as np
import scipy
from PIL import Image
import torch
import torchvision.transforms as tforms
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline#, StableDiffusion3Pipeline
import torchvision
from torch.autograd import Variable 
import torch_dct as dct
from DWT import *
from utils import * 
import pandas as pd 
import os 
import glob 
import random 
import lpips
import argparse

import cv2 


def to_numpy(clean_img):
    clean_img = (clean_img / 2 + 0.5).clamp(0, 1).squeeze()
    clean_img = (clean_img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    return clean_img


def psnr_PIL(img1: Image, img2: Image) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two PIL images.

    @param img1: The first PIL image.
    @param img2: The second PIL image.
    
    @return: The PSNR value between the two images.
    """
    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path_adv",
        type=str,
        default="",
    )
    parser.add_argument(
        "--path_org",
        type=str,
        default="coco_org_10k",
    )
    

    return parser.parse_args()

args = parse_args()


paths_org = sorted(glob.glob(args.path_org+ '/*'))
paths_adv = sorted(glob.glob(args.path_adv+ '/*'))


l2 = 0
l_inf = 0
l1 = 0
# fid = 0
# ssim = 0
lipis_score = 0


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_fn_alex = lpips.LPIPS(net='alex')#.to(device) # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

asr = 0

img_dist_org = []
img_dist_adv = []

total =0

p_water = []
p_non = []
p_ours = []

for i in range(200):#, path in enumerate(paths_adv):
    # print(i)
    try:
        path = glob.glob(args.path_adv+ f'/{i}_*')[0]
        adv = transform_img(load_clean_img(path)).unsqueeze(0)#.to(device)
    except:
        continue 

    # if i ==192 or i==174 or i==172:
    #     print(paths_org[i])
    #     org = transform_img(load_clean_img(paths_org[i])).unsqueeze(0)
    #     save_img(org, f"{i}.png")
    #     continue 
    # else:
    #     continue 
        
    org = transform_img(load_clean_img(paths_org[i])).unsqueeze(0) #.to(device)
    
    if org.shape[1] !=3:
        continue 

    asr += int(float(path.split("_")[-1].split('.png')[0]) <= 0.05 )
    
    p_ours.append(float(path.split("_")[-1].split('.png')[0]))
    p_water.append(float(path.split("_")[-3]))
    p_non.append(float(path.split("_")[-2]))

    lipis_score += loss_fn_alex(adv, org).item()
    l1 += torch.linalg.norm(torch.flatten(adv - org), ord=1)
    l2 += torch.linalg.norm(torch.flatten(adv - org), ord=2)
    l_inf += torch.linalg.norm(torch.flatten(adv - org), ord=float('inf'))
    total+=1
    
    img_dist_org.append((org+1)/2.)
    img_dist_adv.append((adv+1)/2.)


print("Total", total)
print("ASR", asr*100/float(total))
print("L1", l1/float(total))
print("L2", l2/float(total))
print("L_inf", l_inf/float(total))
print("lpips", lipis_score/float(total))

# print(len(img_dist_org), img_dist_org[0].shape)
img_dist_org = torch.cat(img_dist_org)
img_dist_adv = torch.cat(img_dist_adv)
# print(img_dist_adv.shape)

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt 
import seaborn as sns 
from torchmetrics.image import PeakSignalNoiseRatio

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
fid = FrechetInceptionDistance(normalize = True)
psnr = PeakSignalNoiseRatio()

fid.update(img_dist_org, real=True)
fid.update(img_dist_adv, real=False)
fid_score = fid.compute()

ssim_score = ssim(img_dist_adv, img_dist_org) 

psnr_score = psnr(img_dist_adv, img_dist_org) 

print("SSIM", ssim_score)

print("PSNR", psnr_score)
print("FID", fid_score)

plt.figure() #figsize=(15,7)
sns.histplot(p_water, bins=2, label='Watermarked', color='g', stat='probability', log_scale=False) #alpha=0.7,
sns.histplot(p_non, bins=10, label='Non-watermarked', color='r', stat='probability', log_scale=False)
sns.histplot(p_ours, bins=10, label='Adv-watermarked', color='b', stat='probability', log_scale=False)
plt.axvline(x = 0.05, color = 'black', linestyle='--', label = 'Threshold')
plt.xlabel('P-Value', fontsize=25)
# plt.xlim(1e-3, 1)
plt.ylabel('Frequency', fontsize=25)
plt.legend(fontsize=15)

plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show()
plt.savefig(f"plot_{args.path_adv.split('/')[-1]}.png")


