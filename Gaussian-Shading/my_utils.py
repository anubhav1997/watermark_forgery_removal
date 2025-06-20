from tqdm import tqdm
import torch
import itertools
import argparse
import os
from datetime import datetime
import pandas as pd
from collections import OrderedDict
from prettytable import PrettyTable

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline#, StableDiffusion3Pipeline
import torch_dct as dct

# from utils import *
from io_utils import *
import torchvision.transforms as tforms

import math
import numpy as np
import scipy
from PIL import Image
import torch
# import torchvision.transforms as tforms
# from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusion3Pipeline
import torchvision
from torch.autograd import Variable 
# import torch_dct as dct
# from DWT import *
# import torchvision
import matplotlib.pyplot as plt 


import hashlib  


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def get_reversed_w(pipe, Fourier_watermark_image_distorted, img_size=512):
    
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    
    # ddim inversion
    if not torch.is_tensor(Fourier_watermark_image_distorted):
        img = transform_img(Fourier_watermark_image_distorted, img_size).unsqueeze(0).to(pipe.vae.dtype).to(pipe.device)
    else:
        img = Fourier_watermark_image_distorted 
        
    Fourier_watermark_image_latents = pipe.vae.encode(img).latent_dist.mode() * (1./pipe.vae.config.scaling_factor) #0.13025
    Fourier_watermark_image_latents = pipe(prompt="", latents=Fourier_watermark_image_latents, guidance_scale=1, num_inference_steps=50, output_type="latent")
    Fourier_watermark_reconstructed_latents = Fourier_watermark_image_latents.images
    
    pipe.scheduler = curr_scheduler
    
    return Fourier_watermark_reconstructed_latents


def pgd_attack_fft(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=1, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(clean_img) #torch.fft.fft(clean_img)
    clean_img_fft = fft#.real
    mask = torch.ones_like(clean_img_fft)
    mask2 = torch.tril(mask, cutoff)
    mask = torchvision.transforms.functional.hflip(mask2)

    # print(torch.testing.assert_close(fft, torch.complex(clean_img_fft, clean_img_fft_imag)))
    data = Variable(clean_img_fft.data, requires_grad=True).to(device)
    generated_image.requires_grad = False 
    vae.requires_grad = False
    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 

    loss_function = torch.nn.MSELoss()
    # lowFre_loss = nn.SmoothL1Loss(reduction='sum')

    if delta != 0:
            
        lowFre_loss = torch.nn.SmoothL1Loss(reduction='sum')
        dwt = DWT_2D_tiny(wavename= wave)
        idwt = IDWT_2D_tiny(wavename= wave)
        inputs_ll = dwt(clean_img.float())
        inputs_ll = idwt(inputs_ll).half()
    
    for i in range(iters) :    
        data.requires_grad = True 
        idct = torch.clamp(dct.idct(data), min=-1, max=1)
        # outputs = vae.encode(torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        outputs = vae.encode(idct).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        # outputs.retain_grad()
        # loss = torch.nn.functional.mse_loss(outputs, generated_image_latents)
        loss = loss_function(outputs, generated_image_latents)
        # print(loss.item())
        if delta!=0:
            adv_ll = dwt(idct.float())
            adv_ll = idwt(adv_ll).half()
    
            loss2 = lowFre_loss(adv_ll, inputs_ll)
        
            loss = loss + delta*loss2 
        
        grad = -1*torch.autograd.grad(loss, data)[0]
        
        adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
        eta = mask*torch.clamp(adv_images - clean_img_fft, min=-lamda, max=lamda)
        # data = torch.clamp(clean_img_fft + eta, min=-1, max=1).detach_()
        data = torch.clamp(clean_img_fft + eta, min=-100000, max=100000).detach_()

    return torch.clamp(dct.idct(data), min=-1, max=1), eta #torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real



def generate(prompt, pipe, w_key, w_mask, img_size=512):
    img = pipe(prompt=prompt, negative_prompt="", num_inference_steps=50, latents=get_noise(pipe, w_key, w_mask, img_size)).images[0]
    # print(img.size)
    # print(max(img), min(img))
    return img 


def load_clean_img(filename):
    img = Image.open(filename)
    return img


def transform_img(image, img_size=512):
    tform = tforms.Compose([tforms.Resize(img_size),tforms.CenterCrop(img_size),tforms.ToTensor()])
    image = tform(image)
    return 2.0 * image - 1.0


def save_img(clean_img, out_path):
    clean_img = (clean_img / 2 + 0.5).clamp(0, 1).squeeze()
    clean_img = (clean_img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    clean_img = Image.fromarray(clean_img)
    clean_img.save(out_path) 




def pgd_attack_fft_closest(generated_image, clean_image_paths, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=1, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(generated_image) 
    generated_image_fft = fft#.real
    mask = torch.ones_like(generated_image_fft)
    mask2 = torch.tril(mask, cutoff)
    mask = torchvision.transforms.functional.hflip(mask2)
    
    data = Variable(generated_image_fft.data, requires_grad=True).to(device)

    loss_function = torch.nn.MSELoss()
    distances = []

    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor)
    with torch.no_grad():
        
        for clean_img_path in clean_image_paths:

            clean_img = load_clean_img(clean_img_path)
            clean_img = transform_img(clean_img).unsqueeze(0).to(vae.dtype).to(vae.device)
            if clean_img.shape[1] !=3:
                distances.append(float("Inf"))
                continue 
            
            clean_image_latents = vae.encode(clean_img).latent_dist.mode() * (1./vae.config.scaling_factor)
            distances.append(loss_function(generated_image_latents, clean_image_latents).item())

    clean_image_path = torch.argmin(torch.tensor(distances))
    clean_image = load_clean_img(clean_img_path)
    clean_image = transform_img(clean_image).unsqueeze(0).to(vae.dtype).to(vae.device)
    
    clean_image.requires_grad = False 
    vae.requires_grad = False
    clean_image_latents = vae.encode(clean_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 

    

    for i in range(iters) :    
        data.requires_grad = True 
        idct = torch.clamp(dct.idct(data), min=-1, max=1)
        outputs = vae.encode(idct).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        loss = loss_function(outputs, clean_image_latents)
        # print(loss.item())

        grad = -1*torch.autograd.grad(loss, data)[0]
        
        adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
        eta = mask*torch.clamp(adv_images - generated_image_fft, min=-lamda, max=lamda)
        data = torch.clamp(generated_image_fft + eta, min=-100000, max=100000).detach_()

    return torch.clamp(dct.idct(data), min=-1, max=1), eta #torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real



def pgd_attack_fft_closest_lambda_l2(generated_image, clean_image_paths, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=1, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(generated_image) 
    generated_image_fft = fft#.real
    mask = torch.ones_like(generated_image_fft)
    mask2 = torch.tril(mask, cutoff)
    mask = torchvision.transforms.functional.hflip(mask2)
    
    data = Variable(generated_image_fft.data, requires_grad=True).to(device)

    loss_function = torch.nn.MSELoss()
    distances = []

    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor)
    with torch.no_grad():
        
        for clean_img_path in clean_image_paths:

            clean_img = load_clean_img(clean_img_path)
            clean_img = transform_img(clean_img).unsqueeze(0).to(vae.dtype).to(vae.device)
            if clean_img.shape[1] !=3:
                distances.append(float("Inf"))
                continue 
            
            clean_image_latents = vae.encode(clean_img).latent_dist.mode() * (1./vae.config.scaling_factor)
            distances.append(loss_function(generated_image_latents, clean_image_latents).item())

    clean_image_path = torch.argmin(torch.tensor(distances))
    clean_image = load_clean_img(clean_img_path)
    clean_image = transform_img(clean_image).unsqueeze(0).to(vae.dtype).to(vae.device)
    
    clean_image.requires_grad = False 
    vae.requires_grad = False
    clean_image_latents = vae.encode(clean_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 

    
    for i in range(iters) :    
        data.requires_grad = True 
        outputs = vae.encode(data).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        outputs.retain_grad()
        loss = torch.nn.functional.mse_loss(outputs, clean_image_latents) + lamda * torch.nn.functional.mse_loss(data, generated_image)
        
        grad = -1*torch.autograd.grad(loss, data)[0]

        data = data + alpha*grad #data.grad.sign()
        # eta = torch.clamp(adv_images - generated_image, min=-lamda, max=lamda)
        data = torch.clamp(data, min=-1, max=1).detach_()
    

    return data, None



def pgd_attack_lamda(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, delta=0, wave='haar', **kwargs):
    
    vae.requires_grad = False 
    data = Variable(clean_img.data, requires_grad=True).to(device)
    generated_image.requires_grad = False 
    vae.requires_grad = False
    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 

    if delta >0:
        lowFre_loss = torch.nn.SmoothL1Loss(reduction='sum')
        dwt = DWT_2D_tiny(wavename= wave)
        idwt = IDWT_2D_tiny(wavename= wave)
    
        inputs_ll = dwt(clean_img.float())
        inputs_ll = idwt(inputs_ll).half()
        

    for i in range(iters) :    
        data.requires_grad = True 
        outputs = vae.encode(data).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        outputs.retain_grad()
        loss = torch.nn.functional.mse_loss(outputs, generated_image_latents) + lamda * torch.nn.functional.mse_loss(data, clean_img)
        if delta>0:
                
            adv_ll = dwt(data.float())
            adv_ll = idwt(adv_ll).half()
    
            loss2 = lowFre_loss(adv_ll, inputs_ll)
        
            loss = loss + delta*loss2 
        
        grad = -1*torch.autograd.grad(loss, data)[0]

        data = data + alpha*grad #data.grad.sign()
        # eta = torch.clamp(adv_images - clean_img, min=-lamda, max=lamda)
        data = torch.clamp(data, min=-1, max=1).detach_()
        
    return data, clean_img - data
