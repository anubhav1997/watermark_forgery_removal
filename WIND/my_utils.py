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

from utils import *
from io_utils import *
import torchvision.transforms as tforms

import math
import numpy as np
import scipy
from scipy import stats
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

def get_random_latents(pipe, img_size=512):
    
    try:
        init_latents = pipe.prepare_latents(1, pipe.unet.in_channels,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, None)
    except:
        init_latents = pipe.prepare_latents(1, pipe.transformer.in_channels,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, None)
    return init_latents

def get_pattern(pipe, shape=None, w_seed=999999, img_size=512, ring_value_range=64, quantization_levels=2, fix_gt=1, time_shift=1, trials=100, M=10000):
    
    base_latents = get_random_latents(pipe, img_size)
    base_latents = base_latents.to(torch.float64)
    original_latents_shape = base_latents.shape
    sing_channel_ring_watermark_mask = torch.tensor(
            ring_mask(
                size = original_latents_shape[-1], 
                r_out = RADIUS, 
                r_in = RADIUS_CUTOFF)
            )
    
    # get heterogeneous watermark mask
    if len(HETER_WATERMARK_CHANNEL) > 0:
        single_channel_heter_watermark_mask = torch.tensor(
                ring_mask(
                    size = original_latents_shape[-1], 
                    r_out = RADIUS, 
                    r_in = RADIUS_CUTOFF)  # TODO: change to whole mask
                )
        heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device)

    watermark_region_mask = []
    for channel_idx in WATERMARK_CHANNEL:
        if channel_idx in RING_WATERMARK_CHANNEL:
            watermark_region_mask.append(sing_channel_ring_watermark_mask)
        else:
            watermark_region_mask.append(single_channel_heter_watermark_mask)
    watermark_region_mask = torch.stack(watermark_region_mask).to(device)  # [C, 64, 64]

    # ###### Make RingID pattern
    single_channel_num_slots = RADIUS - RADIUS_CUTOFF
    key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-ring_value_range, ring_value_range, quantization_levels).tolist(), repeat = len(RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
    key_value_combinations = list(itertools.product(*key_value_list))

    # # random select from all possible value combinations, then generate patterns for selected ones.
    # if args.assigned_keys > 0:
    #     assert args.assigned_keys <= len(key_value_combinations)
    #     key_value_combinations = random.sample(key_value_combinations, k=args.assigned_keys)
        
    Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(device, list(combo), base_latents, radius=RADIUS, radius_cutoff=RADIUS_CUTOFF, ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL, heter_watermark_region_mask=heter_watermark_region_mask if len(HETER_WATERMARK_CHANNEL)>0 else None) for _, combo in enumerate(key_value_combinations)]            

    ring_capacity = len(Fourier_watermark_pattern_list)

    if fix_gt:
        Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in Fourier_watermark_pattern_list]
    
    if time_shift:
        for Fourier_watermark_pattern in Fourier_watermark_pattern_list:
            # Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * args.time_shift_factor)
            Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))
    
    # key_indices_to_evaluate = np.random.choice(ring_capacity, size = trials, replace = True).tolist()

    # Fourier_watermark_pattern = Fourier_watermark_pattern_list[628] # 628 was the value used by the original authors 
    
    # return Fourier_watermark_pattern, watermark_region_mask
    
    watermarker = WatermarkClass(device, M, ring_capacity, Fourier_watermark_pattern_list, watermark_region_mask, pipe)

    return watermarker

    
def get_noise(pipe, Fourier_watermark_pattern, watermark_region_mask, img_size=512):
    
    # key_index = key_indices_to_evaluate[prompt_index]

    # this_seed = args.general_seed + prompt_index
    # this_prompt = dataset[prompt_index][prompt_key]

    # set_random_seed(this_seed)
    no_watermark_latents = get_random_latents(pipe, img_size)
    
    Fourier_watermark_latents = generate_Fourier_watermark_latents(
        device = device,
        radius = RADIUS, 
        radius_cutoff = RADIUS_CUTOFF, 
        original_latents = no_watermark_latents, 
        watermark_pattern = Fourier_watermark_pattern, 
        watermark_channel = WATERMARK_CHANNEL,
        watermark_region_mask = watermark_region_mask,
    )
    return Fourier_watermark_latents



def detect(Fourier_watermark_image_distorted, pipe, Fourier_watermark_pattern, watermark_region_mask, channel = WATERMARK_CHANNEL, img_size=512):

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
    
    # Fourier_watermark_image_latents = pipe.get_image_latents(Fourier_watermark_image_distorted, sample = False)  # [N, c, h, w]

    # Fourier_watermark_reconstructed_latents = pipe.forward_diffusion(
    #         latents=Fourier_watermark_image_latents,
    #         text_embeddings=torch.cat([""] * len(Fourier_watermark_image_latents)),
    #         guidance_scale=1,
    #         num_inference_steps=args.test_num_inference_steps,
    #     )

    Fourier_watermark_reconstructed_latents_fft = fft(Fourier_watermark_reconstructed_latents)  # [N，c, h, w]

    inverted_latents_fft = Fourier_watermark_reconstructed_latents_fft[0][channel][watermark_region_mask].flatten()
    target = Fourier_watermark_pattern[0][channel][watermark_region_mask].flatten()

    inverted_latents_fft = torch.concatenate([inverted_latents_fft.real, inverted_latents_fft.imag])
    target = torch.concatenate([target.real, target.imag])
    
    # return Fourier_watermark_reconstructed_latents_fft

    sigma = inverted_latents_fft.std()
    lamda = (target ** 2 / sigma ** 2).sum().item()
    x = (((inverted_latents_fft - target) / sigma) ** 2).sum().item()
    p_value = stats.ncx2.cdf(x=x, df=len(target), nc=lamda)

    # revert scheduler
    pipe.scheduler = curr_scheduler

    return p_value 



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



class WatermarkClass:
    def __init__(self, device, M, N, Fourier_watermark_pattern_list, watermark_region_mask, pipe):
        # self.args = args
        self.device = device
        self.M = M
        self.N = N
        self.Fourier_watermark_pattern_list = Fourier_watermark_pattern_list
        self.watermark_region_mask = watermark_region_mask
        self.watermarked_latents = {}
        self.salt = "ekofijorfgjirejoiconime"
        # self.text_embeddings = pipe.get_text_embedding('')


    def _generate_hash(self, m):
        n = m % self.N
        hash_input = f"{m}{self.salt}{n}{(m % n + 1) if n != 0 else random.randint(1, 100)}".encode('utf-8')
        hash_object = hashlib.sha256(hash_input)
        hash_output = int.from_bytes(hash_object.digest(), byteorder='big') % (2**32 - 1)
        return hash_output, n


    def encode_and_generate(self, pipe, prompt, iteration):
        m = random.randint(1, self.M)
        hash_value, n = self._generate_hash(m)
        
        # Use hash as seed to generate noise
        torch.manual_seed(hash_value)
        torch.cuda.manual_seed_all(hash_value)
        noise = torch.randn(4, 64, 64, device=self.device, dtype=torch.float16)
        
        # Add the n-th key/ring pattern to the noise
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            device=self.device,
            radius=RADIUS,
            radius_cutoff=RADIUS_CUTOFF,
            original_latents=noise.unsqueeze(0),
            watermark_pattern=self.Fourier_watermark_pattern_list[n],
            watermark_channel=WATERMARK_CHANNEL,
            watermark_region_mask=self.watermark_region_mask,
        )
        
        encoded_noise = Fourier_watermark_latents.to(dtype=pipe.unet.dtype, device=pipe.device)
        
        self.watermarked_latents[(n, m)] = encoded_noise.cpu()
        
        with torch.no_grad():
            image = pipe(prompt, latents=encoded_noise, num_inference_steps=50).images[0]
        return image, n, m, encoded_noise, self.Fourier_watermark_pattern_list[n], self.watermark_region_mask


        
    def detect_watermark_stage1(self, pipe, image):
        # Extract latent representation
        image_tensor = transform_img(image).unsqueeze(0).to(device=self.device, dtype=pipe.vae.dtype)
        image_latents = pipe.get_image_latents(image_tensor, sample=False)

        # Reverse diffusion process
        reversed_latent = pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.args.test_num_inference_steps,
        )

        # Identify the correct ring (key)
        min_key_distance = float('inf')
        detected_n = -1
        reversed_latent_fft = fft(reversed_latent)

        for n in range(self.N):
            key_pattern = self.Fourier_watermark_pattern_list[n]
            
            distance = get_distance(
                key_pattern, 
                reversed_latent_fft, 
                self.watermark_region_mask,
                p=2,  # Using L2 norm
                mode='complex',
                channel_min=self.args.channel_min,
                channel=WATERMARK_CHANNEL
            )
            
            if distance < min_key_distance:
                min_key_distance = distance
                detected_n = n

        return detected_n, reversed_latent
    

    def detect_and_evaluate(self, pipe, image):
        detected_n, reversed_latent = self.detect_watermark_stage1(pipe, image)

        min_distance = float('inf')
        detected_m = -1
        rotation_angles = [0,74,76] # The method is robust to any degree of rotation by checking rotation patterns like 2n or 4n+1.

        # Search among the group with the same ring (key)
        for (n, m), watermarked_latent in self.watermarked_latents.items():
            if n == detected_n:
                watermarked_latent = watermarked_latent.to(self.device)
                for angle in rotation_angles:
                    watermarked_latent = transforms.RandomRotation((angle, angle))(watermarked_latent)
                    distance = torch.norm(reversed_latent - watermarked_latent)
                    if distance < min_distance:
                        min_distance = distance
                        detected_m = m


        true_distance = min_distance

        # Random distances based on L2 norms
        random_distances = [torch.norm(reversed_latent - torch.randn_like(reversed_latent)).item() for _ in range(100)]


        return detected_n, detected_m, true_distance, random_distances



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

