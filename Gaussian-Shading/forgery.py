# import all the libraries
import math
import numpy as np
import scipy
from PIL import Image
import torch
import torchvision.transforms as tforms
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline #, StableDiffusion3Pipeline, FluxTransformer2DModel, Transformer2DModel, FluxPipeline, PixArtSigmaPipeline
# from diffusers.models import AutoencoderKL
# import gradio as gr
# import torch.nn as nn 
import torchvision
from torch.autograd import Variable 
import torch_dct as dct
# from DWT import *
# from utils import * 
# from datasets import load_dataset
import pandas as pd 
import os 
import glob 
import random 
# import pandas as pd
import argparse
from my_utils import * 
import hashlib  
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *

def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",#"stabilityai/stable-diffusion-2-base"#"stabilityai/stable-diffusion-3-medium-diffusers", #"stabilityai/stable-diffusion-2",#"/scratch/aj3281/DCR/DCR/sd-finetuned-org_parameters_instancelevel_blip_nodup_laion/checkpoint/",
    )
    parser.add_argument(
        "--vae_optimization",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="coco_adv_sd1_sd1",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DPM",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="an astronaut riding a green horse",
    )
    parser.add_argument(
        "--clean_img",
        type=str,
        default="/home/aj3281/scratch/diffusers/examples/text_to_image/coco_org_10k/43433.jpg",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5/255,
    )
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=15000,
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0,
    )
    # parser.add_argument(
    #     "--chacha",
    #     type=bool,
    #     default=False,
    # )
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--user_number', default=1000000, type=int)
    return parser.parse_args()


args = parse_args()

# Before you do anything else, seed everything. 

# Load model 
model_id = args.pretrained_model_name_or_path #"CompVis/stable-diffusion-v1-4"# "stabilityai/stable-diffusion-2-1" #
img_size = args.img_size #512 #1024

if args.vae_optimization is not None:
    vae_optimization = AutoencoderKL.from_pretrained(args.vae_optimization, subfolder="vae", torch_dtype=torch.float32).to("cuda")
else:
    vae_optimization = None 

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

if model_id == "CompVis/stable-diffusion-v1-4"  or model_id == "stabilityai/stable-diffusion-2-base":
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, vae=vae, unet=unet, torch_dtype=torch.float32) #
    shape = (1, pipe.unet.in_channels, img_size//8, img_size//8)
elif model_id=="stabilityai/stable-diffusion-xl-base-1.0":
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, unet=unet, torch_dtype=torch.float32) #
    shape = (1, pipe.unet.in_channels, img_size//8, img_size//8)
elif model_id == "PixArt-alpha/PixArt-Sigma-XL-2-512-MS":
    transformer = Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    pipe = PixArtSigmaPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.float16)
    shape = (1, pipe.transformer.in_channels, img_size//8, img_size//8)
elif model_id == "black-forest-labs/FLUX.1-dev":
    transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    pipe = FluxPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.float16)
    shape = (1, pipe.transformer.in_channels, img_size//8, img_size//8)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


w_channel = 0
w_radius = 16 # the suggested r from section 4.4 of paper

files = sorted(glob.glob('/scratch/aj3281/watermarking_project/tree-ring-watermark/coco_org_10k/*'))
# prompts = load_dataset("Gustavosta/Stable-Diffusion-Prompts")["test"]


# splits = {'train': 'data/train.parquet', 'test': 'data/eval.parquet'}
# prompts = pd.read_parquet("hf://datasets/Gustavosta/Stable-Diffusion-Prompts/" + splits["test"]).to_numpy()

prompts = pd.read_parquet("hf://datasets/yuvalkirstain/runwayml-stable-diffusion-v1-5-eval-random-prompts/data/train-00000-of-00001-e6b6f8777640f9fc.parquet")['prompt'].to_numpy()


# prompts = pd.read_csv("/scratch/aj3281/concept_erasure_project/Diffusion-MU-Attack/prompts/coco_10k.csv")

asr = 0 
avg = 0 
total = 0

i = args.start_iter 


while i < args.end_iter:

    file = files[i]
    
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    generator = torch.manual_seed(seed) #.to(torch_device)  # Seed generator to create the initial latent noise
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    w_seed = seed #7433 # TREE :)
    
    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        #a simple implement,
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    
    init_latents_w = watermark.create_watermark_and_return_w()
    print(init_latents_w.dtype)
    generated_image = pipe(prompt=prompts[i][0], num_inference_steps=50, latents=init_latents_w.to(vae.dtype)).images[0]
    # pipe = pipe.to(vae.dtype)
    # generated_image = generated_image.to(vae.dtype)
    gen_img_score = watermark.eval_watermark(get_reversed_w(pipe, generated_image))
    
    # gen_img_score = detect(generated_image, pipe, w_key, w_mask, img_size=img_size)
        
    print("p value for generated image", gen_img_score)
    # print(generated_image.shape, "generated image shape", torch.max(generated_image))
    clean_img = load_clean_img(file)
    generated_image = transform_img(generated_image).unsqueeze(0).to(vae.dtype).to(pipe.device)
    

    clean_img = transform_img(clean_img).unsqueeze(0).to(vae.dtype).to(pipe.device)
    
    if clean_img.shape[1] !=3:
        i+=1
        continue 
        
    clean_img_initial = clean_img.detach() 
    # save_img(clean_img, "org_img2.png")

    # init_p_val = detect(clean_img, pipe, w_key, w_mask, img_size=img_size)

    
    init_p_val = watermark.eval_watermark(get_reversed_w(pipe, clean_img))
    
    print("initial p value", init_p_val)
    # clean_image_latents = pipe.vae.encode(clean_img).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    
    n_iters = args.n_iters #1000
    lamda=args.lamda
    alpha=args.alpha#5/255
    
    loss_function = torch.nn.MSELoss()
    
    if vae_optimization is not None:
        clean_img, adv_noise = pgd_attack_lamda(clean_img, generated_image, vae_optimization, lamda=lamda, alpha=alpha, iters=n_iters, cutoff=args.cutoff, delta=args.delta)
    else:
        clean_img, adv_noise = pgd_attack_lamda(clean_img, generated_image, vae, lamda=lamda, alpha=alpha, iters=n_iters, cutoff=args.cutoff, delta=args.delta)
    # torchvision.utils.save_image(clean_img, 'adv_img.jpg')
    final_p = watermark.eval_watermark(get_reversed_w(pipe, clean_img))
    print("final p value", final_p)
    save_img(clean_img, f"{args.outdir}/{i}_{gen_img_score}_{init_p_val}_{final_p}.png")

    asr += int(final_p <= 0.05)
    avg += final_p
    total+=1 
    
    print(asr/float(total))
    print(avg/float(total))
    print(total)
    i+=1 
    
    
print(asr/float(total))
print(avg/float(total))
print(total)

