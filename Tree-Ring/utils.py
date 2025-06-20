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
# from DWT import *
import torchvision
import matplotlib.pyplot as plt 
from scipy import stats

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# watermarking helper functions. paraphrased from the reference impl of arXiv:2305.20030

def circle_mask(size=128, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_pattern(shape, pipe, w_seed=999999, img_size=512):
    g = torch.Generator(device=pipe.device)
    g.manual_seed(w_seed)
    try:
        gt_init = pipe.prepare_latents(1, pipe.unet.in_channels,
                                       img_size, img_size,
                                       pipe.vae.dtype, pipe.device, g)
    except:
        gt_init, _ = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, g)
        print(gt_init.shape)
    # print(gt_init[0].shape, gt_init[1].shape, pipe.transformer.in_channels )
    gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
    # ring pattern. paper found this to be effective
    gt_patch_tmp = gt_patch.clone().detach()
    for i in range(shape[-1] // 2, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask)
        for j in range(gt_patch.shape[1]):
            gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def transform_img(image, img_size=512):
    tform = tforms.Compose([tforms.Resize(img_size),tforms.CenterCrop(img_size),tforms.ToTensor()])
    image = tform(image)
    return 2.0 * image - 1.0


def get_noise(pipe, w_key, w_mask, img_size=512):
    # moved w_key and w_mask to globals

    # inject watermark
    try:
            
        init_latents = pipe.prepare_latents(1, pipe.unet.in_channels,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, None)
        latent_image_ids = None
        
    except:

        
        init_latents, latent_image_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, None)
        
    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    # print(init_latents.shape, w_mask.shape, w_key.shape)
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real
    # hot fix to prevent out of bounds values. will "properly" fix this later
    init_latents[init_latents == float("Inf")] = 4
    init_latents[init_latents == float("-Inf")] = -4

    return init_latents, latent_image_ids


def detect(image, pipe, w_key, w_mask, img_size=512):
    # invert scheduler
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    # ddim inversion
    if not torch.is_tensor(image):
        img = transform_img(image, img_size).unsqueeze(0).to(pipe.vae.dtype).to(pipe.device)
    else:
        img = image 

    print("here inside detect", img.shape)
    
    image_latents = pipe.vae.encode(img).latent_dist.mode() * (1./pipe.vae.config.scaling_factor) #0.13025
    # print(image_latents)
    
    inverted_latents = pipe(prompt="", latents=image_latents.to(pipe.vae.dtype), guidance_scale=1, num_inference_steps=50, output_type="latent")
    inverted_latents = inverted_latents.images
    # print(inverted_latents)

    # calculate p-value instead of detection threshold. more rigorous, plus we can do a non-boolean output
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))[w_mask].flatten()
    target = w_key[w_mask].flatten()
    inverted_latents_fft = torch.concatenate([inverted_latents_fft.real, inverted_latents_fft.imag])
    target = torch.concatenate([target.real, target.imag])

    sigma = inverted_latents_fft.std()
    lamda = (target ** 2 / sigma ** 2).sum().item()
    # print(sigma, lamda)
    x = (((inverted_latents_fft - target) / sigma) ** 2).sum().item()
    p_value = stats.ncx2.cdf(x=x, df=len(target), nc=lamda)

    # revert scheduler
    pipe.scheduler = curr_scheduler

    return p_value 
    


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        # reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        # reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        # no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')
    
    return w_metric


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    # reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    # reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    # sigma_no_w = reversed_latents_no_w_fft.std()
    # lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    # x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    # p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_w


def generate(prompt, pipe, w_key, w_mask, img_size=512):
    latent, latent_image_ids = get_noise(pipe, w_key, w_mask, img_size)
    
    # print(latent.shape, "HEREEEE", prompt, pipe)
    
    if latent_image_ids is not None:
        img = pipe(prompt=prompt, negative_prompt="", num_inference_steps=50, latents=latent).images[0]
    else:    
        img = pipe(prompt=prompt, negative_prompt="", num_inference_steps=50, latents=latent).images[0]
    # print(img.size)
    # print(max(img), min(img))
    return img 


def generate_no_watermark(prompt, pipe, img_size=512):
    # latent, latent_image_ids = get_noise(pipe, w_key, w_mask, img_size)
    init_latents = pipe.prepare_latents(1, pipe.unet.in_channels,
                                            img_size, img_size,
                                            pipe.vae.dtype, pipe.device, None)
    
    # if latent_image_ids is not None:
    #     img = pipe(prompt=prompt, negative_prompt="", num_inference_steps=50, latents=latent).images[0]
    # else:    
    
    img = pipe(prompt=prompt, negative_prompt="", num_inference_steps=50, latents=init_latents).images[0]
    # print(img.size)
    # print(max(img), min(img))
    return img 
    

def load_clean_img(filename):
    img = Image.open(filename)
    return img





def pgd_attack_fft_clip(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=1, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(clean_img) #torch.fft.fft(clean_img)
    clean_img_fft = fft#.real

    mask = torch.ones_like(clean_img_fft)
    mask2 = torch.tril(mask, cutoff)
    mask = torchvision.transforms.functional.hflip(mask2)
    
    # mask = get_mask(clean_img_fft, cutoff)

    # print(torch.testing.assert_close(fft, torch.complex(clean_img_fft, clean_img_fft_imag)))
    data = Variable(clean_img_fft.data, requires_grad=True).to(device)
    generated_image.requires_grad = False 
    vae.requires_grad = False
    generated_image_latents = vae.encode_image(generated_image) #.latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
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
        outputs = vae.encode_image(idct) #.latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
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




def pgd_attack_fft(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=0, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(clean_img) #torch.fft.fft(clean_img)
    clean_img_fft = fft#.real

    # mask = get_mask(clean_img_fft, cutoff)

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

    
    for i in range(iters) :    
        data.requires_grad = True 
        idct = torch.clamp(dct.idct(data), min=-1, max=1)
        # outputs = vae.encode(torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        outputs = vae.encode(idct).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        # outputs.retain_grad()
        # loss = torch.nn.functional.mse_loss(outputs, generated_image_latents)
        loss = loss_function(outputs, generated_image_latents)
        
        grad = -1*torch.autograd.grad(loss, data)[0]
        
        adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
        eta = mask*torch.clamp(adv_images - clean_img_fft, min=-lamda, max=lamda)
        # data = torch.clamp(clean_img_fft + eta, min=-1, max=1).detach_()
        data = torch.clamp(clean_img_fft + eta, min=-100000, max=100000).detach_()

    
    return torch.clamp(dct.idct(data), min=-1, max=1), eta #torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real






def pgd_attack_fft_multi(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=0, wave='haar'):
  
    vae.requires_grad = False 

    fft = dct.dct(clean_img) #torch.fft.fft(clean_img)
    clean_img_fft = fft#.real

    # mask = get_mask(clean_img_fft, cutoff)

    mask = torch.ones_like(clean_img_fft)
    mask2 = torch.tril(mask, cutoff)
    mask = torchvision.transforms.functional.hflip(mask2)

    
    # print(torch.testing.assert_close(fft, torch.complex(clean_img_fft, clean_img_fft_imag)))
    data = Variable(clean_img_fft.data, requires_grad=True).to(device)
    generated_image.requires_grad = False 
    vae.requires_grad = False

    # generated_image_latents = []

    # for i in range(len)
    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 

    loss_function = torch.nn.MSELoss()

    
    for i in range(iters) :    
        data.requires_grad = True 
        idct = torch.clamp(dct.idct(data), min=-1, max=1)
        # outputs = vae.encode(torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        outputs = vae.encode(idct).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
        # outputs.retain_grad()
        # loss = torch.nn.functional.mse_loss(outputs, generated_image_latents)

        outputs = outputs.repeat(len(generated_image_latents), 1, 1, 1)
        
        loss = torch.sum(loss_function(outputs, generated_image_latents))
        
        grad = -1*torch.autograd.grad(loss, data)[0]
        
        adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
        eta = mask*torch.clamp(adv_images - clean_img_fft, min=-lamda, max=lamda)
        # data = torch.clamp(clean_img_fft + eta, min=-1, max=1).detach_()
        data = torch.clamp(clean_img_fft + eta, min=-100000, max=100000).detach_()

    
    return torch.clamp(dct.idct(data), min=-1, max=1), eta #torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real



def pgd_attack(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, delta=0, wave='haar', **kwargs):
    
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
        loss = torch.nn.functional.mse_loss(outputs, generated_image_latents)
        if delta>0:
                
            adv_ll = dwt(data.float())
            adv_ll = idwt(adv_ll).half()
    
            loss2 = lowFre_loss(adv_ll, inputs_ll)
        
            loss = loss + delta*loss2 

        grad = -1*torch.autograd.grad(loss, data)[0]

        adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
        eta = torch.clamp(adv_images - clean_img, min=-lamda, max=lamda)
        data = torch.clamp(clean_img + eta, min=-1, max=1).detach_()
            
    return data


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
            
    return data

import lpips
def pgd_attack_lpips(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, delta=0, wave='haar', **kwargs):
    
    vae.requires_grad = False 
    data = Variable(clean_img.data, requires_grad=True).to(device)
    generated_image.requires_grad = False 
    vae.requires_grad = False
    generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
    # generated_image_latents.requires_grad = False 
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
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
        loss = torch.nn.functional.mse_loss(outputs, generated_image_latents) + lamda * loss_fn_alex(data, clean_img)
        if delta>0:
                
            adv_ll = dwt(data.float())
            adv_ll = idwt(adv_ll).half()
    
            loss2 = lowFre_loss(adv_ll, inputs_ll)
        
            loss = loss + delta*loss2 
        
        grad = -1*torch.autograd.grad(loss, data)[0]

        data = data + alpha*grad #data.grad.sign()
        # eta = torch.clamp(adv_images - clean_img, min=-lamda, max=lamda)
        data = torch.clamp(data, min=-1, max=1).detach_()
            
    return data



def pgd_attack3(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, delta=0, wave='haar', **kwargs):
    
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

        data = data + alpha*torch.sign(grad) #data.grad.sign()
        # eta = torch.clamp(adv_images - clean_img, min=-lamda, max=lamda)
        data = torch.clamp(data, min=-1, max=1).detach_()
            
    return data


def save_img(clean_img, out_path):
    clean_img = (clean_img / 2 + 0.5).clamp(0, 1).squeeze()
    clean_img = (clean_img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    clean_img = Image.fromarray(clean_img)
    clean_img.save(out_path) 



def get_mask(generated_image_fft, cutoff):

    mask = torch.ones_like(generated_image_fft)
    
    print(mask.shape)
    cut = 512  - cutoff
    
    for i in range(mask.shape[2]):
        for j in range(mask.shape[3]):
            if (i**2 + j**2)**0.5 < cut:
                for k in range(3):
                    mask[0][k][i][j] = 0
                # print(i, j, mask.shape, mask[0].shape, mask[0][:][0][0].shape)
                # print(torch.zeros_like(mask[:][:][i][j]).shape)
                # print(mask[:][:][i][j].shape)
                # mask[:][:][i][j] = torch.zeros_like(mask[:][:][i][j])

    save_img(mask, "mask_takashi.png")
    return mask



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



# def pgd_attack_fft_removal(clean_img, generated_image, vae, lamda=0.03, alpha=0.0001, iters=10, cutoff=100, delta=1, wave='haar'):
  
#     vae.requires_grad = False 

#     fft = dct.dct(clean_img) #torch.fft.fft(clean_img)
#     clean_img_fft = fft#.real
#     mask = torch.ones_like(clean_img_fft)
#     mask2 = torch.tril(mask, cutoff)
#     mask = torchvision.transforms.functional.hflip(mask2)

#     # print(torch.testing.assert_close(fft, torch.complex(clean_img_fft, clean_img_fft_imag)))
#     data = Variable(clean_img_fft.data, requires_grad=True).to(device)
#     generated_image.requires_grad = False 
#     vae.requires_grad = False
#     generated_image_latents = vae.encode(generated_image).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025
#     # generated_image_latents.requires_grad = False 

#     loss_function = torch.nn.MSELoss()
#     # lowFre_loss = nn.SmoothL1Loss(reduction='sum')

#     if delta != 0:
            
#         lowFre_loss = torch.nn.SmoothL1Loss(reduction='sum')
#         dwt = DWT_2D_tiny(wavename= wave)
#         idwt = IDWT_2D_tiny(wavename= wave)
#         inputs_ll = dwt(clean_img.float())
#         inputs_ll = idwt(inputs_ll).half()
    
#     for i in range(iters) :    
#         data.requires_grad = True 
#         idct = torch.clamp(dct.idct(data), min=-1, max=1)
#         # outputs = vae.encode(torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
#         outputs = vae.encode(idct).latent_dist.mode() * (1./vae.config.scaling_factor) #0.13025  #model(images)
#         # outputs.retain_grad()
#         # loss = torch.nn.functional.mse_loss(outputs, generated_image_latents)
#         loss = loss_function(outputs, generated_image_latents)
#         # print(loss.item())

#         if delta!=0:
#             adv_ll = dwt(idct.float())
#             adv_ll = idwt(adv_ll).half()
    
#             loss2 = lowFre_loss(adv_ll, inputs_ll)
        
#             loss = loss + delta*loss2 
        
#         grad = torch.autograd.grad(loss, data)[0]
        
#         adv_images = data + alpha*torch.sign(grad) #data.grad.sign()
#         eta = mask*torch.clamp(adv_images - clean_img_fft, min=-lamda, max=lamda)
#         # data = torch.clamp(clean_img_fft + eta, min=-1, max=1).detach_()
#         data = torch.clamp(clean_img_fft + eta, min=-100000, max=100000).detach_()

    
#     return torch.clamp(dct.idct(data), min=-1, max=1), eta #torch.fft.ifft2(torch.complex(data, clean_img_fft_imag)).real
