# Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image

This repository contains the official codebase for the paper "Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image". We provide an approach to forge and remove latent noise based diffusion watermarks. You can read more about the approach in our paper [ArXiv](http://www.arxiv.org/pdf/2504.20111)

## Watermarking methods attacked: 

We specifically focus on the following latent-noise based watermarking schemes - 

1. [Tree-Ring](https://github.com/YuxinWenRick/tree-ring-watermark)
2. [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading)
3. [Ring-ID](https://github.com/showlab/RingID)
4. [WIND](https://github.com/Kasraarabi/Hidden-in-the-Noise)


## Running Attacks on Tree-Ring Watermarking Scheme

To run the forgery attack on SDv2.1 using the VAE from SDv1.4 use the following script. To update the VAE utilized for optimization you can update this in the parameter ```--vae_optimization```. You can update the model you want to attack using the parameter ```--pretrained_model_name_or_path```. We also utilize a hyper-parameter lambda in our optimization this can be controlled using the parameter ```--eps```. 

```
python3 Tree-Ring/forgery.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```

Similarly you can run the removal attack using the following command: 

```
python3 Tree-Ring/removal.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```


## Running Attacks on Gaussian Shading Watermarking Scheme

To run the forgery attack on the Gaussian Shading watermarking scheme, use the following command: 

```
python3 Gaussian-Shading/forgery.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```

Similarly you can run the removal attack using the following command: 

```
python3 Gaussian-Shading/removal.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```


## Running Attacks on RingID Watermarking Scheme


To run the forgery attack on the RingID watermarking scheme, use the following command: 

```
python3 Gaussian-Shading/forgery.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```

Similarly you can run the removal attack using the following command: 

```
python3 Gaussian-Shading/removal.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```



## Running Attacks on WIND Watermarking Scheme


To run the forgery attack on the WIND watermarking scheme, use the following command: 

```
python3 WIND/forgery.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```

Similarly you can run the removal attack using the following command: 

```
python3 WIND/removal.py --eps 1e4 --start_iter 0 --end_iter 200 --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-base" --vae_optimization "CompVis/stable-diffusion-v1-4" --outdir forged_images
```

