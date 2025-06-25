'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import argparse
import os

_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3: Zero-shot One Image to 3D Object'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)

            print("c", c.shape)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im

def generate_multi_view_images(models, sampler, output_dir, raw_im, device,
            azimuths, elevations, distances,
            preprocess=True, scale=3.0, n_samples=1, 
            ddim_steps=50, ddim_eta=1.0,
            precision='fp32', h=256, w=256):

    # preprocess PIL image
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    input_im = preprocess_image(models, raw_im, preprocess)
    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    print("input_im.shape", input_im.shape)

    # make output directory
    os.makedirs(output_dir, exist_ok=True)
    output_ims = [raw_im] # PIL Images

    for i in range(len(azimuths)):
        x, y, z = elevations[i],  azimuths[i], distances[i]

        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                      ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_pil = Image.fromarray(x_sample.astype(np.uint8))
            output_ims.append(output_pil)
            image_path = os.path.join(output_dir, f"image_{x}_{y}_{z}.png")
            output_pil.save(image_path)
            print(f"Saved image to {image_path}")

    return output_ims

@torch.no_grad()
def sample_model_with_multiple_inputs(models, cond_ims, device, scale=3.0, n_samples=1, ddim_steps=50, ddim_eta=1.0,
                                    precision='fp32', h=256, w=256):

    model = models['turncam']

    # Preprocessing pipeline (resize, tensor, normalize to [-1, 1])
    transform = transforms.Compose([
        transforms.Lambda(lambda im: im.thumbnail([1536, 1536], Image.Resampling.LANCZOS) or im),
        transforms.Lambda(lambda im: preprocess_image(models, im, preprocess=True)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2 - 1),
        transforms.Resize((h, w))
    ])

    # Apply to all conditioning images
    processed_images = [transform(im).to(device) for im in cond_ims]
    xc = torch.stack(processed_images, dim=0)  # shape: [B, 3, H, W]

    print("xc.shape of conditioning images", xc.shape)

    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            clip_emb = model.get_learned_conditioning(xc).tile(n_samples, 1, 1)
            
            print("clip_emb.shape", clip_emb.shape)

            Ts = []
            x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
            y = int(np.round(np.random.uniform(-150.0, 150.0)))
            z = 0.0
            for i in range(clip_emb.shape[0]):
                t = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
                Ts.append(t)
            T = torch.stack(Ts).repeat(n_samples, 1, 1).to(device)  # [B, 1, 4]
            T = T.permute(1, 0, 2)
            print("T.shape", T.shape)

            c = model.cc_projection(torch.cat([clip_emb, T], dim=-1))
            
            c = c.permute(1, 0, 2)

            print("c.shape", c.shape)
            print("model.encode_first_stage((xc.to(device))).mode().detach().repeat(n_samples, 1, 1, 1)", model.encode_first_stage((xc.to(device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1).shape)
            
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((xc.to(device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]

            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=cond,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu(), x, y, z

def load_png_images(folder_path, image_path, x):
    # Get list of all .png files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png') and f.lower().startswith('image')]
    
    # Check if there are at least x images
    if len(png_files) == x:
        # Load only the first x .png images
        images = [Image.open(image_path)]
        images += [Image.open(os.path.join(folder_path, png_files[i])) for i in range(x)]
        print(f"Loaded {x} previously generated images from {folder_path}")
        return images
    else:
        print(f"Only found {len(png_files)} PNG images, but expected {x}.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run model initialization.")
    parser.add_argument('--config', default="configs/sd-objaverse-finetune-c_concat-256.yaml", type=str, help="Path to the config file")
    parser.add_argument('--ckpt', default="105000.ckpt", type=str, help="Path to the model checkpoint")
    parser.add_argument('--device_idx', type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the input image")

    args = parser.parse_args()

    # setup device and config
    device = torch.device(f'cuda:{args.device_idx}' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(args.config)

    # read image
    print(f"Loading input image from: {args.image}")
    raw_im = Image.open(args.image).convert("RGB")

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, args.ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    sampler = DDIMSampler(models['turncam'])

    azimuths = [0, 45, -45]
    elevations = [0, 0, 0]
    distances = [0, 0, 0]

    cond_ims = load_png_images(args.output_dir, args.image, 3)
    if not cond_ims:
        cond_ims = generate_multi_view_images(models, sampler, args.output_dir, raw_im, device, azimuths, elevations, distances)
    x_samples_ddim, x, y, z = sample_model_with_multiple_inputs(models, cond_ims, device)
    
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_pil = Image.fromarray(x_sample.astype(np.uint8))
        output_ims.append(output_pil)
        image_path = os.path.join(args.output_dir, f"output_{x}_{y}_{z}.png")
        output_pil.save(image_path)
        print(f"Saved image to {image_path}")
    