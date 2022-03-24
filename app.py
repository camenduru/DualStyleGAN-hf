#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys
import tarfile
from typing import Callable

if os.environ.get('SYSTEM') == 'spaces':
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/fused_act.py")
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/upfirdn2d.py")

sys.path.insert(0, 'DualStyleGAN')

import dlib
import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from model.dualstylegan import DualStyleGAN
from model.encoder.align_all_parallel import align_face
from model.encoder.psp import pSp
from util import load_image, visualize

TOKEN = os.environ['TOKEN']

MODEL_REPO = 'hysts/DualStyleGAN'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def download_cartoon_images() -> None:
    image_dir = pathlib.Path('cartoon')
    if not image_dir.exists():
        path = huggingface_hub.hf_hub_download('hysts/DualStyleGAN-Cartoon',
                                               'cartoon.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()


def load_encoder(device: torch.device) -> nn.Module:
    ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                'models/encoder.pt',
                                                use_auth_token=TOKEN)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    opts = ckpt['opts']
    opts['device'] = 'cpu'
    opts['checkpoint_path'] = ckpt_path
    opts = argparse.Namespace(**opts)
    model = pSp(opts)
    model.to(device)
    model.eval()
    return model


def load_generator(style_type: str, device: torch.device) -> nn.Module:
    model = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    ckpt_path = huggingface_hub.hf_hub_download(
        MODEL_REPO, f'models/{style_type}/generator.pt', use_auth_token=TOKEN)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['g_ema'])
    model.to(device)
    model.eval()
    return model


def load_exstylecode(style_type: str) -> dict[str, np.ndarray]:
    if style_type in ['cartoon', 'caricature', 'anime']:
        filename = 'refined_exstyle_code.npy'
    else:
        filename = 'exstyle_code.npy'
    path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                           f'models/{style_type}/{filename}',
                                           use_auth_token=TOKEN)
    exstyles = np.load(path, allow_pickle=True).item()
    return exstyles


def create_transform() -> Callable:
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform


def create_dlib_landmark_model():
    path = huggingface_hub.hf_hub_download(
        'hysts/dlib_face_landmark_model',
        'shape_predictor_68_face_landmarks.dat',
        use_auth_token=TOKEN)
    return dlib.shape_predictor(path)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)


def postprocess(tensor: torch.Tensor) -> PIL.Image.Image:
    tensor = denormalize(tensor)
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    return PIL.Image.fromarray(image)


@torch.inference_mode()
def run(
    image,
    style_id: int,
    dlib_landmark_model,
    encoder: nn.Module,
    generator: nn.Module,
    exstyles: dict[str, np.ndarray],
    transform: Callable,
    device: torch.device,
    style_image_dir: pathlib.Path,
) -> tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    stylename = list(exstyles.keys())[style_id]

    image = align_face(filepath=image.name, predictor=dlib_landmark_model)
    input_data = transform(image).unsqueeze(0).to(device)

    img_rec, instyle = encoder(input_data,
                               randomize_noise=False,
                               return_latents=True,
                               z_plus_latent=True,
                               return_z_plus_latent=True,
                               resize=False)
    img_rec = torch.clamp(img_rec.detach(), -1, 1)

    latent = torch.tensor(exstyles[stylename]).repeat(2, 1, 1)
    # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
    latent[1, 7:18] = instyle[0, 7:18]
    exstyle = generator.generator.style(
        latent.reshape(latent.shape[0] * latent.shape[1],
                       latent.shape[2])).reshape(latent.shape)

    img_gen, _ = generator([instyle.repeat(2, 1, 1)],
                           exstyle,
                           z_plus_latent=True,
                           truncation=0.7,
                           truncation_latent=0,
                           use_res=True,
                           interp_weights=[0.6] * 7 + [1] * 11)
    img_gen = torch.clamp(img_gen.detach(), -1, 1)
    # deactivate color-related layers by setting w_c = 0
    img_gen2, _ = generator([instyle],
                            exstyle[0:1],
                            z_plus_latent=True,
                            truncation=0.7,
                            truncation_latent=0,
                            use_res=True,
                            interp_weights=[0.6] * 7 + [0] * 11)
    img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

    img_rec = postprocess(img_rec[0])
    img_gen0 = postprocess(img_gen[0])
    img_gen1 = postprocess(img_gen[1])
    img_gen2 = postprocess(img_gen2[0])

    style_image = PIL.Image.open(style_image_dir / stylename)

    return image, style_image, img_rec, img_gen0, img_gen1, img_gen2


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    style_type = 'cartoon'
    style_image_dir = pathlib.Path(style_type)

    download_cartoon_images()
    dlib_landmark_model = create_dlib_landmark_model()
    encoder = load_encoder(device)
    generator = load_generator(style_type, device)
    exstyles = load_exstylecode(style_type)
    transform = create_transform()

    func = functools.partial(run,
                             dlib_landmark_model=dlib_landmark_model,
                             encoder=encoder,
                             generator=generator,
                             exstyles=exstyles,
                             transform=transform,
                             device=device,
                             style_image_dir=style_image_dir)
    func = functools.update_wrapper(func, run)

    repo_url = 'https://github.com/williamyang1991/DualStyleGAN'
    title = 'williamyang1991/DualStyleGAN'
    description = f'A demo for {repo_url}'
    article = None

    image_paths = sorted(pathlib.Path('images').glob('*'))
    examples = [[path.as_posix(), 26] for path in image_paths]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='file', label='Image'),
            gr.inputs.Slider(0, 316, step=1, default=26, label='Style'),
        ],
        [
            gr.outputs.Image(type='pil', label='Aligned face'),
            gr.outputs.Image(type='pil', label='Style'),
            gr.outputs.Image(type='pil', label='Reconstructed'),
            gr.outputs.Image(type='pil', label='Gen 1'),
            gr.outputs.Image(type='pil', label='Gen 2'),
            gr.outputs.Image(type='pil', label='Gen 3'),
        ],
        examples=examples,
        theme=args.theme,
        title=title,
        description=description,
        article=article,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
