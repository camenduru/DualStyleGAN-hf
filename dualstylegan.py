from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Callable

import dlib
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as T

if os.getenv('SYSTEM') == 'spaces':
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/fused_act.py")
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/upfirdn2d.py")

app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'DualStyleGAN'
sys.path.insert(0, submodule_dir.as_posix())

from model.dualstylegan import DualStyleGAN
from model.encoder.align_all_parallel import align_face
from model.encoder.psp import pSp

MODEL_REPO = 'CVPR/DualStyleGAN'


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.landmark_model = self._create_dlib_landmark_model()
        self.encoder_dict = self._load_encoder()
        self.transform = self._create_transform()
        self.encoder_type = 'z+'

        self.style_types = [
            'cartoon',
            'caricature',
            'anime',
            'arcane',
            'comic',
            'pixar',
            'slamdunk',
        ]
        self.generator_dict = {
            style_type: self._load_generator(style_type)
            for style_type in self.style_types
        }
        self.exstyle_dict = {
            style_type: self._load_exstylecode(style_type)
            for style_type in self.style_types
        }

    @staticmethod
    def _create_dlib_landmark_model():
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        path = pathlib.Path('shape_predictor_68_face_landmarks.dat')
        if not path.exists():
            bz2_path = 'shape_predictor_68_face_landmarks.dat.bz2'
            torch.hub.download_url_to_file(url, bz2_path)
            subprocess.run(f'bunzip2 -d {bz2_path}'.split())
        return dlib.shape_predictor(path.as_posix())

    def _load_encoder(self) -> nn.Module:
        ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                    'models/encoder.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts = argparse.Namespace(**opts)
        model = pSp(opts)
        model.to(self.device)
        model.eval()

        ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                    'models/encoder_wplus.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts['output_size'] = 1024
        opts = argparse.Namespace(**opts)
        model2 = pSp(opts)
        model2.to(self.device)
        model2.eval()

        return {'z+': model, 'w+': model2}

    @staticmethod
    def _create_transform() -> Callable:
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        return transform

    def _load_generator(self, style_type: str) -> nn.Module:
        model = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        ckpt_path = huggingface_hub.hf_hub_download(
            MODEL_REPO, f'models/{style_type}/generator.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['g_ema'])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _load_exstylecode(style_type: str) -> dict[str, np.ndarray]:
        if style_type in ['cartoon', 'caricature', 'anime']:
            filename = 'refined_exstyle_code.npy'
        else:
            filename = 'exstyle_code.npy'
        path = huggingface_hub.hf_hub_download(
            MODEL_REPO, f'models/{style_type}/{filename}')
        exstyles = np.load(path, allow_pickle=True).item()
        return exstyles

    def detect_and_align_face(self, image_path) -> np.ndarray:
        image = align_face(filepath=image_path, predictor=self.landmark_model)
        x, y = np.random.randint(255), np.random.randint(255)
        r, g, b = image.getpixel((x, y))
        image.putpixel(
            (x, y), (r, g + 1, b)
        )  # trick to make sure run reconstruct_face() once any input setting changes
        return image

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self.denormalize(tensor)
        return tensor.cpu().numpy().transpose(1, 2, 0)

    @torch.inference_mode()
    def reconstruct_face(self, image: np.ndarray,
                         encoder_type: str) -> tuple[np.ndarray, torch.Tensor]:
        if encoder_type == 'Z+ encoder (better stylization)':
            self.encoder_type = 'z+'
            z_plus_latent = True
            return_z_plus_latent = True
        else:
            self.encoder_type = 'w+'
            z_plus_latent = False
            return_z_plus_latent = False
        image = PIL.Image.fromarray(image)
        input_data = self.transform(image).unsqueeze(0).to(self.device)
        img_rec, instyle = self.encoder_dict[self.encoder_type](
            input_data,
            randomize_noise=False,
            return_latents=True,
            z_plus_latent=z_plus_latent,
            return_z_plus_latent=return_z_plus_latent,
            resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        img_rec = self.postprocess(img_rec[0])
        return img_rec, instyle

    @torch.inference_mode()
    def generate(self, style_type: str, style_id: int, structure_weight: float,
                 color_weight: float, structure_only: bool,
                 instyle: torch.Tensor) -> np.ndarray:

        if self.encoder_type == 'z+':
            z_plus_latent = True
            input_is_latent = False
        else:
            z_plus_latent = False
            input_is_latent = True

        generator = self.generator_dict[style_type]
        exstyles = self.exstyle_dict[style_type]

        style_id = int(style_id)
        stylename = list(exstyles.keys())[style_id]

        latent = torch.tensor(exstyles[stylename]).to(self.device)
        if structure_only and self.encoder_type == 'z+':
            latent[0, 7:18] = instyle[0, 7:18]
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0] * latent.shape[1],
                           latent.shape[2])).reshape(latent.shape)
        if structure_only and self.encoder_type == 'w+':
            exstyle[:, 7:18] = instyle[:, 7:18]

        img_gen, _ = generator([instyle],
                               exstyle,
                               input_is_latent=input_is_latent,
                               z_plus_latent=z_plus_latent,
                               truncation=0.7,
                               truncation_latent=0,
                               use_res=True,
                               interp_weights=[structure_weight] * 7 +
                               [color_weight] * 11)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        img_gen = self.postprocess(img_gen[0])
        return img_gen
