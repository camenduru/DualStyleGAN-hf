from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Union

import dlib
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.transforms as T

if os.environ.get('SYSTEM') == 'spaces':
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/fused_act.py")
    os.system("sed -i '10,17d' DualStyleGAN/model/stylegan/op/upfirdn2d.py")

sys.path.insert(0, 'DualStyleGAN')

from model.dualstylegan import DualStyleGAN
from model.encoder.align_all_parallel import align_face
from model.encoder.psp import pSp

HF_TOKEN = os.environ['HF_TOKEN']
MODEL_REPO = 'hysts/DualStyleGAN'


class Model:
    def __init__(self, device: Union[torch.device, str]):
        self.device = torch.device(device)
        self.landmark_model = self._create_dlib_landmark_model()
        self.encoder = self._load_encoder()
        self.transform = self._create_transform()

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
        path = huggingface_hub.hf_hub_download(
            'hysts/dlib_face_landmark_model',
            'shape_predictor_68_face_landmarks.dat',
            use_auth_token=HF_TOKEN)
        return dlib.shape_predictor(path)

    def _load_encoder(self) -> nn.Module:
        ckpt_path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                                    'models/encoder.pt',
                                                    use_auth_token=HF_TOKEN)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        opts = ckpt['opts']
        opts['device'] = self.device.type
        opts['checkpoint_path'] = ckpt_path
        opts = argparse.Namespace(**opts)
        model = pSp(opts)
        model.to(self.device)
        model.eval()
        return model

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
            MODEL_REPO,
            f'models/{style_type}/generator.pt',
            use_auth_token=HF_TOKEN)
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
            MODEL_REPO,
            f'models/{style_type}/{filename}',
            use_auth_token=HF_TOKEN)
        exstyles = np.load(path, allow_pickle=True).item()
        return exstyles

    def detect_and_align_face(self, image) -> np.ndarray:
        image = align_face(filepath=image.name, predictor=self.landmark_model)
        return image

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp((tensor + 1) / 2 * 255, 0, 255).to(torch.uint8)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self.denormalize(tensor)
        return tensor.cpu().numpy().transpose(1, 2, 0)

    @torch.inference_mode()
    def reconstruct_face(self,
                         image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        image = PIL.Image.fromarray(image)
        input_data = self.transform(image).unsqueeze(0).to(self.device)
        img_rec, instyle = self.encoder(input_data,
                                        randomize_noise=False,
                                        return_latents=True,
                                        z_plus_latent=True,
                                        return_z_plus_latent=True,
                                        resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)
        img_rec = self.postprocess(img_rec[0])
        return img_rec, instyle

    @torch.inference_mode()
    def generate(self, style_type: str, style_id: int, structure_weight: float,
                 color_weight: float, structure_only: bool,
                 instyle: torch.Tensor) -> np.ndarray:
        generator = self.generator_dict[style_type]
        exstyles = self.exstyle_dict[style_type]

        style_id = int(style_id)
        stylename = list(exstyles.keys())[style_id]

        latent = torch.tensor(exstyles[stylename]).to(self.device)
        if structure_only:
            latent[0, 7:18] = instyle[0, 7:18]
        exstyle = generator.generator.style(
            latent.reshape(latent.shape[0] * latent.shape[1],
                           latent.shape[2])).reshape(latent.shape)

        img_gen, _ = generator([instyle],
                               exstyle,
                               z_plus_latent=True,
                               truncation=0.7,
                               truncation_latent=0,
                               use_res=True,
                               interp_weights=[structure_weight] * 7 +
                               [color_weight] * 11)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        img_gen = self.postprocess(img_gen[0])
        return img_gen
