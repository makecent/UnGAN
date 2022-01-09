import random

import torch
from mmgen.core.runners.fp16_utils import auto_fp16
from mmgen.models.architectures.common import get_module_device
from mmgen.models.architectures.stylegan import StyleGANv2Generator, StyleGAN2Discriminator
from mmgen.models.builder import MODULES
from torch.nn import functional as F
import random

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from mmgen.core.runners.fp16_utils import auto_fp16
from mmgen.models.architectures import PixelNorm
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES
from mmgen.models.architectures.stylegan.modules.styleganv2_modules import (ConstantInput, ConvDownLayer,
                                         EqualLinearActModule,
                                         ModMBStddevLayer, ModulatedStyleConv,
                                         ModulatedToRGB, ResBlock)
from mmgen.models.architectures.stylegan.utils import get_mean_latent, style_mixing
from mmgen.utils import get_root_logger


# @MODULES.register_module()
# class ConditionalStyleGANv2Generator(StyleGANv2Generator):
#
#     def __init__(self, style_channels=8, noise_channels=120, *args, **kwargs):
#         super().__init__(style_channels=style_channels*4+noise_channels*4, *args, **kwargs)
#         self.noise_channels = noise_channels
#         self.embedding_layer = torch.nn.Embedding(10, style_channels)
#
#     def embed(self, styles):
#         embedded_styles = []
#         for style in styles:
#             embeddings = []
#             for digits in style:
#                 noise = torch.randn(self.noise_channels).cuda()
#                 embedding = torch.cat([torch.cat([self.embedding_layer(i), noise]) for i in digits])  # (num_digit, (style_chl + noise_chl)) = (4, 10+224)
#                 embeddings.append(embedding)
#             embedded_styles.append(torch.stack(embeddings))  # (batch_size * num_digit, style_chl + noise_chl) = (bs * 4, 234)
#         return embedded_styles
#
#     @auto_fp16()
#     def forward(self,
#                 styles,
#                 num_batches=-1,
#                 return_noise=False,
#                 return_latents=False,
#                 inject_index=None,
#                 truncation=1,
#                 truncation_latent=None,
#                 input_is_latent=False,
#                 injected_noise=None,
#                 randomize_noise=True):
#
#         # receive noise and conduct sanity check.
#         device = get_module_device(self)
#         if styles is None:
#             assert num_batches > 0 and not input_is_latent
#             styles = [torch.randint(10, [num_batches, 4]).to(device)]
#         if not input_is_latent:
#             noise_batch = styles
#         else:
#             noise_batch = None
#
#         styles = self.embed(styles)
#         styles = [self.style_mapping(s) for s in styles]
#
#         if injected_noise is None:
#             if randomize_noise:
#                 injected_noise = [None] * self.num_injected_noises
#             else:
#                 injected_noise = [
#                     getattr(self, f'injected_noise_{i}')
#                     for i in range(self.num_injected_noises)
#                 ]
#         # use truncation trick
#         if truncation < 1:
#             style_t = []
#             # calculate truncation latent on the fly
#             if truncation_latent is None and not hasattr(
#                     self, 'truncation_latent'):
#                 self.truncation_latent = self.get_mean_latent()
#                 truncation_latent = self.truncation_latent
#             elif truncation_latent is None and hasattr(self,
#                                                        'truncation_latent'):
#                 truncation_latent = self.truncation_latent
#
#             for style in styles:
#                 style_t.append(truncation_latent + truncation *
#                                (style - truncation_latent))
#
#             styles = style_t
#         # no style mixing
#         if len(styles) < 2:
#             inject_index = self.num_latents
#
#             if styles[0].ndim < 3:
#                 latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
#
#             else:
#                 latent = styles[0]
#         # style mixing
#         else:
#             if inject_index is None:
#                 inject_index = random.randint(1, self.num_latents - 1)
#
#             latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
#             latent2 = styles[1].unsqueeze(1).repeat(
#                 1, self.num_latents - inject_index, 1)
#
#             latent = torch.cat([latent, latent2], 1)
#
#         # 4x4 stage
#         out = self.constant_input(latent)
#         out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
#         skip = self.to_rgb1(out, latent[:, 1])
#
#         _index = 1
#
#         # 8x8 ---> higher resolutions
#         for up_conv, conv, noise1, noise2, to_rgb in zip(
#                 self.convs[::2], self.convs[1::2], injected_noise[1::2],
#                 injected_noise[2::2], self.to_rgbs):
#             out = up_conv(out, latent[:, _index], noise=noise1)
#             out = conv(out, latent[:, _index + 1], noise=noise2)
#             skip = to_rgb(out, latent[:, _index + 2], skip)
#             _index += 2
#
#         img = skip.to(torch.float32)
#         # img = img.unflatten(dim=0,
#         #                     sizes=(num_batches, -1))  # (bs * num_digits, 3, H, W) --> # (bs, num_digits, 3, H, W)
#         # # num_digits = img.shape[1]
#         # # img = img[:, :, :, 1:-1, 1:-1]
#         # img = img.permute(0, 2, 3, 1, 4)  # (bs, num_digits, 3, H, W) --> # (bs, 3, H, num_digits, W)
#         #
#         # img = img.flatten(start_dim=3)  # (bs, 3, H, num_digits, W) --> # (bs, 3, H, num_digits * W)
#
#         # img = F.interpolate(img, scale_factor=(1, 1 / num_digits),
#         #                     mode='nearest')  # (bs, 3, H, num_digits * W) --> # (bs, 3, H, W) H=W
#
#         if return_latents or return_noise:
#             output_dict = dict(
#                 fake_img=img,
#                 latent=latent,
#                 inject_index=inject_index,
#                 noise_batch=noise_batch)
#             return output_dict
#
#         return img

class CustomInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.channel = channel
        self.embedding_layer = torch.nn.Embedding(10, channel * int(np.prod(size)))

    def forward(self, digits):
        with torch.no_grad():
            embeddings = self.embedding_layer(digits)
        embeddings = embeddings.view(-1, self.channel, *self.size)
        return embeddings


@MODULES.register_module()
class ConditionalStyleGANv2Generator(StyleGANv2Generator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_input = CustomInput(512, size=4)

    @auto_fp16()
    def forward(self,
                styles,
                digits=None,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True):

        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        if digits is None:
            digits = torch.randint(10, [num_batches])
        out = self.constant_input(digits.cuda())
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        img = skip.to(torch.float32)

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return img


@MODULES.register_module()
class ConditionalStyleGAN2Discriminator(nn.Module):

    def __init__(self,
                 in_channels,
                 num_conv=4,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 out_fp32=True,
                 convert_input_fp32=True,
                 pretrained=None):
        super().__init__()
        self.num_conv = num_conv
        self.num_fp16_scale = num_fp16_scales
        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32
        self.out_fp32 = out_fp32

        _use_fp16 = num_fp16_scales > 0
        convs = [ConvDownLayer(3, 512, 1)]

        for i in range(num_conv):
            out_channel = 512
            convs.append(
                ResBlock(
                    in_channels,
                    out_channel,
                    blur_kernel,
                    fp16_enabled=_use_fp16,
                    convert_input_fp32=convert_input_fp32))

        self.convs = nn.Sequential(*convs)

        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, 512, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                512,
                512,
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(512, 1),
        )
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def forward(self, x):
        x = self.convs(x)

        x = self.mbstd_layer(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')