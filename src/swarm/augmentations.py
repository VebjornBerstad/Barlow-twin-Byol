import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomCropWidth(nn.Module):
    def __init__(self, target_frames):
        super().__init__()
        self.target_frames = target_frames

    def forward(self, log_mel_spectrogram):
        num_frames = log_mel_spectrogram.shape[-1]
        if num_frames > self.target_frames:
            start = torch.randint(num_frames - self.target_frames + 1, (1,)).item()
            log_mel_spectrogram = log_mel_spectrogram[:, :, start:start + self.target_frames]
        elif num_frames < self.target_frames:
            padding = torch.zeros(1, 64, self.target_frames - num_frames)
            log_mel_spectrogram = torch.cat((log_mel_spectrogram, padding), dim=-1)
        return log_mel_spectrogram


class Normalize(nn.Module):
    def __init__(self, eps=1e-8):
        super(Normalize, self).__init__()
        self.eps = eps

        self.mean: torch.Tensor = torch.tensor([0.0], requires_grad=False)
        self.std: torch.Tensor = torch.tensor([1.0], requires_grad=False)
        self.examples_seen: int = 0

    def forward(self, x):
        batch_size = x.shape[0]

        batch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        self.mean = self.mean.to(x.device)
        self.mean = (self.mean * self.examples_seen + batch_mean * batch_size) / (self.examples_seen + batch_size)

        batch_std = torch.std(x, dim=(0, 2, 3), unbiased=False, keepdim=True).to(x.device) + self.eps
        self.std = self.std.to(x.device)
        self.std = (self.std * self.examples_seen + batch_std * batch_size) / (self.examples_seen + batch_size)

        self.examples_seen += batch_size

        return (x - self.mean) / self.std


class RandomLinearFader(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

    def forward(self, lms):
        head, tail = self.gain * ((2.0 * np.random.rand(2)) - 1.0)
        T = lms.shape[1]
        slope = torch.linspace(head, tail, T, dtype=lms.dtype).reshape(1, T).to(lms.device)
        y = lms + slope
        return y


class RandomResizeCrop(nn.Module):
    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        assert time_scale[1] >= 1.0
        assert freq_scale[1] >= 1.0
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], lms.shape[1], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        lh, lw = virtual_crop_area.shape[-2:]
        h, w = lms.shape[-2:]
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, :, y:y+h, x:x+w] = lms
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, :, i:i+h, j:j+w]
        lms = F.interpolate(crop, size=lms.shape[-2:],
                            mode=self.interpolation, align_corners=True)
        return lms.to(torch.float)


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    """Mixup for BYOL-A.
    Args:
        ratio: Alpha in the paper.
        n_memory: Size of memory bank FIFO.
        log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
    """

    def __init__(self, ratio: float, n_memory: int, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))].to(x.device)
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                else alpha * z + (1. - alpha) * x
        else:
            mixed = x
        # update memory bank
        self.memory_bank = (self.memory_bank + [x.cpu()])[-self.n:]

        return mixed.to(torch.float)


def aug_pipeline(
    mixup_ratio: float = 0.4,
    mixup_memory_size: int = 2048,
    linear_fader_gain: float = 1.0,
    rrc_crop_scale_min: float = 1.0,
    rrc_crop_scale_max: float = 1.5,
    rrc_freq_scale_min: float = 0.6,
    rrc_freq_scale_max: float = 1.5,
    rrc_time_scale_min: float = 0.6,
    rrc_time_scale_max: float = 1.5,
):
    return nn.Sequential(
        Normalize(),
        MixupBYOLA(ratio=mixup_ratio, n_memory=mixup_memory_size),
        RandomResizeCrop(
            virtual_crop_scale=(rrc_crop_scale_min, rrc_crop_scale_max),
            freq_scale=(rrc_freq_scale_min, rrc_freq_scale_max),
            time_scale=(rrc_time_scale_min, rrc_time_scale_max),
        ),
        RandomLinearFader(
            gain=linear_fader_gain,
        ),
        Normalize(),
    )
