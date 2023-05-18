import random

import numpy as np
import torch as T
import torch.nn.functional as F


class RandomCropWidth(T.nn.Module):
    """Randomly crop the input spectrogram to a fixed width."""

    def __init__(self, target_frames: int):
        """
        Args:
            target_frames: The target number of frames.
        """
        super().__init__()
        self.target_frames = target_frames

    def forward(self, spectrogram: T.Tensor) -> T.Tensor:
        """Randomly crop the input spectrogram to a fixed width.

        Args:
            spectrogram: The input spectrogram.

        Returns:
            A cropped spectrogram
        """
        num_frames = spectrogram.shape[-1]
        if num_frames > self.target_frames:
            start = T.randint(num_frames - self.target_frames + 1, (1,)).item()
            spectrogram = spectrogram[:, :, start:start + self.target_frames]
        elif num_frames < self.target_frames:
            padding = T.zeros(1, 64, self.target_frames - num_frames)
            spectrogram = T.cat((spectrogram, padding), dim=-1)
        return spectrogram


class Normalize(T.nn.Module):
    """Normalization of an input tensor along the channel dimension using a weighted mean and average."""

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: A small value to avoid division by zero.
        """
        super(Normalize, self).__init__()
        self.eps = T.nn.Parameter(T.tensor([eps], dtype=T.float32), requires_grad=False)
        self.examples_seen = T.nn.Parameter(T.tensor([0], dtype=T.long), requires_grad=False)

        self.mean: T.Tensor = T.nn.Parameter(T.tensor([0.0]), requires_grad=False)
        self.std: T.Tensor = T.nn.Parameter(T.tensor([1.0]), requires_grad=False)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Normalize the input tensor along the channel dimension.

        Args:
            x: The input tensor.

        Returns:
            The normalized input tensor.
        """

        # TODO: Consider using weights instead of directly multiplying by self.examples_seen and batch_size.

        x = x.to(self.mean.device)
        batch_size = x.shape[0]

        batch_mean = T.mean(x, dim=(0, 2, 3))
        self.mean.data = ((self.mean * self.examples_seen + batch_mean * batch_size) / (self.examples_seen + batch_size)).data

        batch_std = T.std(x, dim=(0, 2, 3), unbiased=False) + self.eps
        self.std.data = ((self.std * self.examples_seen + batch_std * batch_size) / (self.examples_seen + batch_size)).data

        self.examples_seen += batch_size
        return (x - self.mean) / (self.std + self.eps)


class RandomLinearFader(T.nn.Module):
    """Randomly add a linear fade to the input spectrogram.

    Idea and implementation from:
    [1] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, ‘BYOL for Audio: Exploring Pre-Trained General-Purpose Audio Representations’,
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 137–151, 2023, doi: 10.1109/TASLP.2022.3221007.
    """

    def __init__(self, gain: float = 1.0):
        """
        Args:
            gain: The gain of the linear fading.
        """
        super().__init__()
        self.gain = gain

    def forward(self, spectrogram: T.Tensor) -> T.Tensor:
        """Randomly add a linear fade to the input spectrogram.

        Args:
            spectrogram: The input spectrogram.

        Returns:
            The spectrogram with a linear fade applied.
        """
        head, tail = self.gain * ((2.0 * np.random.rand(2)) - 1.0)
        t = spectrogram.shape[1]
        slope = T.linspace(head, tail, t, dtype=spectrogram.dtype).reshape(1, t).to(spectrogram.device)
        y = spectrogram + slope
        return y


class RandomResizeCrop(T.nn.Module):
    """Randomly crop the input spectrogram to a fixed width.

    Idea and implementation from:
    [1] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, ‘BYOL for Audio: Exploring Pre-Trained General-Purpose Audio Representations’,
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 137–151, 2023, doi: 10.1109/TASLP.2022.3221007.
    [2] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, ‘BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation’,
        in 2021 International Joint Conference on Neural Networks (IJCNN), Jul. 2021, pp. 1–8. doi: 10.1109/IJCNN52387.2021.9534474.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        assert time_scale[1] >= 1.0
        assert freq_scale[1] >= 1.0
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'

    @staticmethod
    def get_params(
        virtual_crop_size: tuple[float, float],
        in_size: tuple[int, int],
        time_scale: tuple[float, float],
        freq_scale: tuple[float, float]
    ):
        """Sample parameters to use for a random resize crop operation.

        Args:
            virtual_crop_size: The size of the virtual crop.
            in_size: The size of the input spectrogram.
            time_scale: The range of the time scale.
            freq_scale: The range of the frequency scale.
        Returns:
            The parameters to use for the random resize crop operation.
        """
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms: T.Tensor) -> T.Tensor:
        """Randomly crop the input spectrogram to a fixed width.

        Args:
            lms: An input log-mel spectrogram.

        Returns:
            The cropped spectrogram.
        """
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (T.zeros((lms.shape[0], lms.shape[1], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(T.float).to(lms.device))
        lh, lw = virtual_crop_area.shape[-2:]
        h, w = lms.shape[-2:]
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, :, y:y+h, x:x+w] = lms
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, :, i:i+h, j:j+w]
        lms = F.interpolate(crop, size=lms.shape[-2:],
                            mode=self.interpolation, align_corners=True)
        return lms.to(T.float)


def log_mixup_exp(xa: T.Tensor, xb: T.Tensor, alpha: float):
    """Mix up two tensors in log-scale.

    Args:
        xa: The first tensor.
        xb: The second tensor.
        alpha: The mixing ratio.

    Returns:
        The mixed tensor.
    """
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return T.log(x + T.finfo(x.dtype).eps)


class MixupBYOLA(T.nn.Module):
    """Simulates adding background noise to the input spectrogram, to separate the foreground (what to learn) from the background (augmentation noise).

    Idea and implementation from:
    [1] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, ‘BYOL for Audio: Exploring Pre-Trained General-Purpose Audio Representations’,
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 137–151, 2023, doi: 10.1109/TASLP.2022.3221007.
    [2] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, ‘BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation’,
        in 2021 International Joint Conference on Neural Networks (IJCNN), Jul. 2021, pp. 1–8. doi: 10.1109/IJCNN52387.2021.9534474.
    """

    def __init__(self, ratio: float, n_memory: int, log_mixup_exp: bool = True):
        """
        Args:
            ratio: Alpha in the paper.
            n_memory: Size of memory bank FIFO.
            log_mixup_exp: Use log-mixup-exp to mix if this is True, or mix without notion of log-scale.
        """
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []

    def forward(self, spectrogram: T.Tensor):
        """Apply mixup to the input spectrogram.

        Args:
            spectrogram: The input spectrogram.
        Returns:
            The mixed spectrogram.
        """
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))].to(spectrogram.device)
            # mix them
            mixed = log_mixup_exp(spectrogram, z, 1. - alpha) if self.log_mixup_exp \
                else alpha * z + (1. - alpha) * spectrogram
        else:
            mixed = spectrogram
        # update memory bank
        self.memory_bank = (self.memory_bank + [spectrogram.cpu()])[-self.n:]

        return mixed.to(T.float)


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
    """Create a pipeline of augmentations to use for training using default augmentations.

    Args:
        mixup_ratio: The ratio of mixup to use.
        mixup_memory_size: The size of the memory bank to use for mixup.
        linear_fader_gain: The gain of the linear fader.
        rrc_crop_scale_min: The minimum crop scale to use for random resize crop.
        rrc_crop_scale_max: The maximum crop scale to use for random resize crop.
        rrc_freq_scale_min: The minimum frequency scale to use for random resize crop.
        rrc_freq_scale_max: The maximum frequency scale to use for random resize crop.
        rrc_time_scale_min: The minimum time scale to use for random resize crop.
        rrc_time_scale_max: The maximum time scale to use for random resize crop.

    Returns:
        A pipeline of augmentations to use for training.
    """
    return T.nn.Sequential(
        Normalize(),  # Pre-normalization
        MixupBYOLA(ratio=mixup_ratio, n_memory=mixup_memory_size),
        RandomResizeCrop(
            virtual_crop_scale=(rrc_crop_scale_min, rrc_crop_scale_max),
            freq_scale=(rrc_freq_scale_min, rrc_freq_scale_max),
            time_scale=(rrc_time_scale_min, rrc_time_scale_max),
        ),
        RandomLinearFader(
            gain=linear_fader_gain,
        ),
        Normalize(),  # Post-normalization
    )
