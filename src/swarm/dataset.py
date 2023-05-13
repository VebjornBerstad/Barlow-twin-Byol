import os

import torch
from torch.utils.data import Dataset

# import random
# import torch.nn.functional as F
# import torchaudio as TA
# from .conv_wav_mel import preprocess_audio
# from torchaudio.transforms import Resample
# from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, main_dir, target_sample_rate, unit_sec, transform=None):
        self.main_dir = main_dir
        self.target_sample_rate = target_sample_rate
        self.transform = transform
        self.class_names = os.listdir(main_dir)
        self.file_list = []
        self.label_list = []
        self.unit_length = int(unit_sec*target_sample_rate)

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(main_dir, class_name)
            file_names = os.listdir(class_dir)
            self.file_list.extend([os.path.join(class_dir, f) for f in file_names])
            self.label_list.extend([idx] * len(file_names))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # waveform, sample_rate = TA.load(self.file_list[idx])
        x = torch.load(self.file_list[idx])

        # if waveform.shape[0] > 1:
        #     waveform = torch.mean(waveform, dim=0, keepdim=True)

        # # Resample to target sample rate
        # if sample_rate != self.target_sample_rate:
        #     resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        #     waveform = resampler(waveform)

        # length_adj = self.unit_length - len(waveform)
        # if length_adj > 0:
        #     half_adj = int(length_adj // 2)
        #     wav = F.pad(waveform, (half_adj, int(length_adj - half_adj)))

        # length_adj = len(waveform) - self.unit_length
        # start = random.randint(0, length_adj) if length_adj > 0 else 0
        # waveform = waveform[start:start + self.unit_length]

        # x = (self.to_mel_spec(wav) + torch.finfo().eps).log()

        # log_mel_spectrogram = preprocess_audio(waveform=waveform, sample_rate=sample_rate)

        if self.transform:
            x = self.transform(x)
        label = self.label_list[idx]

        return x, label


class AudiosetDataset(Dataset):
    def __init__(self, data_root, target_sample_rate, unit_sec, transform=None):
        self.data_root = data_root
        self.target_sample_rate = target_sample_rate
        self.transform = transform
        self.unit_length = int(unit_sec*target_sample_rate)
        self.file_list = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith('.pt')]
        self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load audio file
        # waveform, sample_rate = TA.load(self.file_list[idx])
        x = torch.load(self.file_list[idx])

        # if waveform.shape[0] > 1:
        #     waveform = torch.mean(waveform, dim=0, keepdim=True)

        # # Resample to target sample rate
        # if sample_rate != self.target_sample_rate:
        #     resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
        #     waveform = resampler(waveform)

        # length_adj = self.unit_length - len(waveform)
        # if length_adj > 0:
        #     half_adj = int(length_adj // 2)
        #     wav = F.pad(waveform, (half_adj, int(length_adj - half_adj)))

        # length_adj = len(waveform) - self.unit_length
        # start = random.randint(0, length_adj) if length_adj > 0 else 0
        # waveform = waveform[start:start + self.unit_length]

        # x = (self.to_mel_spec(wav) + torch.finfo().eps).log()

        # log_mel_spectrogram = preprocess_audio(waveform=waveform, sample_rate=sample_rate)

        if self.transform:
            x = self.transform(x)

        return x, self.label
