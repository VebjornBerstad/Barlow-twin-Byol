import os

import torch
from torch.utils.data import Dataset


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
        x = torch.load(self.file_list[idx])
        if x.ndim == 2:
            x = x.unsqueeze(0)

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
        x = torch.load(self.file_list[idx])
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, self.label
