import os

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from torchvision.transforms import Compose
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.class_names = os.listdir(main_dir)
        self.file_list = []
        self.label_list = []

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
    def __init__(
        self,
        audio_path: Path,
        labels_desc_csv: Path,
        labels_csv: Path,
        transform: torch.nn.Module | Compose | None = None,
    ):
        self.data_root = audio_path
        self.transform = transform
        self.class_labels = pd.read_csv(labels_desc_csv)

        df_labels = pd.read_csv(labels_csv)
        self.df_labels = df_labels.join(df_labels["positive_labels"].str.get_dummies(sep=","))

        self.ytid_to_label_idx = {ytid: idx for idx, ytid in enumerate(self.df_labels["YTID"].tolist())}
        self.df_labels_only = self.df_labels.iloc[:, 4:]
        self.labels = torch.tensor(self.df_labels_only.to_numpy(dtype=np.int32))

        ytids = set(self.df_labels['YTID'].tolist())
        self.file_list = [x for x in audio_path.glob("*.pt") if x.stem in ytids]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_list[idx]
        file_name = file_path.stem
        y = self.labels[self.ytid_to_label_idx[file_name]]
        x = torch.load(file_path)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, y

    def label_to_index(self, label: str) -> int:
        return self.df_labels_only.columns.get_loc(label)

    def index_to_label(self, index: int) -> str:
        return str(self.df_labels_only.columns[index])
