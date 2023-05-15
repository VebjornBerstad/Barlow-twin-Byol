import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class GtzanDataset(Dataset):
    """A dataset class for working with the GTZAN dataset.

    Dataset source:
    https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

    See the DVC pipeline (dvc.yaml) for how to preprocess the dataset.
    """

    def __init__(
        self,
        main_dir: Path,
        transforms: T.nn.Module | Compose | None = None
    ):
        """Data should be stored in a directory with the following structure:
            ./<dataset dir>/<class name>/<file>.pt
        E.g:
            ./datasets/gtzan_train_mel_split/blues/blues.00000_0.pt

        Args:
            main_dir: A string path to the directory containing the GTZAN dataset preprocessed
                from wavs into mel spectrogram tensors.
            transforms: A PyTorch transformation stack to apply to the data.
        """
        self.main_dir = main_dir
        self.transforms = transforms
        self.class_names = os.listdir(main_dir)
        self.file_list = []
        self.label_list = []

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(main_dir, class_name)
            file_names = os.listdir(class_dir)
            self.file_list.extend([os.path.join(class_dir, f) for f in file_names])
            self.label_list.extend([idx] * len(file_names))

    def __len__(self):
        """
        Returns:
            The length of the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: The index of the item to retrieve. Currently only supports ints.
        """
        x = T.load(self.file_list[idx])
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.transforms:
            x = self.transforms(x)
        label = self.label_list[idx]

        return x, label


class AudiosetDataset(Dataset):
    """A dataset class for working with the Audioset dataset.


    Dataset sources:
    - Train: https://www.kaggle.com/datasets/zfturbo/audioset
    - Validation: https://www.kaggle.com/datasets/zfturbo/audioset-valid
    - Test: https://www.kaggle.com/c/audioset-leaderboard/data (Unlabeled test set)
    """

    def __init__(
        self,
        audio_path: Path,
        labels_desc_csv: Path,
        labels_csv: Path,
        transform: T.nn.Module | Compose | None = None,
    ):
        """Data should be split into train/val directories with the following structure:
            ./<project dir>/<dataset dir>/<file>.pt
        E.g:
            ./datasets/audioset_train_mel_split/__0OQemumqg.pt

        Additionally, there are accompanying metadata files for the train/val datasets.
        These are used to map the YTID to the labels.

        Args:
            audio_path: A Path to the directory containing the preprocessed mel spectrogram audio tensors.
            labels_desc_csv: A Path to the CSV containing the label descriptions (class_labels_indices.csv).
            labels_csv: A Path to the CSV containing the YTID to label mappings (e.g. train.csv).
            transform: A PyTorch transformation stack to apply to the data.
        """
        self.data_root = audio_path
        self.transform = transform
        self.class_labels = pd.read_csv(labels_desc_csv)

        df_labels = pd.read_csv(labels_csv)

        # Convert labels to one-hot encoding
        self.df_labels = df_labels.join(df_labels["positive_labels"].str.get_dummies(sep=","))

        # Files are named as <YTID>.pt, so we want an efficient mapping between them and the labels.
        self.ytid_to_label_idx = {ytid: idx for idx, ytid in enumerate(self.df_labels["YTID"].tolist())}
        self.df_labels_only = self.df_labels.iloc[:, 4:]
        self.labels = T.tensor(self.df_labels_only.to_numpy(dtype=np.int32))

        # Only load files that are in the labels CSV.
        ytids = set(self.df_labels['YTID'].tolist())
        self.file_list = [x for x in audio_path.glob("*.pt") if x.stem in ytids]

    def __len__(self):
        """
        Returns:
            The length of the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[T.Tensor, T.Tensor]:
        """
        Args:
            idx: The index of the item to retrieve. Currently only supports ints.
        """
        file_path = self.file_list[idx]
        file_name = file_path.stem
        y = self.labels[self.ytid_to_label_idx[file_name]]
        x = T.load(file_path)

        # Add channel dimension if it's missing, i.e. if we forgot to unsqueeze during preprocess.
        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, y

    def label_to_index(self, label: str) -> int:
        return self.df_labels_only.columns.get_loc(label)

    def index_to_label(self, index: int) -> str:
        return str(self.df_labels_only.columns[index])
