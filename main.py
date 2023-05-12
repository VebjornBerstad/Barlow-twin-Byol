from models import barlowBYOL, LinearEvaluationCallback, ConvNet
from dataset import AudioDataset, AudiosetDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn

from torchvision.models import resnet18

from augmentations import RandomCropWidth

from pytorch_lightning.loggers import TensorBoardLogger


def main():

    sample_rate = 16000
    mean = 0.0008
    std = 0.0060
    transform = transforms.Compose([
        RandomCropWidth(target_frames=96), #96
        ])
    
    train_dir = '/home/vebjo/version_1/datasets/gtzan_train_mel_split'
    val_dir = '/home/vebjo/version_1/datasets/gtzan_val_mel_split'
    audio_dir = '/home/vebjo/version_1/datasets/audioset_train_mel_split'
    batch_size = 512

    audioset_dataset = AudiosetDataset(audio_dir, target_sample_rate=sample_rate, unit_sec=1, transform=transform)

    gtzan_train_dataset = AudioDataset(train_dir, target_sample_rate=sample_rate, unit_sec=1, transform=transform)
    gtzan_val_dataset = AudioDataset(val_dir, target_sample_rate=sample_rate, unit_sec=1, transform=transform)

    # Split
    train_size = int(0.9 * len(audioset_dataset))    
    valid_size = len(audioset_dataset) - train_size
    audioset_train_dataset, audioset_val_dataset = random_split(audioset_dataset, [train_size, valid_size])

    audioset_train_dataloader = DataLoader(audioset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    audioset_val_dataloader = DataLoader(audioset_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    gtzan_train_dataloader = DataLoader(gtzan_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gtzan_val_dataloader = DataLoader(gtzan_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # encoder = resnet18()
    # encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # encoder.fc = nn.Identity()

    encoder = ConvNet(in_channels=1, out_features=1024)

    logger = TensorBoardLogger("logs", name="Barlow_BYOL")
    
    barlow_byol = barlowBYOL(encoder=encoder, tau=0.99, encoder_out_dim=1024, num_training_samples=len(audioset_dataset), batch_size=batch_size)

    linear_evaluation = LinearEvaluationCallback(encoder_output_dim=1024, num_classes=10, train_dataloader=gtzan_train_dataloader, val_dataloader=gtzan_val_dataloader)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=50, save_top_k=-1, save_last=True)

    barlow_byol_trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=500,
        callbacks=[linear_evaluation, checkpoint_callback],
        logger=logger,
        )
    barlow_byol_trainer.fit(barlow_byol, train_dataloaders=audioset_train_dataloader, val_dataloaders=audioset_val_dataloader)


if __name__ == '__main__':
    main()