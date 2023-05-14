import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from swarm.augmentations import RandomCropWidth
from swarm.dataset import AudioDataset, AudiosetDataset
from swarm.models import ConvNet, LinearOnlineEvaluationCallback, barlowBYOL

from swarm.config import parse_dvc_training_config, parse_dvc_augmentation_config
from dataclasses import dataclass

from pathlib import Path
import argparse


@dataclass
class Config:
    train_dir: Path
    val_dir: Path
    audio_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=Path, help="The input directory containing the GTZAN dataset WAV files.")
    parser.add_argument('--val_dir', type=Path, help='The output directory to save the training dataset.')
    parser.add_argument('--audio_dir', type=Path, help='The output directory to save the validation dataset.')
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    config = parse_args()
    training_config = parse_dvc_training_config()
    augmentation_config = parse_dvc_augmentation_config()

    transform = transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    batch_size = training_config.batch_size

    audioset_dataset = AudiosetDataset(config.audio_dir, transform=transform)

    gtzan_train_dataset = AudioDataset(config.train_dir, transform=transform)
    gtzan_val_dataset = AudioDataset(config.val_dir, transform=transform)

    # Split
    valid_size = int(training_config.val_split * len(audioset_dataset))
    train_size = len(audioset_dataset) - valid_size
    audioset_train_dataset, audioset_val_dataset = random_split(audioset_dataset, [train_size, valid_size])

    audioset_train_dataloader = DataLoader(audioset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    audioset_val_dataloader = DataLoader(audioset_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    gtzan_train_dataloader = DataLoader(gtzan_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gtzan_val_dataloader = DataLoader(gtzan_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    X_train_example, _ = next(iter(audioset_train_dataloader))
    X_train_example = X_train_example[:1]

    # encoder = resnet18()
    # encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # encoder.fc = nn.Identity()

    emb_dim_size = 4096

    encoder = ConvNet(in_channels=1, emb_dim_size=emb_dim_size, X_train_example=X_train_example, device='cuda')
    # encoder = Autoencoder(emb_dim_size=emb_dim_size, X_train_example=X_train_example, device='cuda')

    logger = TensorBoardLogger("logs", name="Barlow_BYOL")

    barlow_byol = barlowBYOL(encoder=encoder, tau=0.99, encoder_out_dim=emb_dim_size, num_training_samples=len(audioset_dataset), batch_size=batch_size)

    linear_evaluation = LinearOnlineEvaluationCallback(
        encoder_output_dim=emb_dim_size,
        num_classes=10,
        train_dataloader=gtzan_train_dataloader,
        val_dataloader=gtzan_val_dataloader
    )
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
