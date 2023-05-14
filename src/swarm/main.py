import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from swarm.augmentations import RandomCropWidth
from swarm.dataset import AudioDataset, AudiosetDataset
from swarm.models import ConvNet, LinearOnlineEvaluationCallback, BarlowTwins

from swarm.config import parse_dvc_training_config, parse_dvc_augmentation_config
from dataclasses import dataclass
from swarm.config import parse_dvc_gtzan_config
from swarm.augmentations import aug_pipeline

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
    gtzan_config = parse_dvc_gtzan_config()

    transform = transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    audioset_dataset = AudiosetDataset(config.audio_dir, transform=transform)
    gtzan_train_dataset = AudioDataset(config.train_dir, transform=transform)
    gtzan_val_dataset = AudioDataset(config.val_dir, transform=transform)

    audioset_dataset_len = len(audioset_dataset)

    # Split
    valid_size = int(training_config.val_split * audioset_dataset_len)
    train_size = audioset_dataset_len - valid_size
    audioset_train_dataset, audioset_val_dataset = random_split(audioset_dataset, [train_size, valid_size])

    batch_size = training_config.batch_size
    audioset_train_dataloader = DataLoader(audioset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    audioset_val_dataloader = DataLoader(audioset_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    gtzan_train_dataloader = DataLoader(gtzan_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gtzan_val_dataloader = DataLoader(gtzan_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    X_train_example, _ = next(iter(audioset_train_dataloader))
    X_train_example = X_train_example[:1]

    augmentations = aug_pipeline(
        mixup_ratio=augmentation_config.mixup_ratio,
        linear_fader_gain=augmentation_config.linear_fader_gain,
        rrc_crop_scale_min=augmentation_config.rrc_crop_scale_min,
        rrc_crop_scale_max=augmentation_config.rrc_crop_scale_max,
        rrc_freq_scale_min=augmentation_config.rrc_freq_scale_min,
        rrc_freq_scale_max=augmentation_config.rrc_freq_scale_max,
        rrc_time_scale_min=augmentation_config.rrc_time_scale_min,
        rrc_time_scale_max=augmentation_config.rrc_time_scale_max,
    )

    encoder_online = ConvNet(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')
    encoder_target = ConvNet(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')
    barlow_byol = BarlowTwins(
        encoder_online=encoder_online,
        encoder_target=encoder_target,
        encoder_out_dim=training_config.emb_dim_size,
        learning_rate=training_config.lr,
        xcorr_lambda=training_config.xcorr_lambda,
        augmentations=augmentations
    )

    linear_evaluation = LinearOnlineEvaluationCallback(
        encoder_output_dim=training_config.emb_dim_size,
        num_classes=gtzan_config.num_classes,
        train_dataloader=gtzan_train_dataloader,
        val_dataloader=gtzan_val_dataloader
    )
    checkpoint_callback = ModelCheckpoint(every_n_epochs=10, save_top_k=-1, save_last=True)
    logger = TensorBoardLogger("logs", name="BarlowTwins")

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
