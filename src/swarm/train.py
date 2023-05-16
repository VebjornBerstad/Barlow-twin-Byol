import argparse
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from swarm.augmentations import RandomCropWidth, aug_pipeline
from swarm.callbacks import (EarlyStoppingFromSlopeCallback,
                             LinearOnlineEvaluationCallback)
from swarm.config import (parse_dvc_augmentation_config,
                          parse_dvc_gtzan_config, parse_dvc_training_config)
from swarm.dataset import AudiosetDataset, GtzanDataset
from swarm.models import BarlowTwins, Encoder


@dataclass
class Config:
    train_dir: Path
    val_dir: Path
    audio_dir: Path
    audioset_train_csv_path: Path
    audioset_class_labels_indices_csv_path: Path
    model_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=Path, help="The input directory containing the GTZAN dataset WAV files.")
    parser.add_argument('--val_dir', type=Path, help='The output directory to save the training dataset.')
    parser.add_argument('--audio_dir', type=Path, help='The output directory to save the validation dataset.')
    parser.add_argument('--audioset_train_csv_path', type=Path, help='The output directory to save the validation dataset.')
    parser.add_argument('--audioset_class_labels_indices_csv_path', type=Path, help='The output directory to save the validation dataset.')
    parser.add_argument('--model_path', type=Path, help='The output directory to save the validation dataset.')
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    config = parse_args()
    training_config = parse_dvc_training_config()
    augmentation_config = parse_dvc_augmentation_config()
    gtzan_config = parse_dvc_gtzan_config()

    # Set up model folder.
    if not config.model_path.exists():
        config.model_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    audioset_dataset = AudiosetDataset(
        audio_path=config.audio_dir,
        labels_desc_csv=config.audioset_class_labels_indices_csv_path,
        labels_csv=config.audioset_train_csv_path,
        transform=transform,
    )
    gtzan_train_dataset = GtzanDataset(config.train_dir, transforms=transform)
    gtzan_val_dataset = GtzanDataset(config.val_dir, transforms=transform)

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
        mixup_memory_size=augmentation_config.mixup_memory_size,
        linear_fader_gain=augmentation_config.linear_fader_gain,
        rrc_crop_scale_min=augmentation_config.rrc_crop_scale_min,
        rrc_crop_scale_max=augmentation_config.rrc_crop_scale_max,
        rrc_freq_scale_min=augmentation_config.rrc_freq_scale_min,
        rrc_freq_scale_max=augmentation_config.rrc_freq_scale_max,
        rrc_time_scale_min=augmentation_config.rrc_time_scale_min,
        rrc_time_scale_max=augmentation_config.rrc_time_scale_max,
    )

    pre_aug_normalize = augmentations[0]

    encoder_online = Encoder(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')
    encoder_target = Encoder(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')
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
        val_dataloader=gtzan_val_dataloader,
        augmentations=pre_aug_normalize,
    )
    early_stopping = EarlyStoppingFromSlopeCallback(
        metric_name='gtzan_val_loss',
        direction=EarlyStoppingFromSlopeCallback.DIRECTION_MINIMIZE,
        patience=training_config.early_stopping_patience,
    )
    logger = TensorBoardLogger("logs", name="BarlowTwins")

    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=50,
        callbacks=[
            linear_evaluation,
            early_stopping,
        ],
        logger=logger,
    )
    trainer.fit(barlow_byol, train_dataloaders=audioset_train_dataloader, val_dataloaders=audioset_val_dataloader)

    best_model: BarlowTwins = early_stopping.best_module  # type: ignore
    best_encoder = best_model.target[0]

    # Save the encoder
    encoder_path = Path('models/encoder.pth')
    T.save(best_encoder.state_dict(), encoder_path)

    # Save the pre-augmentation normalization
    pre_aug_normalize_path = Path('models/pre_aug_normalize.pth')
    T.save(pre_aug_normalize.state_dict(), pre_aug_normalize_path)


if __name__ == '__main__':
    main()
