import argparse
import json
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
from swarm.configs.augmentations import parse_dvc_augmentation_config
from swarm.configs.dataset_gtzan import parse_dvc_gtzan_config
from swarm.configs.training import parse_dvc_training_config
from swarm.datasets import AudiosetDataset, GtzanDataset
from swarm.models import BarlowTwins, Encoder


@dataclass
class Config:
    train_dir: Path
    audio_dir: Path
    audioset_train_csv_path: Path
    audioset_class_labels_indices_csv_path: Path
    model_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=Path, help="The input directory containing the GTZAN dataset WAV files.", required=True)
    parser.add_argument('--audio_dir', type=Path, help='The output directory to save the validation dataset.', required=True)
    parser.add_argument('--audioset_train_csv_path', type=Path, help='The output directory to save the validation dataset.', required=True)
    parser.add_argument('--audioset_class_labels_indices_csv_path', type=Path, help='The output directory to save the validation dataset.', required=True)
    parser.add_argument('--model_path', type=Path, help='The output path for the encoder model.', required=True)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    config = parse_args()
    training_config = parse_dvc_training_config()
    augmentation_config = parse_dvc_augmentation_config()
    gtzan_config = parse_dvc_gtzan_config()

    # Set up model folder.
    if not config.model_path.exists():
        config.model_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up default dataset transform, which cuts a set number of frames
    # from the mel spectrograms.
    transform = transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),
    ])

    batch_size = training_config.batch_size

    # Set up GTZAN.
    gtzan_train_dataset = GtzanDataset(config.train_dir, transforms=transform)
    gtzan_train_size = int(gtzan_config.train_val_split * len(gtzan_train_dataset))
    gtzan_val_size = len(gtzan_train_dataset) - gtzan_train_size
    gtzan_train_dataset, gtzan_val_dataset = random_split(gtzan_train_dataset, [gtzan_train_size, gtzan_val_size])
    gtzan_train_dataloader = DataLoader(gtzan_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    gtzan_val_dataloader = DataLoader(gtzan_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Set up Audioset.
    audioset_dataset = AudiosetDataset(
        audio_path=config.audio_dir,
        labels_desc_csv=config.audioset_class_labels_indices_csv_path,
        labels_csv=config.audioset_train_csv_path,
        transform=transform,
    )
    audioset_dataset_len = len(audioset_dataset)
    valid_size = int(training_config.val_split * audioset_dataset_len)
    train_size = audioset_dataset_len - valid_size
    audioset_train_dataset, audioset_val_dataset = random_split(audioset_dataset, [train_size, valid_size])
    audioset_train_dataloader = DataLoader(audioset_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    audioset_val_dataloader = DataLoader(audioset_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # Set up training augmentations.
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

    # Extract pre-normalize augmentations, used to normalize input data.
    pre_aug_normalize = augmentations[0]

    # Define encoders for the Barlow Twins mode.
    X_train_example, _ = next(iter(audioset_train_dataloader))
    X_train_example = X_train_example[:1]
    encoder_online = Encoder(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')
    encoder_target = Encoder(in_channels=1, emb_dim_size=training_config.emb_dim_size, X_train_example=X_train_example, device='cuda')

    # Set up the Barlow Twins model.
    barlow_byol = BarlowTwins(
        encoder_online=encoder_online,
        encoder_target=encoder_target,
        encoder_out_dim=training_config.emb_dim_size,
        learning_rate=training_config.lr,
        xcorr_lambda=training_config.xcorr_lambda,
        augmentations=augmentations
    )

    # Define Pytorch Lightning callbacks and loggers.
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

    # Define the Pytorch lightning trainer.
    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=training_config.max_epochs,
        callbacks=[
            linear_evaluation,
            early_stopping,
        ],
        logger=logger,
    )

    # Fit the model to the data.
    trainer.fit(barlow_byol, train_dataloaders=audioset_train_dataloader, val_dataloaders=audioset_val_dataloader)

    # Extract and save the trained model.
    best_model: BarlowTwins = early_stopping.best_module  # type: ignore
    best_encoder = best_model.target[0]
    model = T.nn.Sequential(pre_aug_normalize, best_encoder).eval()
    sd = model.cpu().state_dict()
    T.save(sd, config.model_path)

    with open(config.model_path.with_suffix('.json'), 'w') as f:
        json.dump(
            obj={
                "X_train_example_shape": list(X_train_example.shape),
                "emb_dim_size": training_config.emb_dim_size,
                "in_channels": 1,
            },
            fp=f,
            indent=4,
        )


if __name__ == '__main__':
    main()
