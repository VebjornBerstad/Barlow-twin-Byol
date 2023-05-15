import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch as T
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

import optuna
from swarm.augmentations import RandomCropWidth, aug_pipeline
from swarm.config import (AugmentationConfig, GtzanConfig, TrainingConfig,
                          parse_dvc_augmentation_config,
                          parse_dvc_gtzan_config, parse_dvc_training_config)
from swarm.dataset import AudioDataset, AudiosetDataset
from swarm.models import BarlowTwins, ConvNet, LinearOnlineEvaluationCallback
from swarm.utils import linear_evaluation_multiclass


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


def train_barlow_twins(
    trial: optuna.Trial,
    config: Config,
    gtzan_config: GtzanConfig,
    _augmentation_config: AugmentationConfig,
    _training_config: TrainingConfig,
) -> float:
    # augmentation_config: AugmentationConfig = AugmentationConfig(
    #     rcw_target_frames=trial.suggest_int('rcw_target_frames', 32, 192),
    #     mixup_ratio=trial.suggest_float('mixup_ratio', 0.1, 0.9),
    #     mixup_memory_size=_augmentation_config.mixup_memory_size,
    #     linear_fader_gain=trial.suggest_float('linear_fader_gain', 0.1, 0.9),
    #     rrc_crop_scale_min=_augmentation_config.rrc_crop_scale_min,
    #     rrc_crop_scale_max=trial.suggest_float('rrc_crop_scale_max', 1.0, 2.0),
    #     rrc_freq_scale_min=trial.suggest_float('rrc_freq_scale_min', 0.2, 1.0),
    #     rrc_freq_scale_max=trial.suggest_float('rrc_freq_scale_max', 1.0, 2.0),
    #     rrc_time_scale_min=trial.suggest_float('rrc_time_scale_min', 0.2, 1.0),
    #     rrc_time_scale_max=trial.suggest_float('rrc_time_scale_max', 1.0, 2.0),
    # )
    # training_config = TrainingConfig(
    #     batch_size=_training_config.batch_size,
    #     val_split=_training_config.val_split,
    #     lr=trial.suggest_float('lr', 1e-6, 1e-1),
    #     xcorr_lambda=trial.suggest_float('xcorr_lambda', 0.1, 0.9),
    #     emb_dim_size=trial.suggest_categorical('emb_dim_size', [128, 256, 512, 1024, 22048, 4096]),
    # )
    augmentation_config = _augmentation_config
    training_config = _training_config

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
        val_dataloader=gtzan_val_dataloader,
        augmentations=pre_aug_normalize,
    )
    logger = TensorBoardLogger("logs", name="BarlowTwins")

    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=5,
        callbacks=[linear_evaluation],
        logger=logger,
    )
    trainer.fit(barlow_byol, train_dataloaders=audioset_train_dataloader, val_dataloaders=audioset_val_dataloader)

    encoder = T.nn.Sequential(
        pre_aug_normalize,
        encoder_online,
    )
    loss_result = linear_evaluation_multiclass(
        encoder=encoder,
        encoder_dims=training_config.emb_dim_size,
        device='cuda',
        num_classes=gtzan_config.num_classes,
        train_dataset=gtzan_train_dataset,
        test_dataset=gtzan_val_dataset,
    )

    trial.set_user_attr('gtzan_val_acc', loss_result.acc)
    trial.set_user_attr('gtzan_val_f1', loss_result.f1)
    trial.set_user_attr('gtzan_val_loss', loss_result.loss)

    return loss_result.loss


def main():
    config = parse_args()
    training_config = parse_dvc_training_config()
    augmentation_config = parse_dvc_augmentation_config()
    gtzan_config = parse_dvc_gtzan_config()

    objective = partial(
        train_barlow_twins,
        config=config,
        gtzan_config=gtzan_config,
        _augmentation_config=augmentation_config,
        _training_config=training_config,
    )

    optuna_logs_dir = Path('optuna')
    if not optuna_logs_dir.exists():
        optuna_logs_dir.mkdir(parents=True)

    study = optuna.create_study(
        direction=optuna.study.StudyDirection.MINIMIZE,
        storage=f'sqlite:///{optuna_logs_dir}/barlowtwins.db',
    )
    study.optimize(objective, n_trials=100)


if __name__ == '__main__':
    main()
