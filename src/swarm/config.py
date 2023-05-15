"""This module contains various configuration classes for the project, and acts as a way
to acces the DVC params.yaml file.

When adding new sections to the DVC params file, make sure to include them here. Use the
existing classes/functions as a guide.
"""


from dataclasses import dataclass

import dvc.api as dvc_api


@dataclass
class SpectrogramConfig:
    """Contains information about the DVC spectrogram params."""
    target_sample_rate: int
    n_mels: int
    f_min: int
    f_max: int


def parse_dvc_spectrogram_config() -> SpectrogramConfig:
    """
    Returns:
        A SpectrogramConfig object containing the spectrogram params.
    """
    params = dvc_api.params_show()
    spectrogram_params = params['spectrogram']

    return SpectrogramConfig(
        target_sample_rate=spectrogram_params['target_sample_rate'],
        n_mels=spectrogram_params['n_mels'],
        f_min=spectrogram_params['f_min'],
        f_max=spectrogram_params['f_max'],
    )


@dataclass
class TrainingConfig:
    """Contains information about the DVC training params."""
    batch_size: int
    val_split: float
    lr: float
    xcorr_lambda: float
    emb_dim_size: int
    early_stopping_patience: int


def parse_dvc_training_config() -> TrainingConfig:
    """
    Returns:
        A TrainingConfig object containing the training params.
    """
    params = dvc_api.params_show()
    training_params = params['training']

    return TrainingConfig(
        batch_size=training_params['batch_size'],
        val_split=training_params['val_split'],
        lr=training_params['lr'],
        xcorr_lambda=training_params['xcorr_lambda'],
        emb_dim_size=training_params['emb_dim_size'],
        early_stopping_patience=training_params['early_stopping_patience'],
    )


@dataclass
class AugmentationConfig:
    """Contains information about the DVC augmentation params."""
    rcw_target_frames: int
    mixup_ratio: float
    mixup_memory_size: int
    linear_fader_gain: float
    rrc_crop_scale_min: float
    rrc_crop_scale_max: float
    rrc_freq_scale_min: float
    rrc_freq_scale_max: float
    rrc_time_scale_min: float
    rrc_time_scale_max: float


def parse_dvc_augmentation_config() -> AugmentationConfig:
    """
    Returns:
        An AugmentationConfig object containing the augmentation params.
    """
    params = dvc_api.params_show()
    augmentation_params = params['augmentations']

    rcw = augmentation_params['random_crop_width']
    mixup = augmentation_params['mixup']
    linear_fader = augmentation_params['linear_fader']
    rrc = augmentation_params['random_resize_crop']

    return AugmentationConfig(
        rcw_target_frames=rcw['target_frames'],
        mixup_ratio=mixup['ratio'],
        mixup_memory_size=mixup['memory_size'],
        linear_fader_gain=linear_fader['gain'],
        rrc_crop_scale_min=rrc['crop_scale']["min"],
        rrc_crop_scale_max=rrc['crop_scale']["max"],
        rrc_freq_scale_min=rrc['freq_scale']["min"],
        rrc_freq_scale_max=rrc['freq_scale']["max"],
        rrc_time_scale_min=rrc['time_scale']["min"],
        rrc_time_scale_max=rrc['time_scale']["max"],
    )


@dataclass
class GtzanConfig:
    """Contains information about the DVC GTZAN dataset params."""
    train_val_split: float
    segment_length_sec: float
    hop_length_sec: float
    num_classes: int


def parse_dvc_gtzan_config() -> GtzanConfig:
    """
    Returns:
        A GtzanConfig object containing the GTZAN dataset params.
    """
    params = dvc_api.params_show()
    gtzan_params = params['datasets']['gtzan']

    return GtzanConfig(
        train_val_split=gtzan_params['train_val_split'],
        segment_length_sec=gtzan_params['segment_length_sec'],
        hop_length_sec=gtzan_params['hop_length_sec'],
        num_classes=gtzan_params['num_classes'],
    )
