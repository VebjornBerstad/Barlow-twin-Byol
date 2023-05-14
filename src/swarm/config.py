from dataclasses import dataclass

import dvc.api as dvc_api


@dataclass
class SpectrogramConfig:
    target_sample_rate: int
    n_mels: int
    f_min: int
    f_max: int


def parse_dvc_spectrogram_config() -> SpectrogramConfig:
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
    batch_size: int
    val_split: float
    lr: float
    xcorr_lambda: float
    emb_dim_size: int


def parse_dvc_training_config() -> TrainingConfig:
    params = dvc_api.params_show()
    training_params = params['training']

    return TrainingConfig(
        batch_size=training_params['batch_size'],
        val_split=training_params['val_split'],
        lr=training_params['lr'],
        xcorr_lambda=training_params['xcorr_lambda'],
        emb_dim_size=training_params['emb_dim_size'],
    )


@dataclass
class AugmentationConfig:
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
    train_val_split: float
    segment_length_sec: float
    hop_length_sec: float
    num_classes: int


def parse_dvc_gtzan_config() -> GtzanConfig:
    params = dvc_api.params_show()
    gtzan_params = params['datasets']['gtzan']

    return GtzanConfig(
        train_val_split=gtzan_params['train_val_split'],
        segment_length_sec=gtzan_params['segment_length_sec'],
        hop_length_sec=gtzan_params['hop_length_sec'],
        num_classes=gtzan_params['num_classes'],
    )
