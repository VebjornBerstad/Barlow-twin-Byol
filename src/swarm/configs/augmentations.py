from dataclasses import dataclass

import dvc.api as dvc_api


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
