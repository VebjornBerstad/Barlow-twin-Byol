from dataclasses import dataclass

import dvc.api as dvc_api


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
