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
