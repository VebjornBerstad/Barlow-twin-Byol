import argparse
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchaudio as TA
from tqdm import tqdm

from swarm.config import parse_dvc_spectrogram_config
from swarm.preprocess.tools import convert_waveform_to_lms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    input_zip: Path
    output_dir_lms: Path
    output_dir_metadata: Path
    temp_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_zip', type=Path, help="The input zip file containing the AudioSet dataset WAV files.")
    parser.add_argument('--output_dir_lms', type=Path, help="The output directory where the preprocessed dataset will be saved.")
    parser.add_argument('--output_dir_metadata', type=Path, help="The output directory where the metadata will be saved.")
    parser.add_argument('--temp_dir', type=Path, help="The temporary directory to unzip the input zip file.")
    args = parser.parse_args()
    return Config(**vars(args))


def preprocess_file(
    input_path: Path,
    dest_folder: Path,
    target_sample_rate: int,
    n_mels: int,
    f_min: int,
    f_max: int,
    device: T.device | str
) -> None:
    if not dest_folder.exists():
        dest_folder.mkdir(parents=True, exist_ok=True)

    try:
        waveform, sample_rate = TA.load(input_path)  # type: ignore
        waveform = waveform.to(device)
        lms = convert_waveform_to_lms(
            waveform=waveform,
            waveform_sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            device=device,
        )
        save_path = (dest_folder / input_path.stem).with_suffix('.pt')
        T.save(lms.cpu(), save_path)
    except Exception as e:
        logger.error(f"Error converting file {input_path} to LMS: {e}")


def main() -> None:
    config = parse_args()
    dvc_spectrogram_config = parse_dvc_spectrogram_config()

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    with zipfile.ZipFile(config.input_zip, 'r') as zip_ref:
        if not config.output_dir_metadata.exists():
            config.output_dir_metadata.mkdir(parents=True, exist_ok=True)

        # Extract all csv files.
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        zip_ref.extractall(config.output_dir_metadata, csv_files)

        # Extract and preprocess all wav files.

        if not config.temp_dir.exists():
            config.temp_dir.mkdir(parents=True, exist_ok=True)
        if not config.output_dir_lms.exists():
            config.output_dir_lms.mkdir(parents=True, exist_ok=True)

        wav_files = [f for f in zip_ref.namelist() if f.endswith('.wav')]
        for file in tqdm(wav_files, desc="Preprocessing AudioSet files"):
            zip_ref.extract(file, config.temp_dir)
            path = config.temp_dir / file

            preprocess_file(
                input_path=path,
                dest_folder=config.output_dir_lms,
                target_sample_rate=dvc_spectrogram_config.target_sample_rate,
                n_mels=dvc_spectrogram_config.n_mels,
                f_min=dvc_spectrogram_config.f_min,
                f_max=dvc_spectrogram_config.f_max,
                device=device,
            )


if __name__ == '__main__':
    main()
