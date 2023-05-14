import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchaudio as TA
from tqdm import tqdm

from swarm.config import parse_dvc_spectrogram_config
from swarm.preprocess.tools import convert_waveform_to_lms, create_audio_segments
from swarm.config import parse_dvc_gtzan_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    input_dir: Path
    output_dir_train: Path
    output_dir_val: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, help="The input directory containing the GTZAN dataset WAV files.")
    parser.add_argument('--output_dir_train', type=Path, help='The output directory to save the training dataset.')
    parser.add_argument('--output_dir_val', type=Path, help='The output directory to save the validation dataset.')
    args = parser.parse_args()
    return Config(**vars(args))


def preprocess_files(
    input_paths: list[Path],
    dest_folder: Path,
    segment_length_sec: float,
    hop_length_sec: float,
    target_sample_rate: int,
    n_mels: int,
    f_min: int,
    f_max: int,
    device: T.device | str
) -> None:
    if not dest_folder.exists():
        dest_folder.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        try:
            waveform, sample_rate = TA.load(input_path)  # type: ignore
            waveform = waveform.to(device)
            audio_segments = create_audio_segments(waveform, sample_rate, segment_length_sec=segment_length_sec, hop_length_sec=hop_length_sec)
            for i, audio_segment in enumerate(audio_segments):
                lms = convert_waveform_to_lms(
                    waveform=audio_segment,
                    waveform_sample_rate=sample_rate,
                    target_sample_rate=target_sample_rate,
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max,
                    device=device,
                )
                save_path = dest_folder / f"{input_path.stem}_{i}.pt"
                T.save(lms.cpu(), save_path)
        except Exception as e:
            logger.error(f"Error converting file {input_path} to LMS: {e}")


def main() -> None:
    config = parse_args()
    dvc_gtzan_config = parse_dvc_gtzan_config()
    dvc_spectrogram_config = parse_dvc_spectrogram_config()

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    root = config.input_dir / "Data" / "genres_original"
    genre_folders = [genre for genre in root.iterdir() if genre.is_dir()]

    for genre_folder in tqdm(genre_folders):
        files = list(genre_folder.glob('*.wav'))
        random.shuffle(files)

        split_idx = int(len(files) * dvc_gtzan_config.train_val_split)

        train_files = files[:split_idx]
        dest_train_folder = config.output_dir_train / genre_folder.name
        preprocess_files(
            input_paths=train_files,
            dest_folder=dest_train_folder,
            segment_length_sec=dvc_gtzan_config.segment_length_sec,
            hop_length_sec=dvc_gtzan_config.hop_length_sec,
            target_sample_rate=dvc_spectrogram_config.target_sample_rate,
            n_mels=dvc_spectrogram_config.n_mels,
            f_min=dvc_spectrogram_config.f_min,
            f_max=dvc_spectrogram_config.f_max,
            device=device
        )

        val_files = files[split_idx:]
        dest_val_folder = config.output_dir_val / genre_folder.name
        preprocess_files(
            input_paths=val_files,
            dest_folder=dest_val_folder,
            segment_length_sec=dvc_gtzan_config.segment_length_sec,
            hop_length_sec=dvc_gtzan_config.hop_length_sec,
            target_sample_rate=dvc_spectrogram_config.target_sample_rate,
            n_mels=dvc_spectrogram_config.n_mels,
            f_min=dvc_spectrogram_config.f_min,
            f_max=dvc_spectrogram_config.f_max,
            device=device
        )


if __name__ == '__main__':
    main()
