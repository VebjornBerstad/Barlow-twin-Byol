import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import kaggle
from tqdm import tqdm


@dataclass
class Config:
    kaggle_dataset: str
    temp_dir: Path
    unzip: bool = True
    output_dir: Path | None = None


def parse_bool(s: str) -> bool:
    if s.lower() in ['true', 't']:
        return True
    elif s.lower() in ['false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {s}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle_dataset', type=str, help='Kaggle dataset name')
    parser.add_argument('--temp_dir', type=Path, help='Temporary directory to download the dataset')
    parser.add_argument('--output_dir', type=Path, help='Directory to save the dataset', required=False)
    parser.add_argument('--unzip', type=parse_bool, help='Unzip the dataset', required=False, default=True)
    args = parser.parse_args()

    if args.unzip and args.output_dir is None:
        raise argparse.ArgumentTypeError("Argument --output_dir is required if --unzip is set to True")

    return Config(**vars(args))


def _download_file(temp_dir: Path, kaggle_dataset: str) -> None:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        kaggle_dataset,
        path=temp_dir,
        unzip=False,
        quiet=False,
    )


def main():
    config = parse_args()
    zip_name = config.kaggle_dataset.split('/')[1] + '.zip'
    zip_path = config.temp_dir / zip_name

    if not zip_path.exists():
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        _download_file(temp_dir=config.temp_dir, kaggle_dataset=config.kaggle_dataset)

    if config.unzip:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()

            for file in tqdm(file_list, desc="Unzipping dataset files"):
                zip_ref.extract(file, config.output_dir)


if __name__ == '__main__':
    main()
