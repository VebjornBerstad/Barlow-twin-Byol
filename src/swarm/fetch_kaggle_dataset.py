import kaggle
from pathlib import Path
import argparse
from dataclasses import dataclass
import zipfile
from tqdm import tqdm


@dataclass
class Config:
    kaggle_dataset: str
    data_dir: Path
    temp_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle_dataset', type=str, help='Kaggle dataset name')
    parser.add_argument('--data_dir', type=Path, help='Directory to save the dataset')
    parser.add_argument('--temp_dir', type=Path, help='Temporary directory to download the dataset')
    args = parser.parse_args()
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

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        for file in tqdm(file_list, desc="Unzipping dataset files"):
            zip_ref.extract(file, config.data_dir)


if __name__ == '__main__':
    main()
