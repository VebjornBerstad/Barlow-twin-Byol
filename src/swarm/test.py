import argparse
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchvision.transforms as TV_transforms

from swarm.augmentations import RandomCropWidth
from swarm.configs.augmentations import parse_dvc_augmentation_config
from swarm.configs.training import parse_dvc_training_config
from swarm.configs.dataset_gtzan import parse_dvc_gtzan_config
from swarm.datasets import GtzanDataset
from swarm.utils import linear_evaluation_multiclass


@dataclass
class Config:
    model_path: Path
    gtzan_path_train: Path
    gtzan_path_test: Path
    linear_eval_model_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, help='The path to the encoder model.', required=True)
    parser.add_argument('--gtzan_path_train', type=Path, help='The path to the data to train on.', required=True)
    parser.add_argument('--gtzan_path_test', type=Path, help='The path to the data to test on.', required=True)
    parser.add_argument('--linear_eval_model_path', type=Path, help='The output path for the linear evaluation model.', required=True)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    config = parse_args()
    augmentation_config = parse_dvc_augmentation_config()
    training_config = parse_dvc_training_config()
    gtzan_config = parse_dvc_gtzan_config()

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    model = T.load(config.model_path).to(device)
    transforms = TV_transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    gtzan_train_dataset = GtzanDataset(config.gtzan_path_train, transforms=transforms)
    gtzan_test_dataset = GtzanDataset(config.gtzan_path_test, transforms=transforms)

    eval_res = linear_evaluation_multiclass(
        encoder=model,
        encoder_dims=training_config.emb_dim_size,
        device=device,
        num_classes=gtzan_config.num_classes,
        train_dataset=gtzan_train_dataset,
        test_dataset=gtzan_test_dataset,
    )

    print(f"Linear evaluation loss: {eval_res.loss}")
    print(f"Linear evaluation accuracy: {eval_res.acc}")
    print(f"Linear evaluation f1: {eval_res.f1}")

    # Save the model.
    T.save(eval_res.model, config.linear_eval_model_path)


if __name__ == '__main__':
    main()
