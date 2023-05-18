import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchvision.transforms as TV_transforms
from torch.utils.data import random_split

from swarm.augmentations import Normalize, RandomCropWidth
from swarm.configs.augmentations import parse_dvc_augmentation_config
from swarm.configs.dataset_gtzan import parse_dvc_gtzan_config
from swarm.configs.training import parse_dvc_training_config
from swarm.datasets import GtzanDataset
from swarm.models import Encoder
from swarm.utils import linear_evaluation_multiclass


@dataclass
class Config:
    model_path: Path
    gtzan_path_train: Path
    linear_eval_model_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, help='The path to the encoder model.', required=True)
    parser.add_argument('--gtzan_path_train', type=Path, help='The path to the data to train on.', required=True)
    parser.add_argument('--linear_eval_model_path', type=Path, help='The output path for the linear evaluation model.', required=True)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    config = parse_args()
    augmentation_config = parse_dvc_augmentation_config()
    training_config = parse_dvc_training_config()
    gtzan_config = parse_dvc_gtzan_config()

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    model_weights = T.load(config.model_path)
    with open(config.model_path.with_suffix('.json'), 'r') as fp:
        model_config = json.load(fp)
    X_train_example = T.randn(model_config['X_train_example_shape'])
    model = T.nn.Sequential(Normalize(), Encoder(
        X_train_example=X_train_example,
        device=device,
        emb_dim_size=model_config['emb_dim_size'],
        in_channels=model_config['in_channels'],
    )).to(device)
    model.load_state_dict(model_weights)

    transforms = TV_transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    gtzan_train_dataset = GtzanDataset(config.gtzan_path_train, transforms=transforms)
    gtzan_train_size = int(gtzan_config.train_val_split * len(gtzan_train_dataset))
    gtzan_val_size = len(gtzan_train_dataset) - gtzan_train_size
    gtzan_train_dataset, gtzan_val_dataset = random_split(gtzan_train_dataset, [gtzan_train_size, gtzan_val_size])

    eval_res = linear_evaluation_multiclass(
        encoder=model,
        encoder_dims=training_config.emb_dim_size,
        device=device,
        num_classes=gtzan_config.num_classes,
        train_dataset=gtzan_train_dataset,
        test_dataset=gtzan_val_dataset,
    )

    print(f"Linear evaluation loss: {eval_res.loss}")
    print(f"Linear evaluation accuracy: {eval_res.acc}")
    print(f"Linear evaluation f1: {eval_res.f1}")

    # Save the model.
    T.save(eval_res.model.cpu().state_dict(), config.linear_eval_model_path)


if __name__ == '__main__':
    main()
