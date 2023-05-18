import argparse
from dataclasses import dataclass
from pathlib import Path

import torch as T
import torchvision.transforms as TV_transforms
from torch.utils.data import DataLoader

from swarm.augmentations import RandomCropWidth
from swarm.configs.augmentations import parse_dvc_augmentation_config
from swarm.configs.dataset_gtzan import parse_dvc_gtzan_config
from swarm.datasets import GtzanDataset
from torchmetrics.functional import accuracy, f1_score, auroc


@dataclass
class Config:
    model_path: Path
    gtzan_path_test: Path
    linear_eval_model_path: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, help='The path to the encoder model.', required=True)
    parser.add_argument('--gtzan_path_test', type=Path, help='The path to the data to test on.', required=True)
    parser.add_argument('--linear_eval_model_path', type=Path, help='The output path for the linear evaluation model.', required=True)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    # Set torch seed.
    T.manual_seed(42)

    config = parse_args()
    augmentation_config = parse_dvc_augmentation_config()
    gtzan_config = parse_dvc_gtzan_config()

    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    transforms = TV_transforms.Compose([
        RandomCropWidth(target_frames=augmentation_config.rcw_target_frames),  # 96
    ])

    gtzan_test_dataset = GtzanDataset(config.gtzan_path_test, transforms=transforms)
    dataloader = DataLoader(gtzan_test_dataset, batch_size=512, shuffle=False)

    encoder = T.load(config.model_path).to(device)
    linear_eval_model = T.load(config.linear_eval_model_path).to(device)

    with T.no_grad():
        loss_fn = T.nn.functional.cross_entropy

        losses = []
        y_hats = []
        y_preds = []
        y_actual = []
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_actual.extend(y.tolist())

            z = encoder(X)
            y_hat = linear_eval_model(z)
            losses.extend(loss_fn(y_hat, y, reduction='none').tolist())
            y_hats.extend(y_hat.tolist())
            y_preds.extend(y_hat.argmax(dim=1).tolist())

        loss = T.tensor(losses).mean().item()

        y_hats = T.tensor(y_hats)
        y_preds = T.tensor(y_preds)
        y_actual = T.tensor(y_actual)

        # Calculate multiclass acc.
        acc = accuracy(y_preds, y_actual, task='multiclass', num_classes=gtzan_config.num_classes, average='macro').item()
        f1 = f1_score(y_preds, y_actual, task='multiclass', num_classes=gtzan_config.num_classes, average='macro').item()
        auc = auroc(y_hats, y_actual, task='multiclass', num_classes=gtzan_config.num_classes, average='macro')

        print(f"Linear evaluation loss: {loss}")
        print(f"Linear evaluation accuracy: {acc}")
        print(f"Linear evaluation f1: {f1}")
        print(f"Linear evaluation auc: {auc}")


if __name__ == '__main__':
    main()
