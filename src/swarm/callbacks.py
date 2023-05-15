from typing import Sequence, Union

import numpy as np
import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy

from swarm.utils import (linear_evaluation_binary_class,
                         linear_evaluation_multiclass)


class LinearOnlineEvaluationCallback(pl.Callback):
    def __init__(
            self,
            encoder_output_dim: int,
            num_classes: int,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            augmentations: T.nn.Module,  # Pre-normalize from aug_pipeline
    ):
        super().__init__()
        self.optimizer: T.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.augmentations = augmentations

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.val_dataloader_iter = iter(self.val_dataloader)

        self.linear_classifier: T.nn.Linear

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.linear_classifier = T.nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = T.optim.Adam(self.linear_classifier.parameters(), lr=1e-4)

    def extract_batch(self, batch: Sequence, device: Union[str, T.device]):
        x, y = batch
        x = self.augmentations(x)
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int
    ):

        try:
            batch = next(self.train_dataloader_iter)
        except StopIteration:
            self.train_dataloader_iter = iter(self.train_dataloader)
            batch = next(self.train_dataloader_iter)

        x, y = self.extract_batch(batch, pl_module.device)

        with T.no_grad():
            features = pl_module.forward(x)

        features = features.detach()
        preds = self.linear_classifier(features)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred_labels = T.argmax(preds, dim=1)
        acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
        pl_module.log("gtzan_train_acc", acc, on_step=False, on_epoch=True)
        pl_module.log("gtzan_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int
    ):

        try:
            batch = next(self.val_dataloader_iter)
        except StopIteration:
            self.val_dataloader_iter = iter(self.val_dataloader)
            batch = next(self.val_dataloader_iter)

        x, y = self.extract_batch(batch, pl_module.device)

        pl_module.eval()

        with T.no_grad():
            features = pl_module.forward(x)

            features = features.detach()
            preds = self.linear_classifier(features)
            loss = F.cross_entropy(preds, y)

        pl_module.train()

        pred_labels = T.argmax(preds, dim=1)
        acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
        pl_module.log("gtzan_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("gtzan_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)


class EarlyStoppingFromSlopeCallback(pl.Callback):
    DIRECTION_MAXIMIZE = "maximize"
    DIRECTION_MINIMIZE = "minimize"

    def __init__(
        self,
        metric_name: str,
        patience: int = 5,
        direction: str = "minimize",
        stop_slope_magnitude: float = 0.0,
    ):
        super().__init__()
        self.patience = patience
        self.metric_history = []
        self.metric_name = metric_name

        if not stop_slope_magnitude >= 0:
            raise ValueError("stop_slope_magnitude should be >= 0")
        self.stop_slope_magnitude = stop_slope_magnitude

        if direction not in (self.DIRECTION_MAXIMIZE, self.DIRECTION_MINIMIZE):
            raise ValueError("direction must be either 'minimize' or 'maximize'")
        self.direction = direction

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get the metric value
        if self.metric_name in trainer.callback_metrics:
            m = trainer.callback_metrics[self.metric_name]
            if m is None or T.isnan(m).any():
                trainer.should_stop = True
            else:
                self.metric_history.append(trainer.callback_metrics[self.metric_name].item())
                if len(self.metric_history) >= self.patience:
                    lr = LinearRegression()
                    selection = self.metric_history[-self.patience:]
                    x = np.arange(len(selection)).reshape(-1, 1)
                    y = np.array(selection).reshape(-1, 1)
                    lr.fit(x, y)
                    slope = lr.coef_[0][0]
                    if self.direction == self.DIRECTION_MAXIMIZE:
                        slope *= -1

                    # If the slope is positive, stop training
                    if slope > self.stop_slope_magnitude:
                        trainer.should_stop = True


class LinearBinaryEvaluationCallback(pl.Callback):
    def __init__(
        self,
        dataset_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        augmentations: T.nn.Module,
        encoder_dims: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.augmentations = augmentations
        self.encoder_dims = encoder_dims

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        res = linear_evaluation_binary_class(
            encoder=T.nn.Sequential(self.augmentations, pl_module),
            encoder_dims=self.encoder_dims,
            device=pl_module.device,
            train_dataset=self.train_dataset,
            test_dataset=self.val_dataset,
        )
        pl_module.log(f"{self.dataset_name}_val_acc", res.acc, on_step=False, on_epoch=True)
        pl_module.log(f"{self.dataset_name}_val_f1", res.f1, on_step=False, on_epoch=True)
        pl_module.log(f"{self.dataset_name}_val_loss", res.loss, on_step=False, on_epoch=True)


class LinearMulticlassEvaluationCallback(pl.Callback):
    def __init__(
        self,
        dataset_name: str,
        num_classes: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        augmentations: T.nn.Module,
        encoder_dims: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.augmentations = augmentations
        self.encoder_dims = encoder_dims

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        res = linear_evaluation_multiclass(
            encoder=T.nn.Sequential(self.augmentations, pl_module),
            num_classes=self.num_classes,
            encoder_dims=self.encoder_dims,
            device=pl_module.device,
            train_dataset=self.train_dataset,
            test_dataset=self.val_dataset,
        )
        pl_module.log(f"{self.dataset_name}_val_acc", res.acc, on_step=False, on_epoch=True)
        pl_module.log(f"{self.dataset_name}_val_f1", res.f1, on_step=False, on_epoch=True)
        pl_module.log(f"{self.dataset_name}_val_loss", res.loss, on_step=False, on_epoch=True)
