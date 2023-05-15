"""Various utilities for the swarm package, including linear evaluation models
and training functions.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch as T
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, f1_score
from tqdm import trange


@dataclass
class EvaluationResult:
    """A set of standard evaluation matrics."""
    acc: float
    f1: float
    loss: float


class LinearModelMulticlass(T.nn.Module):
    """A linear model for multiclass classification."""

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: The number of input features.
            out_features: The number of output features.
        """
        super().__init__()
        self.linear = T.nn.Linear(in_features, out_features)
        self.softmax = T.nn.Softmax(dim=1)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Args:
            x: The input tensor.
        Returns:
            The output tensor.
        """
        return self.softmax(self.linear(x))


def linear_evaluation_multiclass(
    encoder: T.nn.Module,
    encoder_dims: int,
    device: T.device | str,
    num_classes: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_epochs: int = 100,
    slope_window: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> EvaluationResult:
    """Trains a linear evaluation model on top of the encoder.

    Model selection is performed by looking at the loss of the testing set and selcting the
    model with the lowest loss. Additionally, this is done in tandem with a slope-based method
    for estimating when the model has started to overfit on the training data. The test dataset
    is used for model selection and final evaluation.

    Args:
        encoder: The encoder model. It should encode the data from the training sets into
            a representation vector of size `encoder_dims`.
        encoder_dims: The size of the representation vector.
        device: The device used during training.
        num_classes: The number of classes in the dataset.
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        max_epochs: The maximum number of epochs to run for.
        slope_window: The number of epochs/loss values to use for slope calculation.
        batch_size: The batch size to use during training.
        lr: The learning rate to use for the optimizer during training.
    Returns:
        The evaluation result.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    linear_model = LinearModelMulticlass(encoder_dims, num_classes)
    linear_model.to(device)

    optimizer = T.optim.Adam(linear_model.parameters(), lr=lr)
    loss_fn = T.nn.functional.cross_entropy

    best_model = deepcopy(linear_model)
    best_val_loss = float('inf')
    val_loss_history = []
    for _ in trange(max_epochs, desc="Linear evaluation"):
        # Model training
        linear_model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with T.no_grad():
                X = encoder(X)

            y_hat = linear_model(X)
            val_loss = loss_fn(y_hat, y, reduction='mean')
            val_loss.backward()
            optimizer.step()

        # Select the best model and to stop training early.
        with T.no_grad():
            linear_model.eval()
            val_losses = []
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                X = encoder(X)
                y_hat = linear_model(X)
                val_losses += loss_fn(y_hat, y, reduction='none').tolist()
            val_loss = sum(val_losses) / len(val_losses)

            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(linear_model)

            if len(val_loss_history) > slope_window:
                lingreg = LinearRegression()
                loss_last = val_loss_history[-slope_window:]
                x = np.arange(len(loss_last)).reshape(-1, 1)
                y = np.array(loss_last).reshape(-1, 1)
                lingreg.fit(x, y)

                if lingreg.coef_[0][0] > 0:
                    break

    # Evaluation metrics are calculated.
    with T.no_grad():
        best_model.eval()

        y_hat = []
        y = []
        val_losses = []
        for X, y_ in val_loader:
            X = X.to(device)
            y_ = y_.to(device)

            X = encoder(X)
            X_lin = best_model(X)
            y_hat.extend(X_lin.argmax(dim=1).tolist())
            y.extend(y_.tolist())

            val_losses += loss_fn(X_lin, y_, reduction='none').tolist()

        val_loss = sum(val_losses) / len(val_losses)
        y_hat = T.tensor(y_hat)
        y = T.tensor(y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=num_classes)
        f1 = f1_score(y_hat, y, task="multiclass", num_classes=num_classes)

        return EvaluationResult(acc=acc.item(), f1=f1.item(), loss=val_loss)


class LinearModelBinaryClass(T.nn.Module):
    """A linear model for binary classification."""

    def __init__(self, in_features: int):
        """
        Args:
            in_features: The number of input features.
        """
        super().__init__()
        self.linear = T.nn.Linear(in_features, 1)
        self.sigmoid = T.nn.Sigmoid()

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Args:
            x: The input tensor.
        Returns:
            The output tensor.
        """
        return self.sigmoid(self.linear(x))


def linear_evaluation_binary_class(
    encoder: T.nn.Module,
    encoder_dims: int,
    device: T.device | str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_iter: int = 100,
    slope_window: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> EvaluationResult:
    """Trains a linear evaluation model on top of the encoder.

    Model selection is performed by looking at the loss of the testing set and selcting the
    model with the lowest loss. Additionally, this is done in tandem with a slope-based method
    for estimating when the model has started to overfit on the training data. The test dataset
    is used for model selection and final evaluation.

    Args:
        encoder: The encoder model. It should encode the data from the training sets into
            a representation vector of size `encoder_dims`.
        encoder_dims: The size of the representation vector.
        device: The device used during training.
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        max_iter: The maximum number of epochs to run for.
        slope_window: The number of epochs/loss values to use for slope calculation.
        batch_size: The batch size to use during training.
        lr: The learning rate to use for the optimizer during training.
    Returns:
        The evaluation result.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    linear_model = LinearModelBinaryClass(encoder_dims)
    linear_model.to(device)

    optimizer = T.optim.Adam(linear_model.parameters(), lr=lr)
    loss_fn = T.nn.functional.binary_cross_entropy

    best_model = deepcopy(linear_model)
    best_val_loss = float('inf')
    val_loss_history = []
    for _ in trange(max_iter, desc="Linear evaluation"):
        # Model training
        linear_model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            with T.no_grad():
                X = encoder(X)

            y_hat = linear_model(X)
            val_loss = loss_fn(y_hat, y, reduction='mean')
            val_loss.backward()
            optimizer.step()

        # Select the best model and to stop training early.
        with T.no_grad():
            linear_model.eval()
            val_losses = []
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                X = encoder(X)
                y_hat = linear_model(X)
                val_losses += loss_fn(y_hat, y, reduction='none').tolist()
            val_loss = sum(val_losses) / len(val_losses)

            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(linear_model)

            if len(val_loss_history) > slope_window:
                linreg = LinearRegression()
                loss_last = val_loss_history[-slope_window:]
                x = np.arange(len(loss_last)).reshape(-1, 1)
                y = np.array(loss_last).reshape(-1, 1)
                linreg.fit(x, y)

                if linreg.coef_[0][0] > 0:
                    break

    # Evaluation metrics are calculated.
    with T.no_grad():
        best_model.eval()

        y_hat = []
        y = []
        val_losses = []
        for X, y_ in val_loader:
            X = X.to(device)
            y_ = y_.to(device)

            X = encoder(X)
            X_lin = best_model(X)
            y_hat.extend(X_lin.argmax(dim=1).tolist())
            y.extend(y_.tolist())

            val_losses += loss_fn(X_lin, y_, reduction='none').tolist()

        val_loss = sum(val_losses) / len(val_losses)
        y_hat = T.tensor(y_hat)
        y = T.tensor(y)
        acc = accuracy(y_hat, y, task="binary")
        f1 = f1_score(y_hat, y, task="binary")

        return EvaluationResult(acc=acc.item(), f1=f1.item(), loss=val_loss)
