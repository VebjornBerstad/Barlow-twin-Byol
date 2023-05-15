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
    acc: float
    f1: float
    loss: float


class LinearModelMulticlass(T.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = T.nn.Linear(in_features, out_features)
        self.softmax = T.nn.Softmax(dim=1)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.linear(x)


def linear_evaluation_multiclass(
    encoder: T.nn.Module,
    encoder_dims: int,
    device: T.device | str,
    num_classes: int,
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_iter: int = 100,
    slope_window: int = 5,
    batch_size: int = 512,
) -> EvaluationResult:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    linear_model = LinearModelMulticlass(encoder_dims, num_classes)
    linear_model.to(device)

    optimizer = T.optim.Adam(linear_model.parameters(), lr=1e-3)
    loss_fn = T.nn.functional.cross_entropy

    best_model = deepcopy(linear_model)
    best_val_loss = float('inf')
    encoder = encoder.to(device)
    val_loss_history = []
    for _ in trange(max_iter, desc="Linear evaluation"):
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
                lr = LinearRegression()
                loss_last = val_loss_history[-slope_window:]
                x = np.arange(len(loss_last)).reshape(-1, 1)
                y = np.array(loss_last).reshape(-1, 1)
                lr.fit(x, y)

                if lr.coef_[0][0] > 0:
                    break

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


