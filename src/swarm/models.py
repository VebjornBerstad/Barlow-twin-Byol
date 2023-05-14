from typing import Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from swarm.augmentations import aug_pipeline, mel_aug


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        assert z1.shape == z2.shape
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        # cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Autoencoder(nn.Module):
    def __init__(self, emb_dim_size: int, X_train_example: torch.Tensor, device: torch.device | str):
        super().__init__()

        self.emb_dim_size = emb_dim_size

        # Define downsample layers for the residual blocks
        downsample1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16)
        )
        downsample2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )
        downsample3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )

        enc_conv = nn.Sequential(
            nn.BatchNorm2d(1),
            ResidualBlock(1, 16, stride=2, downsample=downsample1),
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, stride=2, downsample=downsample2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2, downsample=downsample3),
            ResidualBlock(64, 64),
            nn.Flatten()
        ).to(device)

        enc_conv_res = enc_conv(X_train_example.to(device))

        enc_linear = nn.Sequential(
            nn.Linear(enc_conv_res.size(1), self.emb_dim_size),
            nn.BatchNorm1d(self.emb_dim_size),
            # nn.Tanh(),
        )

        self.encoder = nn.Sequential(enc_conv, enc_linear)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        return z


class ConvNet(nn.Module):
    def __init__(self, X_train_example: torch.Tensor,  device: torch.device | str, emb_dim_size: int, in_channels=1):
        super(ConvNet, self).__init__()

        self.emb_dim_size = emb_dim_size

        enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        ).to(device)

        enc_conv_res = enc_conv(X_train_example.to(device))

        fc = nn.Sequential(
            # nn.Linear(64*16*24, 2048),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(enc_conv_res.size(1), emb_dim_size),
            # nn.LeakyReLU(),
        )

        self.encoder = nn.Sequential(
            enc_conv,
            fc,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=4096, output_dim=8192):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.BatchNorm1d(hidden_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.LeakyReLU()
        )

        self.mean_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        projected = self.projection_head(x)
        concat = torch.cat((x, projected), dim=-1)
        avg_pool = self.mean_pool(concat)
        max_pool = self.max_pool(concat)
        concat = torch.cat((avg_pool, max_pool), dim=-1)

        return concat


class BarlowTwins(pl.LightningModule):
    def __init__(self,
                 encoder_online: nn.Module,
                 encoder_target: nn.Module,
                 encoder_out_dim,
                 learning_rate=5e-5,
                 ):

        super().__init__()
        self.augment = aug_pipeline()

        self.learning_rate = learning_rate

        self.online = nn.Sequential(encoder_online, ProjectionHead(input_dim=encoder_out_dim))
        self.target = nn.Sequential(encoder_target, ProjectionHead(input_dim=encoder_out_dim))

        self.loss = BarlowTwinsLoss(batch_size=64)

    def forward(self, x):
        return self.online[0](x)

    def training_step(self, batch, batch_idx):
        x = batch[0]

        self.online.train()
        self.target.train()

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y_online = self.online(x1)
        y_target = self.target(x2)

        loss = self.loss(y_online, y_target)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        self.online.eval()
        self.target.eval()
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            y_online = self.online(x1)
            y_target = self.target(x2)
            loss = self.loss(y_online, y_target)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.online.train()
        self.target.train()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LinearOnlineEvaluationCallback(pl.Callback):
    def __init__(
            self,
            encoder_output_dim: int,
            num_classes: int,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
    ):
        super().__init__()
        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.augmentations = mel_aug()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.val_dataloader_iter = iter(self.val_dataloader)

        self.linear_classifier: torch.nn.Linear

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.linear_classifier = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(self.linear_classifier.parameters(), lr=1e-4)

    def extract_batch(self, batch: Sequence, device: Union[str, torch.device]):
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

        with torch.no_grad():
            features = pl_module.forward(x)

        features = features.detach()
        preds = self.linear_classifier(features)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred_labels = torch.argmax(preds, dim=1)
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

        with torch.no_grad():
            features = pl_module.forward(x)

            features = features.detach()
            preds = self.linear_classifier(features)
            loss = F.cross_entropy(preds, y)

        pl_module.train()

        pred_labels = torch.argmax(preds, dim=1)
        acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
        pl_module.log("gtzan_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("gtzan_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

# class LinearEvaluationCallback(pl.Callback):
#     def __init__(
#             self,
#             encoder_output_dim: int,
#             num_classes: int,
#             train_dataloader: torch.utils.data.DataLoader,
#             val_dataloader: torch.utils.data.DataLoader,
#             ):
#         super().__init__()

#         self.encoder_output_dim = encoder_output_dim
#         self.num_classes = num_classes

#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader

#     def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         current_epoch = trainer.current_epoch
#         if current_epoch % 5 == 4:
#             self.standard_evaluation(pl_module)

#     def standard_evaluation(self, pl_module: pl.LightningModule):
#         pl_module.linear_classifier = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
#         optimizer = torch.optim.Adam(pl_module.linear_classifier.parameters(), lr=1e-4)

#         best_val_loss = float('inf')
#         patience = 5
#         no_improvement_counter = 0
#         while no_improvement_counter < patience:
#             # Training loop
#             train_accs = []
#             for batch in self.train_dataloader:
#                 x, y = batch
#                 x, y = x.to(pl_module.device), y.to(pl_module.device)

#                 with torch.no_grad():
#                     features = pl_module.forward(x)

#                 features = features.detach()
#                 preds = pl_module.linear_classifier(features)
#                 loss = F.cross_entropy(preds, y)

#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 pred_labels = torch.argmax(preds, dim=1)
#                 acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
#                 train_accs.append(acc.item())

#             avg_train_acc = np.mean(train_accs)
#             pl_module.log("gtzan_train_acc", avg_train_acc, on_epoch=True)

#             # Validation loop
#             val_losses = []
#             val_accs = []
#             for batch in self.val_dataloader:
#                 x, y = batch
#                 x, y = x.to(pl_module.device), y.to(pl_module.device)

#                 with torch.no_grad():
#                     features = pl_module.forward(x)
#                     preds = pl_module.linear_classifier(features)
#                     loss = F.cross_entropy(preds, y)

#                 pred_labels = torch.argmax(preds, dim=1)
#                 acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
#                 val_accs.append(acc.item())
#                 val_losses.append(loss.item())

#             # Check for early stopping
#             avg_val_loss = np.mean(val_losses)
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 no_improvement_counter = 0
#             else:
#                 no_improvement_counter += 1

#             # Log average validation accuracy
#             avg_val_acc = np.mean(val_accs)
#             pl_module.log("gtzan_val_acc", avg_val_acc, on_epoch=True)
