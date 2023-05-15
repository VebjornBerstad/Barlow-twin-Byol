import pytorch_lightning as pl
import torch
import torch.nn as nn
from swarm.optimizers import LARS
import typing as t


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_=5e-3):
        super().__init__()

        self.lambda_ = lambda_

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

        return on_diag + self.lambda_ * off_diag


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

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
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
            nn.BatchNorm1d(emb_dim_size),
            nn.Tanh(),
        )

        self.encoder = nn.Sequential(
            enc_conv,
            fc,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, n_layers: int, scaling_factor: float = 1.6):
        super().__init__()

        layer_dims = [input_dim] + [int(input_dim * scaling_factor ** i) for i in range(1, n_layers + 1)]

        self.projection_head = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i + 1], bias=False),
                nn.BatchNorm1d(layer_dims[i + 1]),
                nn.LeakyReLU(),
            )
            for i in range(n_layers)
        ])

        self.mean_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.projection_head(x)


class BarlowTwins(pl.LightningModule):
    def __init__(self,
                 encoder_online: nn.Module,
                 encoder_target: nn.Module,
                 encoder_out_dim: int,
                 learning_rate: float,
                 xcorr_lambda: float,
                 augmentations: nn.Module,
                 ):

        super().__init__()
        self.augment = augmentations

        self.learning_rate = learning_rate

        self.online = nn.Sequential(encoder_online, ProjectionHead(input_dim=encoder_out_dim, n_layers=3, scaling_factor=2))
        self.target = nn.Sequential(encoder_target, ProjectionHead(input_dim=encoder_out_dim, n_layers=3, scaling_factor=2))

        self.loss = BarlowTwinsLoss(lambda_=xcorr_lambda)

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
