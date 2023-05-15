"""Project models are defined here."""

import pytorch_lightning as pl
import torch as T

from swarm.losses import CrossCorrelationLoss


class Encoder(T.nn.Module):
    """CNN used as an encoder for the Barlow Twins model.
    """

    def __init__(
        self,
        X_train_example: T.Tensor,
        device: T.device | str,
        emb_dim_size: int,
        in_channels: int = 1
    ):
        """
        Args:
            X_train_example: Input data example, to calculate the output shape of the conv layers.
            device: The device to use when running the training example through the conv net.
            emb_dim_size: The output size of the encoder.
            in_channels: The number of channels in the input data (mono expected).
        """
        super(Encoder, self).__init__()

        self.emb_dim_size = emb_dim_size

        # 2 layers were used in BYOL-a (AudioNTT encoder), but led to a lot of parameters in the model.
        # Chose to use the same structure as in BYOL-A, but replaced ReLU activations for LeakyReLU
        # to ensure proper gradient flow.
        # See: https://github.com/nttcslab/byol-a/blob/master/v2/byol_a2/models.py
        enc_conv = T.nn.Sequential(
            T.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding='same'),
            T.nn.BatchNorm2d(64),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(kernel_size=2, stride=2),

            T.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            T.nn.BatchNorm2d(64),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(kernel_size=2, stride=2),

            T.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            T.nn.BatchNorm2d(64),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(kernel_size=2, stride=2),

            T.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            T.nn.BatchNorm2d(64),
            T.nn.LeakyReLU(),
            T.nn.MaxPool2d(kernel_size=2, stride=2),

            T.nn.Flatten(),
        ).to(device)

        enc_conv_res = enc_conv(X_train_example.to(device))

        fc = T.nn.Sequential(
            T.nn.Linear(enc_conv_res.size(1), emb_dim_size),
            T.nn.BatchNorm1d(emb_dim_size),
            T.nn.Tanh(),
        )

        self.encoder = T.nn.Sequential(
            enc_conv,
            fc,
        )

    def forward(self, x: T.Tensor):
        """
        Args:
            x: Input data.

        Returns:
            The encoded input data.
        """
        x = self.encoder(x)
        return x


class Projector(T.nn.Module):
    """The Barlow twins projector. Linear layers used as an upscaler.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        scaling_factor: float = 1.6
    ):
        """

        Args:
            input_dim: The input dimension of the projector.
            n_layers: The number of linear layers to use.
            scaling_factor: The scaling factor to use for each linear layer. Neurons from layer to
                layer grows exponentially with this factor, with 2^i * input_dim neurons in layer i.
        """
        super().__init__()

        # The number of input and out neurons in the layers.
        layer_dims = [input_dim] + [int(input_dim * scaling_factor ** i) for i in range(1, n_layers + 1)]

        self.projection_head = T.nn.Sequential(*[
            T.nn.Sequential(
                T.nn.Linear(layer_dims[i], layer_dims[i + 1], bias=False),
                T.nn.BatchNorm1d(layer_dims[i + 1]),
                T.nn.LeakyReLU(),
            )
            for i in range(n_layers)
        ])

    def forward(self, x: T.Tensor):
        """
        Args:
            x: The input tensor.

        Returns:
            The projected input tensor.
        """
        return self.projection_head(x)


class BarlowTwins(pl.LightningModule):
    """The PyTorch Lightning module for our Barlow twins implementation.

    The module is a wrapper around the logic for training the model.

    We have implemented it to the best of our abilities follow the original paper:
    J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny,
        ‘Barlow Twins: Self-Supervised Learning via Redundancy Reduction’,
        2021, doi: 10.48550/ARXIV.2103.03230.

    """

    def __init__(
        self,
        encoder_online: T.nn.Module,
        encoder_target: T.nn.Module,
        encoder_out_dim: int,
        learning_rate: float,
        xcorr_lambda: float,
        augmentations: T.nn.Module,
        val_loss_metric: str = 'val_loss',
    ):
        """
        To follow BYOL-A's augmentation setup, use the following augmentations from swarm.augmentations:
        1) Pre-normlization, 2) Mixup, 3) Random resize crop, 4) Random linear fader,5) Post-normalization

        Args:
            encoder_online: The online encoder. Can be any PyTorch module, as long as the output dimensions
                matches those of encoder_target.
            encoder_target: The target encoder.
            encoder_out_dim: The output dimension of the encoder, i.e. the representation's dimension.
            learning_rate: The learning rate to use for the optimizer.
            xcorr_lambda: The lambda parameter for the cross correlation loss.
            augmentations: The augmentations to use during training.
        """

        super().__init__()
        self.augment = augmentations

        self.learning_rate = learning_rate

        self.target = T.nn.Sequential(encoder_online, Projector(input_dim=encoder_out_dim, n_layers=3, scaling_factor=1.5))
        self.online = T.nn.Sequential(encoder_target, Projector(input_dim=encoder_out_dim, n_layers=3, scaling_factor=1.5))

        self.loss = CrossCorrelationLoss(lambda_=xcorr_lambda)

    def forward(self, x: T.Tensor):
        """
        Args:
            x: Input data.

        Returns:
            The output of the target encoder.
        """

        # Gets the encoder part from the target.
        return self.target[0](x)

    def training_step(self, batch: T.Tensor, *args, **kwargs):
        """Performs a training step on a batch.

        Args:
            batch: The batch to train on.

        Returns:
            The loss of the training step.
        """
        x, _ = batch

        self.target.train()
        self.online.train()

        with T.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y_online = self.target(x1)
        y_target = self.online(x2)

        loss = self.loss(y_online, y_target)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        """Performs a validation step on a batch.

        Args:
            batch: The batch to validate on.
        """
        self.target.eval()
        self.online.eval()
        x = batch[0]

        with T.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            y_online = self.target(x1)
            y_target = self.online(x2)
            loss = self.loss(y_online, y_target)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.target.train()
        self.online.train()

    def configure_optimizers(self):
        """Configure the optimizer to use for training.

        Returns:
            The optimizer to use for training.
        """
        optimizer = T.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
