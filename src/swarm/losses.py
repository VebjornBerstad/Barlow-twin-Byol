"""Custom loss functions are defined here.
"""

import torch as T


class CrossCorrelationLoss(T.nn.Module):
    """Cross correlation loss, as defined in the Barlow Twins paper.
    """

    def __init__(self, lambda_: float = 5e-3, eps: float = 1e-12):
        """
        Args:
            lambda_: The lambda parameter for the cross correlation loss.
            eps: Epsilon used for numeric stability.
        """
        super().__init__()

        self.lambda_ = lambda_
        self.eps = eps

    def off_diagonal_ele(self, x: T.Tensor):
        """
        Returns the off-diagonal elements of a square matrix.

        Ref. implementation from:
        https://github.com/jonahanton/SSL_audio/blob/master/utils/utils.py

        Args:
            x: A square matrix.

        Returns:
            A vector containing the off-diagonal elements of x.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1: T.Tensor, z2: T.Tensor):
        """Calculates the cross-correlation loss between the two input tensors z1 and z2.

        Ref. Algorithm 1 from:
        J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny,
            ‘Barlow Twins: Self-Supervised Learning via Redundancy Reduction’,
            2021, doi: 10.48550/ARXIV.2103.03230.

        Args:
            z1: The first embedding tensor.
            z2: The second embedding tensor.

        Returns:
            The cross-correlation loss between the two input tensors.
        """
        assert z1.shape == z2.shape

        # normalize repr. along the batch dimension
        z1_norm = (z1 - T.mean(z1, dim=0)) / (T.std(z1, dim=0) + self.eps)
        z2_norm = (z2 - T.mean(z2, dim=0)) / (T.std(z2, dim=0) + self.eps)

        # cross-correlation matrix
        cross_corr = T.matmul(z1_norm.T, z2_norm) / z1.shape[0]

        on_diag = T.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        loss = on_diag + self.lambda_ * off_diag

        return loss
