from dataclasses import dataclass

import dvc.api as dvc_api


@dataclass
class TrainingConfig:
    """Contains information about the DVC training params."""
    batch_size: int
    val_split: float
    lr: float
    xcorr_lambda: float
    emb_dim_size: int
    early_stopping_patience: int
    max_epochs: int


def parse_dvc_training_config() -> TrainingConfig:
    """
    Returns:
        A TrainingConfig object containing the training params.
    """
    params = dvc_api.params_show()
    training_params = params['training']

    return TrainingConfig(
        batch_size=training_params['batch_size'],
        val_split=training_params['val_split'],
        lr=training_params['lr'],
        xcorr_lambda=training_params['xcorr_lambda'],
        emb_dim_size=training_params['emb_dim_size'],
        early_stopping_patience=training_params['early_stopping_patience'],
        max_epochs=training_params['max_epochs'],
    )
