from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.models.base.base_config import BaseAEConfig


@dataclass
class UltraVAEConfig(BaseAEConfig):
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'l1', 'mse']. Default: 'mse'
    """

    reconstruction_loss: Literal["bce", "mse", "l1"] = "l1"

    p_do: float = 0.0

    anti_confounders_strategy: int = 0
    n_confounders: int = 0
    n_confounders_bincat: int = 0
    n_confounders_mulcat: int = 0
    n_confounders_cont: int = 0
    n_anti_confounder_layers: int = 2 #for strategy 1
    lambda_conf_corr_loss: float = 1.0 #for strategy 2

    learn_phenotypes: bool = False
    n_phenotypes: int = 0
    n_phenotype_layers: int = 2
    lambda_pheno_loss: float = 1.0
