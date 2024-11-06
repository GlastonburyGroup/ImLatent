from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.models.base.base_config import BaseAEConfig


@dataclass
class UltraCEVAEConfig(BaseAEConfig):
    """CE-VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'l1', 'mse']. Default: 'mse'
    """

    reconstruction_loss: Literal["bce", "mse", "l1"] = "l1"

    ce_factor: float = 1.0 #StRegA uses 0.5
    recon_factor: float = 1.0 #StRegA uses 1.0
    vae_factor: float = 1.0 #StRegA uses 0.5 (but for better latent generation, this should be 1.0)
    
    square_size_factor: float = 2 #StRegA uses 2: the square will be maximum half the size of the image
    min_n_squares: int = 1 #If this is set to 0 (like the original StRegA), then the the sum of ce_factor and recon_factor should be 1.
    max_n_squares: int = 3 #StRegA uses 3

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
