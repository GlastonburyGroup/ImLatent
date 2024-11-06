from pydantic.dataclasses import dataclass

from pythae.models.vae import VAEConfig


@dataclass
class UltraFactorVAEConfig(VAEConfig):
    r"""
    FactorVAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'l1', 'mse']. Default: 'mse'
        gamma (float): The balancing factor before the Total Correlation. Default: 0.5

        anti_confounders_strategy (int): The strategy to handle confounders. Default: 0 Ignore, 1: Learn2ignore, 2: Neg-correlation
        learn_phenotypes (bool): Whether to learn from the supplied phenotypes or not. Default: False
    """
    gamma: float = 0.5
    uses_default_discriminator: bool = True
    discriminator_input_dim: int = None

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
