from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.models.base.base_config import BaseAEConfig


@dataclass
class iVAEsConfig(BaseAEConfig):
    """VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'l1', 'mse']. Default: 'mse'
    """

    ivae_mode: Literal["iVAE", "IDVAE", "CI-iVAE"] = "CI-iVAE"
    
    reconstruction_loss: Literal["bce", "mse", "l1"] = "l1"

    n_phenotypes: int = 0
    learn_phenotypes: bool = True #Should never be changed. It's just a placeholder to make the pipeline work.
    
    aggressive_post: bool = False #Not implemented
    kl_annealing: bool = False #Not implemented
    
    beta: float = 0.001 #the coefficient of kl divergence terms
    alpha_step: float = 0.025
    
    #newly added params
    hidden_nodes_label_prior: int = 256
    hidden_nodes_label_decoder: int = 256 #only used for IDVAE
    M: int = 100 #number of samples to draw from the posterior and prior distributions
    predict_reparameterised: bool = False #if True, the model will predict the reparameterised latent space (considering both mean and std), otherwise it will predict the latent space. Default is True according to the CI-iVAE code. But why?
    orig_recon_loss: bool = True