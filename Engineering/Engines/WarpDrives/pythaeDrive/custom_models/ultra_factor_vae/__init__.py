"""This module is the implementation of the FactorVAE proposed in
(https://arxiv.org/abs/1802.05983).
This model adds a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and the Total Correlation.


Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .ultra_factor_vae_config import UltraFactorVAEConfig
from .ultra_factor_vae_model import UltraFactorVAE

__all__ = ["UltraFactorVAE", "UltraFactorVAEConfig"]
