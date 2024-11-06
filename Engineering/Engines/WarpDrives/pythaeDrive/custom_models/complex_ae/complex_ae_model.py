import os
from typing import Optional

import torch

import torch.nn.functional as F

from pythae.data.datasets import BaseDataset
from pythae.models.base import BaseAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_AE_MLP


from .complex_ae_config import CVAEConfig


class CVAE(BaseAE):
    """Vanilla Autoencoder model.

    Args:
        model_config (AEConfig): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: CVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x).embedding
        recon_x = self.decoder(z)["reconstruction"]

        loss = self.loss_function(recon_x, x)

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

    def loss_function(self, recon_x, x):
        if self.model_config.reconstruction_loss == "crecon":

            return  self._recon_loss_mae_complex(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)
        
        elif self.model_config.reconstruction_loss == "dualcoordsL1":

            return  self._dualcoords_loss_mae_complex(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)
        
        elif self.model_config.reconstruction_loss == "dualcoords":

            return  self._dualcoords_loss(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)
        
        elif self.model_config.reconstruction_loss == "polar":

            return  self._polar_loss(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "basel1":

            return F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).mean(dim=0)

        elif self.model_config.reconstruction_loss == "basel1mean":

            return F.l1_loss(recon_x, x)

    def _recon_loss_mae_complex(self, x, x_hat):    
        '''
        Xie et al. Complex Recurrent Variational Autoencoder with Application to Speech Enhancement. 2023. arXiv:2204.02195v2
        '''
        real_loss = torch.abs(x.real - x_hat.real).sum() # Calculate the L1 loss between the real parts of x and x_hat    
        imag_loss = torch.abs(x.imag - x_hat.imag).sum() # Calculate the L1 loss between the imaginary parts of x and x_hat    
        mag_loss = torch.abs(torch.abs(x) - torch.abs(x_hat)).sum() # Calculate the L1 loss between the magnitudes of x and x_hat
        return real_loss + imag_loss + mag_loss
    
    def _dualcoords_loss_mae_complex(self, x, x_hat):    
        '''
        It's an extension to the recon loss function proposed by Xie et al that includes the phase information
        '''
        real_loss = torch.abs(x.real - x_hat.real).sum()
        imag_loss = torch.abs(x.imag - x_hat.imag).sum()
        mag_loss = torch.abs(torch.abs(x) - torch.abs(x_hat)).sum()
        phase_loss = torch.abs(torch.angle(x) - torch.angle(x_hat)).sum()
        return real_loss + imag_loss + mag_loss + phase_loss
    
    def _dualcoords_loss(self, x, x_hat):    
        '''
        Novel loss function that computes L1 loss for real, imag, and magnitude, and cosine similarity for phase
        '''
        real_loss = F.l1_loss(x_hat.real, x.real)
        imag_loss = F.l1_loss(x_hat.imag, x.imag)
        mag_loss = F.l1_loss(torch.abs(x_hat), torch.abs(x))
        phase_loss = 1 - F.cosine_similarity(x_hat.angle(), x.angle())
        return real_loss + imag_loss + mag_loss + phase_loss
    
    def _polar_loss(self, x, x_hat):
        '''
        Novel loss function that computes L1 loss for magnitude, and cosine similarity for phase
        '''
        magnitude_loss = F.l1_loss(torch.abs(x_hat), torch.abs(x))
        phase_loss = 1 - F.cosine_similarity(x_hat.angle(), x.angle())
        return magnitude_loss + phase_loss



