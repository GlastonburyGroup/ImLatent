import os
from typing import Optional

import numpy as np
import torch
import torchcomplex
import torch.nn.functional as F

from pythae.data.datasets import BaseDataset
from pythae.models.base import BaseAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP

from .complex_vae_config import CVVAEConfig

def complex_clamp(input, min=None, max=None):
    # convert to polar coordinates
    magnitude = torch.abs(input)
    angle = torch.angle(input)

    # clamp the magnitude
    magnitude = torch.clamp(magnitude, min=min, max=max)

    # convert back to Cartesian coordinates
    clamped_real = magnitude * torch.cos(angle)
    clamped_imag = magnitude * torch.sin(angle)

    return torch.complex(clamped_real, clamped_imag)


class CVVAE(BaseAE):
    """Vanilla Complex-valued Variational Autoencoder model.

    Args:
        model_config (CVVAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

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
        model_config: CVVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "VAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var, log_pvar = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance

        z = self._sample_gauss_complex(mu, log_var, log_pvar)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, log_pvar)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, log_pvar):
        if self.model_config.reconstruction_loss == "crecon":

            recon_loss = self._recon_loss_mae_complex(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "basel1":

            recon_loss = F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = self._calculate_kld_complex(mu, log_var, log_pvar)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _sample_gauss_complex(self, mu, logvar, logdelta):
        # Reparametrization trick for complex Gaussian
        logvar = torch.clamp(logvar, min=-80, max=80) #to avoid over and underflow. 88 is generally the point for float32
        logdelta = complex_clamp(logdelta, min=-80, max=80)

        sigma = torch.exp(logvar) + torch.finfo(torch.float32).eps
        delta = torch.exp(logdelta) + torch.finfo(torch.float32).eps

        # to make sure delta<=sigma
        mag_delta = abs(delta) + torch.finfo(torch.float32).eps
        new_element = delta * ((sigma * 0.99 / mag_delta) + 0j)
        delta = torch.where(mag_delta >= abs(sigma), new_element, delta)
        
        denom = torch.sqrt((2 * sigma) + (2 * delta.real)) + torch.finfo(torch.float32).eps

        kappa_x = (sigma + delta) / denom
        kappa_y = 1j * torch.sqrt(sigma**2 - torch.abs(delta)**2) / denom
        
        eps_x = torch.randn_like(sigma)
        eps_y = torch.randn_like(sigma)

        return mu + kappa_x * eps_x + kappa_y * eps_y

    def _calculate_kld(self, mu, log_var):
        return-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    
    def _calculate_kld_complex_orig_(self, mu, logvar, delta):
        return (mu * mu.conj()).sum(-1).real + (logvar.exp() - 1 - 0.5 * torch.log(logvar.exp() - torch.abs(delta)**2)).abs().sum(-1)  
    
    def _calculate_kld_complex(self, mu, logvar, logdelta):
        logvar = torch.clamp(logvar, min=-80, max=80) #to avoid over and underflow. 88 is generally the point for float32
        logdelta = complex_clamp(logdelta, min=-80, max=80)

        sigma = torch.exp(logvar) + torch.finfo(torch.float32).eps
        sigma_squared = sigma**2
        delta_squared = torch.exp((2 * logdelta)) + torch.finfo(torch.float32).eps  # Since (exp(log(delta)))^2 = exp(2*logdelta) [for better numerical stability, in place of torch.exp(logdelta)]

        # to make sure delta^2<=sigma^2
        mag_delta_squared = torch.abs(delta_squared) + torch.finfo(torch.float32).eps
        new_element = delta_squared * ((sigma_squared * 0.99 / mag_delta_squared) + 0j)  
        delta_squared = torch.where(mag_delta_squared >= (sigma_squared), new_element, delta_squared)

        return (mu * mu.conj()).sum(-1).real + (sigma - 1 - 0.5 * torch.log(sigma_squared - delta_squared + torch.finfo(torch.float32).eps)).abs().sum(-1)

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):
                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                z = self._sample_gauss_complex(mu, log_var, log_pvar)

                log_var = torch.clamp(log_var, min=-88, max=88)
                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "crecon":

                    log_p_x_given_z = -0.5 * self._recon_loss_mae_complex(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )

                elif self.model_config.reconstruction_loss == "basel1":

                    log_p_x_given_z = -0.5 *F.l1_loss(
                        recon_x.reshape(x.shape[0], -1),
                        x.reshape(x.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)

    def _recon_loss_mae(self, x, x_hat):
        return torch.abs(x - x_hat).sum()

    def _recon_loss_mae_complex(self, x, x_hat):    
        '''
        Xie et al. Complex Recurrent Variational Autoencoder with Application to Speech Enhancement. 2023. arXiv:2204.02195v2
        '''
        real_loss = torch.abs(x.real - x_hat.real).sum() # Calculate the L1 loss between the real parts of x and x_hat    
        imag_loss = torch.abs(x.imag - x_hat.imag).sum() # Calculate the L1 loss between the imaginary parts of x and x_hat    
        mag_loss = torch.abs(torch.abs(x) - torch.abs(x_hat)).sum() # Calculate the L1 loss between the magnitudes of x and x_hat
        return real_loss + imag_loss + mag_loss