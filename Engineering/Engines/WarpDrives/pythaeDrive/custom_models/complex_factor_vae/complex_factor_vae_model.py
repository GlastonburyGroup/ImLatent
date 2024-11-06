import os
from typing import Optional, Union, Callable

import torch
import torchcomplex
import torch.nn.functional as F

from pythae.customexception import BadInheritanceError
from pythae.data.datasets import BaseDataset
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.models.vae import VAE

from .complex_factor_vae_config import CVFactorVAEConfig
from .complex_factor_vae_utils import CVFactorVAEDiscriminator


class CVFactorVAE(VAE):
    """
    FactorVAE model.

    Args:
        model_config (FactorVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        custom_recon_loss_func: (torch.nn.Module or Callable): A custom loss function for calculation the reconstruction loss. 
            This is only used when the `reconstruction_loss` parameter in `model_config` is set to `custom`. This can be either
            an instance of `torch.nn.Module` or a callable function. In either case, the function must take the following arguments:
            - `recon_x`: The reconstructed data
            - `x`: The original data. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: CVFactorVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        custom_recon_loss_func: Optional[Union[torch.nn.Module, Callable]] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        
        self.discriminator = CVFactorVAEDiscriminator(latent_dim=model_config.latent_dim)
        
        self.model_name = "FactorVAE"
        self.gamma = model_config.gamma
        self.custom_recon_loss_func = custom_recon_loss_func

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError(
                (
                    "Discriminator must inherit from BaseDiscriminator class from "
                    "pythae.models.base_architectures.BaseDiscriminator. Refer to documentation."
                )
            )

        self.discriminator = discriminator

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x_in = inputs["data"]
        if x_in.shape[0] <= 1:
            raise ArithmeticError(
                "At least 2 samples in a batch are required for the `FactorVAE` model"
            )

        idx = torch.randperm(x_in.shape[0])
        idx_1 = idx[int(x_in.shape[0] / 2) :]
        idx_2 = idx[: int(x_in.shape[0] / 2)]

        # first batch
        x = inputs["data"][idx_1]

        if self.model_config.reconstruction_loss == "custom_masked":
            if "mask" not in inputs.keys():
                raise ValueError(
                    "No mask not present in the input for `custom_masked` reconstruction loss"
                )
            mask = inputs["mask"][idx_1]

        encoder_output = self.encoder(x)

        mu, log_var, log_pvar = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance

        z = self._sample_gauss_complex(mu, log_var, log_pvar)
        recon_x = self.decoder(z)["reconstruction"]

        # second batch
        x_bis = inputs["data"][idx_2]

        encoder_output = self.encoder(x_bis)

        mu_bis, log_var_bis, log_pvar_bis = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance

        z_bis = self._sample_gauss_complex(mu_bis, log_var_bis, log_pvar_bis)

        z_bis_permuted = self._permute_dims(z_bis).detach()

        if self.model_config.reconstruction_loss != "custom_masked":
            recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
                recon_x, x, mu, log_var, log_pvar, z, z_bis_permuted
            )
        else:
            recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
                recon_x, x, mu, log_var, log_pvar, z, z_bis_permuted, mask
            )


        loss = autoencoder_loss + discriminator_loss

        return ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            autoencoder_loss=autoencoder_loss,
            discriminator_loss=discriminator_loss,
            recon_x=recon_x,
            recon_x_indices=idx_1,
            z=z,
            z_bis_permuted=z_bis_permuted,
        )

    def loss_function(self, recon_x, x, mu, log_var, log_pvar, z, z_bis_permuted, mask=None):
        if self.model_config.reconstruction_loss == "crecon":

            recon_loss = self._recon_loss_mae_complex(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1)).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "basel1":

            recon_loss = F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "custom":

            recon_loss = self.custom_recon_loss_func(recon_x, x)

        elif self.model_config.reconstruction_loss == "custom_masked":

            recon_loss = self.custom_recon_loss_func(recon_x, x, mask)

        KLD = self._calculate_kld_complex(mu, log_var, log_pvar)

        latent_adversarial_score = self.discriminator(z)

        TC = (latent_adversarial_score[:, 0] - latent_adversarial_score[:, 1]).mean()
        autoencoder_loss = recon_loss + KLD + self.gamma * TC

        # discriminator loss
        permuted_latent_adversarial_score = self.discriminator(z_bis_permuted)

        true_labels = (
            torch.ones(z_bis_permuted.shape[0], requires_grad=False)
            .type(torch.LongTensor)
            .to(z.device)
        )
        fake_labels = (
            torch.zeros(z.shape[0], requires_grad=False)
            .type(torch.LongTensor)
            .to(z.device)
        )

        TC_permuted = F.cross_entropy(
            latent_adversarial_score, fake_labels
        ) + F.cross_entropy(permuted_latent_adversarial_score, true_labels)

        discriminator_loss = 0.5 * TC_permuted

        return (
            (recon_loss).mean(dim=0),
            (autoencoder_loss).mean(dim=0),
            (discriminator_loss).mean(dim=0),
        )

    def reconstruct(self, inputs: torch.Tensor):
        """This function returns the reconstructions of given input data.

        Args:
            inputs (torch.Tensor): The inputs data to be reconstructed of shape [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape

        Returns:
            torch.Tensor: A tensor of shape [B x input_dim] containing the reconstructed samples.
        """
        encoder_output = self.encoder(inputs)

        mu, log_var, log_pvar = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance

        z = self._sample_gauss_complex(mu, log_var, log_pvar)
        
        return self.decoder(z)["reconstruction"]

    def interpolate(
        self,
        starting_inputs: torch.Tensor,
        ending_inputs: torch.Tensor,
        granularity: int = 10,
    ):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], (
            "The number of starting_inputs should equal the number of ending_inputs. Got "
            f"{starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} "
            "for endinging_inputs."
        )

        encoder_output = self.encoder(starting_inputs)
        mu, log_var, log_pvar = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance
        starting_z = self._sample_gauss_complex(mu, log_var, log_pvar)

        encoder_output = self.encoder(ending_inputs)
        mu, log_var, log_pvar = encoder_output.embedding, encoder_output.log_covariance, encoder_output.log_psuedocovariance
        ending_z = self._sample_gauss_complex(mu, log_var, log_pvar)

        t = torch.linspace(0, 1, granularity).to(starting_inputs.device)
        intep_line = (
            torch.kron(
                starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)
            )
            + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        ).reshape((starting_z.shape[0] * t.shape[0],) + (starting_z.shape[1:]))

        return self.decoder(intep_line).reconstruction.reshape(
            (
                starting_inputs.shape[0],
                t.shape[0],
            )
            + (starting_inputs.shape[1:])
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _sample_gauss_complex(self, mu, logvar, logdelta):
        # Reparametrization trick for complex Gaussian
        logvar = torch.clamp(logvar, min=-88, max=88) #to avoid over and underflow. 88 is generally the point for float32
        logdelta = torchcomplex.clamp(logdelta, min=-88, max=88)

        sigma = torch.exp(logvar)
        delta = torch.exp(logdelta)

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
        logvar = torch.clamp(logvar, min=-88, max=88) #to avoid over and underflow. 88 is generally the point for float32
        logdelta = torchcomplex.clamp(logdelta, min=-88, max=88)

        sigma = torch.exp(logvar)
        sigma_squared = sigma**2
        delta_squared = torch.exp((2 * logdelta))  # Since (exp(log(delta)))^2 = exp(2*logdelta) [for better numerical stability, in place of torch.exp(logdelta)]

        # to make sure delta^2<=sigma^2
        mag_delta_squared = torch.abs(delta_squared) + torch.finfo(torch.float32).eps
        new_element = delta_squared * ((sigma_squared * 0.99 / mag_delta_squared) + 0j)  
        delta_squared = torch.where(mag_delta_squared >= (sigma_squared), new_element, delta_squared)

        return (mu * mu.conj()).sum(-1).real + (sigma - 1 - 0.5 * torch.log(sigma_squared - delta_squared + torch.finfo(torch.float32).eps)).abs().sum(-1)

    def _permute_dims(self, z):
        permuted = torch.zeros_like(z)

        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0]).to(z.device)
            permuted[:, i] = z[perms, i]

        return permuted

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