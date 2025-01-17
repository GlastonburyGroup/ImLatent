import os
from typing import Optional, Union, Callable

import torch
import torch.nn.functional as F

from pythae.customexception import BadInheritanceError
from pythae.data.datasets import BaseDataset
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.models.vae import VAE

from .factor_vae_config import FactorVAEConfig
from .factor_vae_utils import FactorVAEDiscriminator, CVFactorVAEDiscriminator


class FactorVAE(VAE):
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
        model_config: FactorVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        custom_recon_loss_func: Optional[Union[torch.nn.Module, Callable]] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        
        if model_config.is_cv:
            self.discriminator = CVFactorVAEDiscriminator(latent_dim=model_config.latent_dim)
        else:
            self.discriminator = FactorVAEDiscriminator(latent_dim=model_config.latent_dim)

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

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        # second batch
        x_bis = inputs["data"][idx_2]

        encoder_output = self.encoder(x_bis)

        mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance

        std_bis = torch.exp(0.5 * log_var_bis)
        z_bis, _ = self._sample_gauss(mu_bis, std_bis)

        z_bis_permuted = self._permute_dims(z_bis).detach()

        if self.model_config.reconstruction_loss != "custom_masked":
            recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
                recon_x, x, mu, log_var, z, z_bis_permuted
            )
        else:
            recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
                recon_x, x, mu, log_var, z, z_bis_permuted, mask
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

    def loss_function(self, recon_x, x, mu, log_var, z, z_bis_permuted, mask=None):

        N = z.shape[0]  # batch size

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "l1":

            recon_loss = F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "custom":

            recon_loss = self.custom_recon_loss_func(recon_x, x)

        elif self.model_config.reconstruction_loss == "custom_masked":

            recon_loss = self.custom_recon_loss_func(recon_x, x, mask)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

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

        if torch.is_complex(latent_adversarial_score):
            latent_adversarial_score = torch.abs(latent_adversarial_score)
            permuted_latent_adversarial_score = torch.abs(permuted_latent_adversarial_score)

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

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
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
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        starting_z, _ = self._sample_gauss(mu, std)

        encoder_output = self.encoder(ending_inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        ending_z, _ = self._sample_gauss(mu, std)

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

    def _permute_dims(self, z):
        permuted = torch.zeros_like(z)

        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0]).to(z.device)
            permuted[:, i] = z[perms, i]

        return permuted