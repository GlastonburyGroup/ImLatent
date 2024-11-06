import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from pythae.data.datasets import BaseDataset
from pythae.models.base import BaseAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.nn.default_architectures import Encoder_VAE_MLP
from pythae.trainers import BaseTrainerConfig

from .ultra_cevae_config import UltraCEVAEConfig
from .ultra_cevae_utils import CESquareNoiseGenerator, UltraPredictor, anti_confounder_loss

from ......utilities import PlasmaConduit

class UltraCEVAE(BaseAE):
    """Context-encoding Variational Autoencoder model.
    Code adapted from StRegA (https://github.com/soumickmj/StRegA)
    Chatterjee et al. StRegA: Unsupervised Anomaly Detection in Brain MRIs using a Compact Context-encoding 
    Variational Autoencoder (Computers in Biology and Medicine, Oct 2022)

    Args:
        model_config (VAEConfig): The Variational Autoencoder configuration setting the main
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
        model_config: UltraCEVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "UltraVAE"

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

        self.square_noise_generator = CESquareNoiseGenerator(dim=len(self.model_config.input_dim), 
                                                            square_size=(1, np.min(self.model_config.input_dim) // self.model_config.square_size_factor), 
                                                            n_squares=(self.model_config.min_n_squares, self.model_config.max_n_squares))
        
        self.anti_confounders_strategy = model_config.anti_confounders_strategy
        self.learn_phenotypes = model_config.learn_phenotypes

        if self.learn_phenotypes:
            self.phenotype_learner = UltraPredictor(latent_dim=model_config.latent_dim, n_layers=model_config.n_phenotype_layers, n_predictions=model_config.n_phenotypes)

        default_training_config_path = "Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/celeba/base_training_config.json"
        self.default_training_config = BaseTrainerConfig.from_json_file(default_training_config_path)
        
        if self.anti_confounders_strategy == 1:
            sys.exit("anti_confounders_strategy not yet ready")
            n_preds = model_config.n_confounders_cont + model_config.n_confounders_bincat
            self.anti_counfounder_learner = UltraPredictor(latent_dim=model_config.latent_dim, n_layers=model_config.n_anti_confounder_layers, n_predictions=model_config.n_confounders)
            self.model_optims = [["model.encoder", "model.decoder"], "model.anti_counfounder_learner"]
            self.loss_optims = ["ae_val_loss", "conf_val_loss"]
            
            if self.learn_phenotypes:
                self.model_optims[0].append("model.phenotype_learner")

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

        ### Ultra VAE Part
        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        if self.learn_phenotypes:
            pheno_pred = self.phenotype_learner(z)
            if "phenotypes" in inputs.keys():
                x_pheno = inputs["phenotypes"]
                pheno_loss = F.mse_loss(pheno_pred, x_pheno) #maybe try L1?
            else:
                pheno_loss = 0
        
        if self.anti_confounders_strategy == 1:
            conf_pred = self.anti_counfounder_learner(z)
            conf_loss = 0
        elif self.anti_confounders_strategy == 2:
            conf_corr_loss = 0

        if self.anti_confounders_strategy and "has_confounders" in inputs.keys() and inputs['has_confounders']:
            conf_loss, conf_corr_loss = self._get_conf_loss(conf_pred, inputs, z)
        ########## VAE part ends

        ## Context-encoding part
        ce_tensor = self.square_noise_generator(data_shape=x.shape, dtype=x.dtype, device=x.device, 
                                                noise_val=(x.min().item(), x.max().item()))
        x_noisy = torch.where(ce_tensor != 0, ce_tensor, x)

        encoder_output_noisy = self.encoder(x_noisy)
        mu_noisy, log_var_noisy = encoder_output_noisy.embedding, encoder_output_noisy.log_covariance
        std_noisy = torch.exp(0.5 * log_var_noisy)
        z_noisy, _ = self._sample_gauss(mu_noisy, std_noisy)
        recon_x_noisy = self.decoder(z_noisy)["reconstruction"]

        ####### Context-encoding part ends

        loss, recon_loss, recon_loss_ce, kld = self.loss_function(recon_x, recon_x_noisy, x, mu, log_var, z)

        if self.learn_phenotypes:
            loss += self.model_config.lambda_pheno_loss * pheno_loss

        out = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            recon_loss_ce=recon_loss_ce,
            reg_loss=kld,
            autoencoder_loss=loss,
            recon_x=recon_x,
            z=z,
        )

        if self.learn_phenotypes:
            out.pheno_loss = pheno_loss
            out.pheno_pred = pheno_pred
            
        if self.anti_confounders_strategy == 1:
            out.conf_loss = conf_loss
            out.conf_pred = conf_pred
        elif self.anti_confounders_strategy == 2:
            out.conf_corr_loss = conf_corr_loss

        return out

    def loss_function(self, recon_x, recon_x_noisy, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            recon_loss_ce = F.mse_loss(
                recon_x_noisy.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            recon_loss_ce = F.binary_cross_entropy(
                recon_x_noisy.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "l1":

            recon_loss = F.l1_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            recon_loss_ce = F.l1_loss(
                recon_x_noisy.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        final_loss = self.model_config.vae_factor*(KLD + (self.model_config.recon_factor*recon_loss)) + self.model_config.ce_factor*recon_loss_ce

        return final_loss.mean(dim=0), recon_loss.mean(dim=0), recon_loss_ce.mean(dim=0), KLD.mean(dim=0)

    def _get_conf_loss(self, conf_pred, inputs, z):
        if self.anti_confounders_strategy == 1:   
            conf_loss = 0
            if "confounders_continuous" in inputs.keys():
                conf_loss += F.mse_loss(conf_pred, inputs["confounders_continuous"]) #maybe try L1?
            if "confounders_binary" in inputs.keys():
                conf_loss += F.binary_cross_entropy(conf_pred, inputs["confounders_binary"])
            if "confounders_multicat" in inputs.keys():
                conf_loss += F.cross_entropy(conf_pred, inputs["confounders_multicat"])
            return conf_loss, 0
        elif self.anti_confounders_strategy == 2:
            x_conf = []
            if "confounders_continuous" in inputs.keys():
                x_conf.append(inputs["confounders_continuous"])
            if "confounders_binary" in inputs.keys():
                x_conf.append(inputs["confounders_binary"])
            if "confounders_multicat" in inputs.keys():
                x_conf.append(inputs["confounders_multicat"])
            return 0, anti_confounder_loss(z, torch.cat(x_conf, dim=-1))
        else:
            raise ValueError("anti_confounders_strategy must be 0, 1 or 2")

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

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

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                elif self.model_config.reconstruction_loss == "l1":

                    log_p_x_given_z = -0.5 * F.l1_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
    
    def custom_step(self, batch, **kwargs):
        inp = PlasmaConduit(data=batch['inp']['data'])
        if self.anti_confounders_strategy:
            inp.has_confounders = False
            if 'confounders_continuous' in batch:
                inp.confounders_continuous = batch['confounders_continuous']
                inp.has_confounders = True
            if 'confounders_binary' in batch:
                inp.confounders_binary = batch['confounders_binary']
                inp.has_confounders = True
            if 'confounders_multicat' in batch:
                inp.confounders_multicat = batch['confounders_multicat']
                inp.has_confounders = True
        if self.learn_phenotypes and 'phenotypes' in batch:
            inp.phenotypes = batch['phenotypes']
        model_output = self(inp, return_only_recon=False)
            
        if "optimisers" in kwargs and kwargs["optimisers"] is not None:
            sys.exit("Only auto_optim is currently supported")
            if self.learn_phenotypes and self.anti_confounders_strategy == 2:
                total_loss = model_output.autoencoder_loss + self.model_config.lambda_pheno_loss * model_output.pheno_loss - self.model_config.lambda_conf_corr_loss * model_output.conf_corr_loss
                kwargs["manual_backward"](total_loss, retain_graph=True)
            elif self.learn_phenotypes:
                total_loss = model_output.autoencoder_loss + self.model_config.lambda_pheno_loss * model_output.pheno_loss
                kwargs["manual_backward"](total_loss, retain_graph=True)
            else:
                kwargs["manual_backward"](model_output.autoencoder_loss, retain_graph=True)

            if self.anti_confounders_strategy == 1:
                kwargs["manual_backward"](model_output.conf_loss, retain_graph=True)

            kwargs["manual_backward"](model_output.discriminator_loss)

            if kwargs["step_optim"]:
                if kwargs["grad_clipping"] is not None:
                    kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][0], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                    kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][1], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                    if self.anti_confounders_strategy == 1:
                        kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][2], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                kwargs["optimisers"][0].step()
                kwargs["optimisers"][0].zero_grad()
                kwargs["optimisers"][1].step()
                kwargs["optimisers"][1].zero_grad()
                if self.anti_confounders_strategy == 1:
                    kwargs["optimisers"][2].step()
                    kwargs["optimisers"][2].zero_grad()

        return model_output

    def predict(self, inputs: torch.Tensor) -> PlasmaConduit:
        """The input data is encoded and decoded without computing loss.
        For other models, predict is accessed from the base_model. But for the ultra models, this must be overriden - to handle the prediction of phenotypes and/or confounders.

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction and embedding (+ predicted phenotypes) (+ predicted confounders)
        """
        z = self.encoder(inputs).embedding
        recon_x = self.decoder(z)["reconstruction"]

        output = PlasmaConduit(
            recon_x=recon_x,
            embedding=z,
        )

        if self.learn_phenotypes:
            output.pheno_pred = self.phenotype_learner(z)

        if self.anti_confounders_strategy == 1:
            output.conf_pred = self.anti_counfounder_learner(z)

        return output
