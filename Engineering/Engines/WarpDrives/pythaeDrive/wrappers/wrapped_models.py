import os
from typing import Optional, Union, Callable
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.data.datasets import BaseDataset
from pythae.models.base.base_utils import ModelOutput
from pythae.models import FactorVAE, FactorVAEConfig
from pythae.models.nn import BaseEncoder, BaseDecoder

from .utils import remove_attributes
from .base_wrapper import LSTMBaseWrapper

class LSTMFactorVAE(FactorVAE,LSTMBaseWrapper):
    def __init__(
        self,
        model_config: FactorVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        custom_recon_loss_func: Optional[Union[torch.nn.Module, Callable]] = None,
    ):
        model_config.latent_dim = model_config.latent_dim*2 if model_config.lstm_bidirectional else model_config.latent_dim
        FactorVAE.__init__(self, model_config, encoder, decoder, custom_recon_loss_func)
        LSTMBaseWrapper.__init__(self, model_config, requires_log=True)
    
    def obtain_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        return z
        
    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        x_in = inputs["data"]
        if x_in.shape[0] <= 1:
            raise ArithmeticError(
                "At least 2 samples in a batch are required for the `FactorVAE` model"
            )
        nTP = x_in.shape[1] // self.n_channels

        idx = torch.randperm(x_in.shape[0])
        idx_1 = idx[int(x_in.shape[0] / 2) :]
        idx_2 = idx[: int(x_in.shape[0] / 2)]

        # first batch
        x = inputs["data"][idx_1]
        bs = x.shape[0]

        # Process the first batch make it LSTM-friendly
        x = self.im_reshape(x, nTP)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        # encoded my to LSTM-ed mu (first batch)
        mu, (mu_hidden, mu_cell) = self.lstm_encode(mu, self.lstm_emb, nTP)
        log_var, (log_hidden, log_cell) = self.lstm_encode(log_var, self.lstm_log, nTP)

        z = self.obtain_z(mu, log_var)
        z_hidden = self.obtain_z(mu_hidden, log_hidden)
        z_cell = self.obtain_z(mu_cell, log_cell)

        recon_lstm = self.lstm_decode(z_hidden[-1] if self.decode_last_hidden else z, z_hidden, z_cell, self.lstm_dec, bs, nTP)

        recon_x = self.im_unreshape(self.decoder(recon_lstm)["reconstruction"], nTP)

        # second batch
        x_bis = inputs["data"][idx_2]

        # Process the second batch to make it LSTM-friendly
        x_bis = self.im_reshape(x_bis, nTP)

        encoder_output = self.encoder(x_bis)

        mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance

        # encoded my to LSTM-ed mu (second batch)
        mu_bis, _ = self.lstm_encode(mu_bis, self.lstm_emb, nTP)
        log_var_bis, _ = self.lstm_encode(log_var_bis, self.lstm_log, nTP)

        z_bis = self.obtain_z(mu_bis, log_var_bis)

        #change the variables to make it non-LSTM
        actual_latent_dim = self.latent_dim*2 if self.bidirectional else self.latent_dim
        mu = mu.reshape(bs*nTP, actual_latent_dim)
        log_var = log_var.reshape(bs*nTP, actual_latent_dim)
        z_bis = z_bis.reshape(bs*nTP, actual_latent_dim) 

        z_bis_permuted = self._permute_dims(z_bis).detach()

        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
            recon_x, x, mu, log_var, z.reshape(bs*nTP, actual_latent_dim), z_bis_permuted
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
    
    def predict(self, inputs: torch.Tensor) -> ModelOutput:
        bs = inputs.shape[0]
        nTP = inputs.shape[1] // self.n_channels

        # Process the first batch make it LSTM-friendly
        inputs = self.im_reshape(inputs, nTP)

        z = self.encoder(inputs).embedding

        # encoded my to LSTM-ed mu 
        z, (z_hidden, z_cell) = self.lstm_encode(z, self.lstm_emb, nTP)

        recon_lstm = self.lstm_decode(z_hidden[-1] if self.decode_last_hidden else z, z_hidden, z_cell, self.lstm_dec, bs, nTP)
        recon_x = self.im_unreshape(self.decoder(recon_lstm)["reconstruction"], nTP)

        return ModelOutput(
                    recon_x=recon_x, 
                    embedding=z_hidden[-1], 
                    embedding_recurrent=z
                )