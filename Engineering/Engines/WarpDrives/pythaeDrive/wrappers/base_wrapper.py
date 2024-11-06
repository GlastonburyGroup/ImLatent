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

class LSTMBaseWrapper:
    def __init__(self, model_config, requires_log):
        self.n_channels = model_config.n_channels
        self.num_layers = model_config.lstm_num_layers
        self.dropout = model_config.lstm_dropout
        self.bidirectional = bool(model_config.lstm_bidirectional)
        self.decode_last_hidden = bool(model_config.lstm_decode_last_hidden)
        self.connect_encdec = bool(model_config.lstm_connect_encdec)
        self.latent_dim = self.latent_dim // 2 if self.bidirectional else self.latent_dim
        self.im_encode_size = self.latent_dim * model_config.lstm_im_encode_factor
        
        self.lstm_emb = nn.LSTM(input_size=self.im_encode_size, hidden_size=self.latent_dim, num_layers=self.num_layers, 
                                batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)    
        if requires_log:    
            self.lstm_log = nn.LSTM(input_size=self.im_encode_size, hidden_size=self.latent_dim, num_layers=self.num_layers, 
                                    batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)  
        
        if self.connect_encdec:
            self.lstm_dec = nn.LSTM(input_size=self.latent_dim*2 if self.bidirectional and not self.decode_last_hidden else self.latent_dim, 
                                    hidden_size=self.latent_dim, num_layers=self.num_layers, 
                                    batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
            if self.bidirectional:
                self.lstm_dec_fc = nn.Linear(self.latent_dim*2, self.im_encode_size*2)
            else:
                self.lstm_dec_fc = nn.Linear(self.latent_dim, self.im_encode_size)
        else:
            self.lstm_dec = nn.LSTM(input_size=self.latent_dim*2 if self.bidirectional and not self.decode_last_hidden else self.latent_dim, 
                                    hidden_size=self.im_encode_size, num_layers=self.num_layers, 
                                    batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        
    def im_reshape(self, data, nTP):
        new_shape = (data.shape[0] * nTP, self.n_channels) + data.shape[2:]
        return data.reshape(new_shape)
    
    def im_unreshape(self, data, nTP):
        new_shape = (data.shape[0] // nTP, nTP*self.n_channels) + data.shape[2:]
        return data.reshape(new_shape)
        
    def lstm_encode(self, data, lstm_layer, nTP):
        data = data.reshape(data.shape[0]//nTP, nTP, self.im_encode_size)
        out, (hidden, cell) = lstm_layer(data)
        return out, (hidden, cell)
    
    def lstm_decode(self, data, hidden, cell, lstm_layer, batch_size, nTP):
        if self.decode_last_hidden:
            data = data.repeat(1, nTP, 1).reshape(batch_size, nTP, self.latent_dim)
        if self.connect_encdec:
            decoded_output, _ = lstm_layer(data, (hidden, cell))
            decoded_output = self.lstm_dec_fc(decoded_output)
        else:
            decoded_output, _ = lstm_layer(data)
        original_shape = (decoded_output.shape[0] * nTP, ) + decoded_output.shape[2:]
        return decoded_output.reshape(original_shape)   