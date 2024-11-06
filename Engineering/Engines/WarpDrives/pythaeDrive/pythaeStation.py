import os
import sys
from copy import deepcopy
import ast
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.trainers import (
    BaseTrainerConfig,
    CoupledOptimizerTrainerConfig,
    AdversarialTrainerConfig,
    CoupledOptimizerAdversarialTrainerConfig
)

from . import *
from ....utilities import import_func_str, PlasmaConduit, string2dict, CustomInitialiseWeights
from .utils import *
from .wrappers import *

class pythaeStation(nn.Module):
    def __init__(self, model_name, input_shape, latent_dim=-1, n_channels=3, dim=2, n_features=64, nn="ResNet", config_path=None, base_dataset="med", pythae_wrapper_mode=0, **kwargs):
        super(pythaeStation, self).__init__()

        if "custom" in model_name:
            is_custom_model = True
            model_name = model_name.replace("custom_", "")
        else:
            is_custom_model = False

        assert model_name != "vqvae", "VQ-VAE is not ready yet. It's giving error as for some reason, the input channel is in the last dim, instead of 2nd."

        if "rhvae" in model_name:
            logging.warning("RHVAE requires gradients during eval as well. Adding valgrads=True.")
            self.valgrads = True

        ### Fetch the model config file path
        if bool(config_path):
            self.model_config_path = f"Engineering/Engines/WarpDrives/pythaeDrive/configs/{config_path}"
        elif not is_custom_model:
            if base_dataset == "med":
                logging.warning("pythaeStation: The 'med' base dataset is our own placeholder creation and does not have any default config files. Using the one from celeba (if available) as default.")
                self.model_config_path = f"Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/celeba/{model_name}_config.json"
            else:
                self.model_config_path = f"Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/{base_dataset}/{model_name}_config.json"
                if not os.path.exists(self.model_config_path):
                    self.model_config_path = f"Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/{base_dataset}/{model_name}/{model_name}_config.json" #Just for the dsprites dataset models, for beta_tc_vae and factorvae
            # assert os.path.exists(self.model_config_path), f"Model config file not found: {self.model_config_path}. If you are not supplying explicitly, then the default config file is not present for the given base dataset and model name. Try with a correct path!"
            if not os.path.exists(self.model_config_path):
                logging.warning(f"Model config file not found: {self.model_config_path}. If you are not supplying explicitly, then the default config file is not present for the given base dataset and model name. Be aware!!! No config file will be used.")
                self.model_config_path = None #TODO: create config files for med base_dataset
        else:
            self.model_config_path = None

        ###### Dynamic Class Import Zone ######

        match nn:
            case "convnet":
                nn = "Conv"
            case "resnet":
                nn = "ResNet"
            case "cconvnet" | "cvconvnet":
                nn = "CVConv"
            case "cresnet" | "cvresnet":
                nn = "CVResNet"
            case _:
                logging.critical(f"Recon Engine: Invalid value for the nn backend (most likely inside config.yaml/model/pyhae/nn or using cmdargs): {nn}")
                sys.exit(f"Recon Engine: Error: Invalid value for the nn backend (most likely inside config.yaml/model/pyhae/nn or using cmdargs):  {nn}")
        self.nn = nn
        if "CV" in nn: #It's a CVCNN
            self.forward = self.CVforward
            self.predict_step = self.CV_predict_step

        base_dataset = "cifar" if base_dataset == "cifar10" else base_dataset
        packstr = f"pythae.models.nn.benchmarks.{base_dataset}"

        ### Fix for the "med" base_dataset type, as it has been locally implemented in our pipeline (for now). If it's included in the main pythae repo, then this can be removed.
        if base_dataset == "med":
            packstr = "Engineering.Engines.WarpDrives.pythaeDrive.nn"

        ### Import the required encoder class
        if model_name == "vqvae":
            if nn == "ResNet":
                cls_enc = import_func_str(packstr, f"Encoder_ResNet_VQVAE_{base_dataset.upper()}")
            else:
                logging.warning("VQ-VAE originally uses ResNet architectures as the encoder and decoder. Using the ConvNet architecture might not be totally comparable to the original work.")
                cls_enc = import_func_str(packstr, f"Encoder_Conv_AE_{base_dataset.upper()}")
        elif model_name == "svae":
            cls_enc = import_func_str(packstr, f"Encoder_{nn}_SVAE_{base_dataset.upper()}")
        elif model_name in ["ae", "wae", "rae_l2", "rae_gp" ]:
            cls_enc = import_func_str(packstr, f"Encoder_{nn}_AE_{base_dataset.upper()}")
        else:
            cls_enc = import_func_str(packstr, f"Encoder_{nn}_VAE_{base_dataset.upper()}")

        ### Import the required decoder class
        if model_name == "vqvae":
            if nn == "ResNet":
                cls_dec = import_func_str(packstr, f"Decoder_ResNet_VQVAE_{base_dataset.upper()}")
            else:
                logging.warning("VQ-VAE originally uses ResNet architectures as the encoder and decoder. Using the ConvNet architecture might not be totally comparable to the original work.")
                cls_dec = import_func_str(packstr, f"Decoder_Conv_AE_{base_dataset.upper()}")
        else:
            cls_dec = import_func_str(packstr, f"Decoder_{nn}_AE_{base_dataset.upper()}")       


        ### Import the required model class and its config class
        clsname_model, clsname_model_config = PYTHAEMODELCLASS[model_name].split(",")
        if is_custom_model:
            cls_model = import_func_str("Engineering.Engines.WarpDrives.pythaeDrive.custom_models", clsname_model.strip())
            cls_model_config = import_func_str("Engineering.Engines.WarpDrives.pythaeDrive.custom_models", clsname_model_config.strip())
        else:
            cls_model = import_func_str("pythae.models", clsname_model.strip())
            cls_model_config = import_func_str("pythae.models", clsname_model_config.strip())

        ###### Dynamic Class Import Zone Ends ######    

        ###### Model and Trainig Config Defination Zone ######   

        ### define the model and the required parameters
        if self.model_config_path is not None:
            self.model_config = cls_model_config.from_json_file(self.model_config_path)

        else:
            self.model_config = cls_model_config()

        self.model_config.input_dim = input_shape
        self.model_config.n_channels = n_channels
        self.model_config.dim = dim
        self.model_config.n_features = n_features

        self.model_config.is_cv = "CV" in nn

        if latent_dim is not None and latent_dim >0:
            self.model_config.latent_dim = latent_dim

        if "models" in kwargs and model_name in kwargs['models']:
            self.model_config.__dict__.update((key, value) for key, value in kwargs['models'][model_name].items() if key in self.model_config.__dict__)
        if kwargs: #If any extra parameters are passed, then update the model config (only if those params are present in the config, otherwise will be ignored!)
            self.model_config.__dict__.update((key, value) for key, value in kwargs.items() if key in self.model_config.__dict__)

        if "learn_phenotypes" in self.model_config.__dict__ and self.model_config.learn_phenotypes:
            self.model_config.n_phenotypes = kwargs['additional_nitems']['n_phenotypes'] if "additional_nitems" in kwargs and "n_phenotypes" in kwargs['additional_nitems'] else self.model_config.n_phenotypes
        if "anti_confounders_strategy" in self.model_config.__dict__ and self.model_config.anti_confounders_strategy:
            self.model_config.n_confounders = kwargs['additional_nitems']['n_confounders'] if "additional_nitems" in kwargs and "n_confounders" in kwargs['additional_nitems'] else self.model_config.n_confounders
            self.model_config.n_confounders_bincat = kwargs['additional_nitems']['n_confounders_bincat'] if "additional_nitems" in kwargs and "n_confounders_bincat" in kwargs['additional_nitems'] else self.model_config.n_confounders_bincat
            self.model_config.n_confounders_mulcat = kwargs['additional_nitems']['n_confounders_mulcat'] if "additional_nitems" in kwargs and "n_confounders_mulcat" in kwargs['additional_nitems'] else self.model_config.n_confounders_mulcat
            self.model_config.n_confounders_cont = kwargs['additional_nitems']['n_confounders_cont'] if "additional_nitems" in kwargs and "n_confounders_cont" in kwargs['additional_nitems'] else self.model_config.n_confounders_cont

        self.enc_config = deepcopy(self.model_config)
        self.dec_config = deepcopy(self.model_config)

        if pythae_wrapper_mode == 1: #LSTM wrapper
            print("LSTM wrapper mode activated!")
            print(str({(f"lstm_{key}", value) for key, value in kwargs['wrappers']['lstm'].items()}))
            cls_model = import_func_str("Engineering.Engines.WarpDrives.pythaeDrive.wrappers", f"LSTM{clsname_model.strip()}")
            self.model_config.__dict__.update((f"lstm_{key}", value) for key, value in kwargs['wrappers']['lstm'].items())
            self.enc_config.latent_dim = self.model_config.latent_dim * self.model_config.lstm_im_encode_factor
            self.dec_config.latent_dim = self.enc_config.latent_dim*2 if self.model_config.lstm_bidirectional else self.enc_config.latent_dim

        if "reconstruction_loss" in self.model_config.__dict__ and self.model_config.reconstruction_loss in ["custom", "custom_masked"]: 
            if "custom_loss_class" not in kwargs:
                raise ValueError("pythae: If the custom loss is selected in the model config, then it is required to pass the custom loss class as a keyword argument to the pythae station (Can be done using config.yaml file: model/pythae/custom_loss_class).")
            loss_class_name = kwargs["custom_loss_class"].split(".")[-1]
            cls_loss = import_func_str(kwargs["custom_loss_class"].replace(f".{loss_class_name}", ""), loss_class_name)
            if "custom_loss_params" in kwargs:
                if type(kwargs["custom_loss_params"]) is str:
                    kwargs["custom_loss_params"] = string2dict(kwargs["custom_loss_params"])
                loss_obj = cls_loss(**kwargs["custom_loss_params"])
            else:
                loss_obj = cls_loss()
            self.model = cls_model(
                                model_config=self.model_config,
                                encoder=cls_enc(self.enc_config),
                                decoder=cls_dec(self.dec_config),
                                custom_recon_loss_func=loss_obj
                            )
        else:
            self.model = cls_model(
                                    model_config=self.model_config,
                                    encoder=cls_enc(self.enc_config),
                                    decoder=cls_dec(self.dec_config),
                                )        
            
        # if self.model_config.is_cv:
        #     self.model.apply(CustomInitialiseWeights)

        ### fetch the default training config file path
        if base_dataset == "dsprites" and model_name == "beta_tc_vae":
            default_training_config_path = "Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/dsprites/beta_tc_vae/base_training_config.json"
        elif base_dataset == "dsprites" and model_name == "factor_vae":
                default_training_config_path = "Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/dsprites/factorvae/base_training_config.json"
        else:
            default_training_config_path = f"Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/{base_dataset}/base_training_config.json"

        ### TODO: it's a temp dirty fix. Create different config files for the "med" base_dataset
        if base_dataset == "med":
            default_training_config_path = default_training_config_path.replace("med", "celeba")
            logging.warning("Currently, the 'med' base_dataset does not have training cofig file. Using the 'celeba' base_dataset training config file instead.")

        ### define the training config
        if "default_training_config" in self.model.__dict__:
            self.default_training_config = self.model.default_training_config
            if "model_optims" in self.model.__dict__:
                self.model_optims = self.model.model_optims
            if "loss_optims" in self.model.__dict__:
                self.loss_optims = self.model.loss_optims
        elif self.model.model_name == "RAE_L2":
            self.default_training_config = CoupledOptimizerTrainerConfig.from_json_file(default_training_config_path) # Two optim: Encoder and Decoder
            self.trainer_mode = "coupled"
            self.model_optims = ["model.encoder", "model.decoder"]
            self.loss_optims = ["encoder_val_loss", "decoder_val_loss"]
        elif self.model.model_name in ["Adversarial_AE", "FactorVAE"]:
            self.default_training_config = AdversarialTrainerConfig.from_json_file(default_training_config_path) # Two optim: AE and Discriminator
            self.trainer_mode = "adversarial"
            self.model_optims = [["model.encoder", "model.decoder"], "model.discriminator"]
            self.loss_optims = ["ae_val_loss", "discriminator_val_loss"]
        elif self.model.model_name == "VAEGAN":
            self.default_training_config = CoupledOptimizerAdversarialTrainerConfig.from_json_file(default_training_config_path) # Three optim: Encoder, Decoder and Discriminator
            self.trainer_mode = "coupled_adversarial"
            self.model_optims = ["model.encoder", "model.decoder", "model.discriminator"]
            self.loss_optims = ["encoder_val_loss", "decoder_val_loss", "discriminator_val_loss"]
        else:
            self.default_training_config = BaseTrainerConfig.from_json_file(default_training_config_path) # One optim for the model
            self.trainer_mode = "base"
        # TODO: currently, self.default_training_config is not used. It is just for a dummy value. Need to fix this and use custom values for each model during training.

        if "custom_step" in dir(self.model):
            self.custom_step = self.model.custom_step

    def forward(self, x, return_only_recon=True):
        return PlasmaConduit(self.model(x)) if not return_only_recon else self.model(
                            ({"data": x} if self.model_config.reconstruction_loss != "custom_masked" else {"data": x, "mask": torch.zeros(x.shape).to(x.device)}) 
                            if isinstance(x, torch.Tensor) else x).recon_x  # Tensor as input shouldn't be when it's supplied from the pipeline, can only happen during sanity checks!
    
    def CVforward(self, x, return_only_recon=True):
        if return_only_recon:
            x = ({"data": x} if self.model_config.reconstruction_loss != "custom_masked" else {"data": x, "mask": torch.zeros(x.shape).to(x.device)}) if isinstance(x, torch.Tensor) else x
        if not torch.is_complex(x["data"]):
            x["data"] = x["data"] + 0j
            return_recon_abs = True
        else:
            return_recon_abs = False
        out = PlasmaConduit(self.model(x))
        if return_recon_abs:
            out.recon_x = torch.abs(out.recon_x)
        return out.recon_x if return_only_recon else out


    def custom_step(self, batch, **kwargs):
        if self.model_config.reconstruction_loss != "custom_masked":
            model_output = self(PlasmaConduit(data=batch['inp']['data']), return_only_recon=False)
        else:
            model_output = self(PlasmaConduit(data=batch['inp']['data'], mask=batch['inp']['mask']), return_only_recon=False)
            
        if "optimisers" in kwargs and kwargs["optimisers"] is not None:
            match self.trainer_mode: 
                case "coupled":
                    if model_output.update_encoder:
                        kwargs["manual_backward"](model_output.encoder_loss, retain_graph=True)

                    if model_output.update_decoder:
                        kwargs["manual_backward"](model_output.decoder_loss, retain_graph=True)

                    if model_output.update_encoder and kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][0], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][0].step()
                        kwargs["optimisers"][0].zero_grad()

                    if model_output.update_decoder and kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][1], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][1].step()
                        kwargs["optimisers"][1].zero_grad()

                case "adversarial":
                    kwargs["manual_backward"](model_output.autoencoder_loss, retain_graph=True)

                    kwargs["manual_backward"](model_output.discriminator_loss)

                    if kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][0], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][1], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][0].step()
                        kwargs["optimisers"][0].zero_grad()
                        kwargs["optimisers"][1].step()
                        kwargs["optimisers"][1].zero_grad()

                case "coupled_adversarial":
                    if model_output.update_encoder:
                        kwargs["manual_backward"](model_output.encoder_loss, retain_graph=True)

                    if model_output.update_decoder:
                        kwargs["manual_backward"](model_output.decoder_loss, retain_graph=True)

                    if model_output.update_discriminator:
                        kwargs["manual_backward"](model_output.discriminator_loss)

                    if model_output.update_encoder and kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][0], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][0].step()
                        kwargs["optimisers"][0].zero_grad()

                    if model_output.update_decoder and kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][1], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][1].step()
                        kwargs["optimisers"][1].zero_grad()

                    if model_output.update_discriminator and kwargs["step_optim"]:
                        if kwargs["grad_clipping"] is not None:
                            kwargs["grad_clipping"]["clipper"](kwargs["optimisers"][2], gradient_clip_val=kwargs["grad_clipping"]["val"], gradient_clip_algorithm=kwargs["grad_clipping"]["algo"])
                        kwargs["optimisers"][2].step()
                        kwargs["optimisers"][2].zero_grad()

        return model_output
    
    def predict_step(self, batch):
        if "valgrads" not in dir(self) or not self.valgrads:
            return self.model.predict(batch['inp']['data'])
        x = batch['inp']['data'].clone().requires_grad_()
        return self.model.predict(x)
    
    def CV_predict_step(self, batch):
        if not torch.is_complex(batch['inp']['data']):
            out = self.model.predict(batch['inp']['data'] + 0j)
            out.recon_x = torch.abs(out.recon_x)
        else:
            out = self.model.predict(batch['inp']['data'])
        return out
        
    ###### Model and Trainig Config Defination Zone Ends ######  
