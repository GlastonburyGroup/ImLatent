import os
import sys
from argparse import ArgumentParser
from os.path import join as pjoin
from statistics import median
from typing import Any, List
from collections import defaultdict
import logging

import pandas as pd
import itertools
import contextlib
import numpy as np
import torch
from Engineering.transforms import augmentations as pytAugmentations
from Engineering.transforms import motion as pytMotion
from Engineering.transforms import transforms as pytTransforms
from Engineering.transforms.tio import augmentations as tioAugmentations
from Engineering.transforms.tio import motion as tioMotion
from Engineering.transforms.tio import transforms as tioTransforms
from Engineering.utilities import (CustomInitialiseWeights, DataHandler,
                                   DataSpaceHandler, ResSaver, Evaluator, fetch_vol_subds, fetch_vol_subds_fastMRI, getSSIM,
                                   log_images, process_slicedict, process_testbatch, pearson_correlation)
from Engineering.Science.losses.pLoss.perceptual_loss import PerceptualLoss
from lightning.pytorch import LightningModule
from pytorch_msssim import MS_SSIM, SSIM
from torch import nn
from torch.utils.data.dataloader import DataLoader

#for packages that are not installed by default (especially on Windows)
with contextlib.suppress(Exception):
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from ..WarpDrives.pythaeDrive.pythaeStation import pythaeStation
from ...utilities import PlasmaConduit, ResHandler, AdditionalResHandler, Res2H5, get_nested_attribute, MetricsSave, filter_dict_keys

class ReconEngine(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = self.hparams.auto_optim

        model_configs = self.hparams.config_params['model']
        match self.hparams.modelID:
            case -1:
                self.net = nn.Identity() #Added for debugging
            case 0:
                pythae_configs = model_configs.pop('pythae')
                model_configs.update(pythae_configs)
                if self.hparams.lossID == -1:
                    model_configs['custom_loss_class'] = "Engineering.Science.losses.pLoss.perceptual_loss.PerceptualLoss"
                    model_configs['custom_loss_params'] = {
                        "loss_model": self.hparams.ploss_model, 
                        "n_level": self.hparams.ploss_level, 
                        "loss_type": self.hparams.ploss_type, 
                        "n_channels": self.hparams.out_channels, 
                        "n_dim":3 if self.hparams.is3D else 2
                    }
                    print("In-house (traditional) perceptual loss will be used")
                elif self.hparams.lossID == -2:
                    model_configs['custom_loss_class'] = "lpips.LPIPS"
                    model_configs['custom_loss_params'] = {
                        "net": self.hparams.ploss_model, 
                        "verbose": False
                    }
                    print("LPIPS (Perceptual Image Patch Similarity) will be used")
                model_configs['additional_nitems'] = {key:val for (key,val) in self.hparams.items() if key in [k for k in self.hparams.keys() if k.startswith("n_")]}
                self.net = pythaeStation(model_name=self.hparams.pythae_model, input_shape=self.hparams.input_shape, n_channels=self.hparams.in_channels, dim=3 if self.hparams.is3D else 2, base_dataset="med", config_path=self.hparams.pythae_config, pythae_wrapper_mode=self.hparams.pythae_wrapper_mode, **model_configs)
            case -13:
                logging.warning("Recon Engine: -13 has been passed as modelID, that means no net will be initalised inside ReconEngine and will be initialised inside its child class. Make sure that it is done properly!")
            case _:
                logging.critical(f"Recon Engine: Invalid modelID: {self.hparams.modelID}")
                sys.exit(f"Recon Engine: Error: Invalid modelID: {self.hparams.modelID}")

        if self.hparams.modelID != -13:
            #if there is a custom_step inside the selected Warp Drive, then use that instead of the default self.shared_step provided here
            if "custom_step" in dir(self.net): 
                self.shared_step = self.custom_shared_step

            if "custom_optimisers" in dir(self.net):
                self.configure_optimizers = self.net.custom_optimisers

            if  bool(self.hparams.preweights_path):
                logging.debug("Recon Engine: Pre-weights found, loding...")
                chk = torch.load(self.hparams.preweights_path, map_location='cpu')
                self.net.load_state_dict(chk['state_dict'])
            
            match self.hparams.lossID:
                case -2:
                    from lpips import LPIPS
                    self.loss_func = LPIPS(net=self.hparams.ploss_model, verbose=False)
                case -1:
                    self.loss_func = PerceptualLoss(loss_model=self.hparams.ploss_model, n_level=self.hparams.ploss_level, loss_type=self.hparams.ploss_type, 
                                                    n_channels=self.hparams.out_channels, n_dim=3 if self.hparams.is3D else 2)
                case 0:
                    self.loss_func = nn.L1Loss()
                case 1:
                    self.loss_func = nn.MSELoss()
                case 2:
                    self.loss_func = nn.SmoothL1Loss()
                case 3:
                    self.loss_func = MS_SSIM(channel=self.hparams.out_channels, data_range=1,
                                        spatial_dims=3 if self.hparams.is3D else 2, nonnegative_ssim=False)
                case 4:
                    self.loss_func = SSIM(channel=self.hparams.out_channels, data_range=1,
                                    spatial_dims=3 if self.hparams.is3D else 2, nonnegative_ssim=False)
                case _:
                    logging.critical(f"Recon Engine: Invalid lossID: {lossID}") #TODO: add perceptual loss
                    sys.exit(f"Recon Engine: Error: Invalid lossID: {lossID}") #TODO: add perceptual loss
                
        self.example_input_array = torch.empty(self.hparams.batch_size, self.hparams.in_channels, *self.hparams.input_shape).float()      
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nothing_yet', type=str)
        return parser

    #Configuration of the LightningModule
    def configure_optimizers(self):
        try:
            self.hparams.optimiserID = int(self.hparams.optimiserID)
            match self.hparams.optimiserID: 
                case 0:
                    optimiser_func = torch.optim.Adam
                case 1:
                    optimiser_func = torch.optim.AdamW
                case 2:
                    optimiser_func = torch.optim.RAdam
                case 3:
                    from Engineering.Science.optims.madam import Madam
                    optimiser_func = Madam
                case 4:
                    from Engineering.Science.optims.mtadam import MTAdam
                    optimiser_func = MTAdam
                case 100:
                    optimiser_func = DeepSpeedCPUAdam
                case 200:
                    optimiser_func = FusedAdam
        except:
            print(f"OptimiserID is not an integer, but {self.hparams.optimiserID}, so fetching that from torch.hub's pytorch-optimizer")
            optimiser_func = torch.hub.load('kozistr/pytorch_optimizer', self.hparams.optimiserID)

        if self.hparams.n_optimisers in [0, -1]: #If n_optimisers is 0 or -1, then it needs to be determined from the model using the model_optims and loss_optims attributes
            if "model_optims" in dir(self.net) and "loss_optims" in dir(self.net) and len(self.net.model_optims) == len(self.net.loss_optims) > 1:
                self.hparams.n_optimisers = len(self.net.model_optims)
                logging.debug(f"Recon Engine: n_optimisers was set to 0 or -1, which has been updated to {self.hparams.n_optimisers} (number of optimisers required by the model, as specified with the model_optims and loss_optims attributes inside the selected Warp Drive)")
            else:
                self.hparams.n_optimisers = 1
                logging.debug(f"Recon Engine: n_optimisers was set to 0 or -1, which has been updated to 1 (as model_optims and loss_optims attributes are not present inside the selected Warp Drive or they are not of equal lengths)")
            
        if self.hparams.n_optimisers == 1: #one optimiser for the whole model
            if "model_optims" in dir(self.net) and "loss_optims" in dir(self.net) and (len(self.net.model_optims) > 1 or len(self.net.loss_optims) > 1):
                logging.warning("Recon Engine: It seems the model requires multiple optimisers (more than one element in model_optims and/or loss_optims attributes inside the selected Warp Drive), but n_optimisers is set to 1. Using 1 optimiser for the whole model. This could be by mistake! So, be careful!")
            optimiser = optimiser_func(self.parameters(), lr=self.lr)
            optim_dict = {
                'optimizer': optimiser,
                'monitor': 'val_loss',
            }
            if self.hparams.lr_decay_type:  # If this is not zero
                optim_dict["lr_scheduler"] = {
                    "scheduler": self.hparams.lrScheduler_func(optimiser, **self.hparams.lrScheduler_param_dict),
                    'monitor': 'val_loss',
                }
            return optim_dict
        
        else: #Multiple optimisers for different parts of the model, determined by the model_optims and loss_optims attributes inside the selected Warp Drive        
            # TODO: multi-LR and multi-scheduler param support. Currently, the same LR from the hparams is used for all optimisers. These can be fetched from the model itself, maybe?
            if "model_optims" in dir(self.net) and "loss_optims" in dir(self.net) and self.hparams.n_optimisers == len(self.net.model_optims) and self.hparams.n_optimisers == len(self.net.loss_optims):
                optims = []
                for i in range(self.hparams.n_optimisers):
                    if type(self.net.model_optims[i]) == list:
                        coupled_params = list(itertools.chain.from_iterable(
                            get_nested_attribute(self.net, attribute_name).parameters()
                            for attribute_name in self.net.model_optims[i]
                        ))
                        optimiser = optimiser_func(coupled_params, lr=self.lr)
                    else:
                        optimiser = optimiser_func(get_nested_attribute(self.net, self.net.model_optims[i]).parameters(), lr=self.lr)
                    optim_dict = {
                        'optimizer': optimiser,
                        'monitor': self.net.loss_optims[i], 
                    }
                    if self.hparams.lr_decay_type:  # If this is not zero
                        optim_dict["lr_scheduler"] = {
                            "scheduler": self.hparams.lrScheduler_func(optimiser, **self.hparams.lrScheduler_param_dict),
                            'monitor': self.net.loss_optims[i], 
                        }
                    optims.append(optim_dict)
                return tuple(optims)
            else:
                logging.critical("Recon Engine: Multiple optimisers requested (n_optimisers > 1), but model does not have a model_optims and loss_optims attributes or n_optimisers is not euqal to the length of the model_optims or loss_optims - to guide which parameters and loss are optimised by which optimiser.")
                sys.exit("Recon Engine: Error: Multiple optimisers requested (n_optimisers > 1), but model does not have a model_optims and loss_optims attributes or n_optimisers is not euqal to the length of the model_optims or loss_optims - to guide which parameters and loss are optimised by which optimiser.")

    #The main forward pass
    def forward(self, x):
        if "valgrads" not in dir(self.net) or not self.net.valgrads:
            return self.net(x)
        with torch.enable_grad():
            return self.net(x)

    #Training 
    # def training_step(self, batch, batch_idx, **kwargs):
    def training_step(self, *args):
        batch, batch_idx = args[0], args[1]
        if self.hparams.auto_optim:
            model_output = self.shared_step(batch=batch)
        else:
            forwardArgs = {
                "batch": batch,
                "optimisers": self.optimizers(), 
                "manual_backward": self.manual_backward, 
                "step_optim": ((batch_idx + 1) % self.hparams.accumulate_gradbatch == 0)
            }
            if bool(self.hparams.grad_clip_algo):
                forwardArgs['grad_clipping'] = {
                    "algo": self.hparams.grad_clip_algo,
                    "val": self.hparams.grad_clip_val,
                    "clipper": self.clip_gradients
                }
            model_output = self.shared_step(**forwardArgs)
        loss_dict = filter_dict_keys(model_output, keycontains="loss", keyprefix="train_")
        self.log_dict(loss_dict, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['inp']['data'].shape[0])
        self.img_logger("train", batch_idx, batch['inp']['data'].cpu(), model_output.recon_x.detach().cpu())
        if self.hparams.auto_optim:
            return model_output.loss    

    #Validation  
    def validation_step(self, batch, batch_idx):
        if "valgrads" not in dir(self.net) or not self.net.valgrads:
            model_output = self.shared_step(batch)
        else:
            with torch.enable_grad():
                model_output = self.shared_step(batch)
        inp = batch['inp']['data'].cpu().numpy()
        prediction = model_output.recon_x.detach().cpu() 
        if "recon_x_indices" in model_output:
            inp = inp[model_output.recon_x_indices.detach().cpu()]
        self.img_logger("val", batch_idx, batch['inp']['data'].cpu(), prediction)
        
        prediction = prediction.numpy() if prediction.dtype not in {torch.bfloat16, torch.float16} else prediction.to(dtype=torch.float32).numpy()
        loss_dict = filter_dict_keys(model_output, keycontains="loss", keyprefix="val_")
        if not np.iscomplexobj(inp):
            loss_dict["val_ssim"] = getSSIM(inp, prediction, data_range=1) 
        else:
            loss_dict["val_ssim_mag"] = getSSIM(abs(inp), abs(prediction), data_range=1) 
            loss_dict["val_pearson_mag"] = pearson_correlation(abs(inp), abs(prediction))
            loss_dict["val_pearson_phase"] = pearson_correlation(np.angle(inp), np.angle(prediction))
        self.log_dict(loss_dict, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['inp']['data'].shape[0])

    #Testing
    def on_test_start(self):
        self.res_holder = ResHandler(save_recon=self.hparams.config_params['training']['save_recon'], pIDs_save_recon=self.trainer.datamodule.test_dataloader().dataset.pIDs_save_recon, ds_flags={"multi_channel": self.hparams.DS_multi_channel, "split_volume": self.hparams.DS_split_volume, "split_time": self.hparams.DS_split_time}) #for storing results as: H5 key -> i (Channels) -> j (Time) -> k (Slice)        
        self.add_res_holder = AdditionalResHandler(ds_flags={"multi_channel": self.hparams.DS_multi_channel, "split_volume": self.hparams.DS_split_volume, "split_time": self.hparams.DS_split_time})
        self.evaluator = Evaluator(is3D=self.hparams.is3D, **self.hparams.config_params['eval'])
        self.metrics_list = []

    def test_step(self, batch, batch_idx):
        if "valgrads" not in dir(self.net) or not self.net.valgrads:
            predict_output = self.net.predict_step(batch) #have two properties: recon_x and embedding
        else:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    predict_output = self.net.predict_step(batch)

        emb = predict_output.embedding.detach().cpu()
        emb = emb.numpy() if emb.dtype not in {torch.bfloat16, torch.float16} else emb.to(dtype=torch.float32).numpy() #TODO: think about raw_embedding as well (e.g. RHVAE) and other possible outputs with "embedding" in their name

        recon = predict_output.recon_x.detach().cpu()
        recon = recon.numpy() if recon.dtype not in {torch.bfloat16, torch.float16} else recon.to(dtype=torch.float32).numpy()

        if "pheno_pred" in predict_output:
            pheno_pred = predict_output.pheno_pred.detach().cpu()
            pheno_pred = pheno_pred.numpy() if pheno_pred.dtype not in {torch.bfloat16, torch.float16} else pheno_pred.to(dtype=torch.float32).numpy()
            self.add_res_holder.store_res(batch, pheno_pred, "pheno_pred") 

        if "conf_pred" in predict_output:
            conf_pred = predict_output.conf_pred.detach().cpu()
            conf_pred = conf_pred.numpy() if conf_pred.dtype not in {torch.bfloat16, torch.float16} else conf_pred.to(dtype=torch.float32).numpy()
            self.add_res_holder.store_res(batch, conf_pred, "conf_pred") 

        gt = batch['inp']['data'].cpu().numpy()

        scores = self.evaluator.get_batch_scores(gt=gt, out=recon, keys=batch['key'], tag="") 
        self.metrics_list.append(scores)      
        self.res_holder.store_res(batch, emb, recon)   

        if not np.iscomplexobj(gt):
            ssim = getSSIM(gt, recon, data_range=1) 
            self.log('test_ssim'+(f"_{self.hparams.output_suffix}" if bool(self.hparams.output_suffix) else ""), ssim, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=gt.shape[0])
        else:
            ssim = getSSIM(abs(gt), abs(recon), data_range=1) 
            self.log('test_ssim_mag'+(f"_{self.hparams.output_suffix}" if bool(self.hparams.output_suffix) else ""), ssim, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=gt.shape[0])
            pearson_mag = pearson_correlation(abs(gt), abs(recon))
            self.log('test_pearson_mag'+(f"_{self.hparams.output_suffix}" if bool(self.hparams.output_suffix) else ""), pearson_mag, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=gt.shape[0])
            pearson_phase = pearson_correlation(np.angle(gt), np.angle(recon))
            self.log('test_pearson_phase'+(f"_{self.hparams.output_suffix}" if bool(self.hparams.output_suffix) else ""), pearson_phase, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=gt.shape[0])
        
    def on_test_end(self) -> None:
        results = self.res_holder.to_array()
        additional = {}
        if self.add_res_holder:
            for tag in self.add_res_holder.keys():
                additional[tag] = self.add_res_holder.to_array(tag)
        Res2H5(results, self.hparams.res_path, save_recon=self.hparams.config_params['training']['save_recon'], additional=additional)

        if len(self.metrics_list) > 1:
            MetricsSave(self.metrics_list, self.hparams.res_path)


    #Prediction
    def predict_step(self, batch, batch_idx):
        predict_output = self.net.predict_step(batch) #have two properties: recon_x and embedding
        emb = predict_output.embedding.detach().cpu()
        return emb.numpy() if emb.dtype not in {torch.bfloat16, torch.float16} else emb.to(dtype=torch.float32).numpy()

    ###############
    # Custom functions, not part of the LightningAPI
    def shared_step(self, batch, optimisers=None, manual_backward=None, step_optim=True, grad_clipping=None): #this function won't mostly be used in favour of the custom_step available inside the selected Warp Drive (i.e. custom_shared_step will be used and will call custom_step)
        assert optimisers is None, "ReconEngine: shared_step inside the ReconEngine class should only be used when auto_optim is True."
        model_output = self(batch['inp']['data']) 
        if isinstance(model_output, torch.Tensor):
            _temp = PlasmaConduit(recon_x = model_output)
            model_output = _temp
        model_output.loss = self.loss_func(model_output.recon_x, batch['inp']['data']).to(model_output.recon_x.dtype)
        if self.hparams.IsNegLoss:
            model_output.loss = -model_output.loss
        return model_output

    def custom_shared_step(self, batch, optimisers=None, manual_backward=None, step_optim=True, grad_clipping=None): #This function will be called from all types of steps (i.e. train, val, test, predict). Must return object of the PlasmaConduit type
        return self.net.custom_step(batch, loss_func=self.loss_func, IsNegLoss=self.hparams.IsNegLoss, optimisers=optimisers, manual_backward=manual_backward, step_optim=step_optim, grad_clipping=grad_clipping)
    
    def img_logger(self, tag, batch_idx, inp, pred) -> None:        
        if (
            not self.hparams.tbactive
            or self.hparams.config_params['training']['im_log_freq'] <= -1
            or batch_idx % self.hparams.config_params['training']['im_log_freq']
            != 0
        ):
            return
        logger = self.loggers[-1] if self.hparams.wnbactive and self.hparams.tbactive else self.logger
        if len(inp.shape) == 5:  # 3D
            central_slice = inp.shape[-3] // 2
            inp = inp[:, :, central_slice]
            pred = pred[:, :, central_slice]

        inp = torch.abs(inp) if torch.is_complex(inp) else inp
        inp if inp.dtype not in {torch.bfloat16, torch.float16} else inp.to(dtype=torch.float32).numpy()

        pred = torch.abs(pred) if torch.is_complex(pred) else pred
        pred if pred.dtype not in {torch.bfloat16, torch.float16} else pred.to(dtype=torch.float32).numpy()

        log_images(logger.experiment, inp, pred, None, batch_idx, tag)
    ###############