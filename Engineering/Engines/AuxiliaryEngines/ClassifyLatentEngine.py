import os
import copy
import sys
from argparse import ArgumentParser
from os.path import join as pjoin
from statistics import median
from typing import Any, List
from collections import defaultdict
import logging

from sklearn.metrics import r2_score
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
                                   log_images, process_slicedict, process_testbatch)
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

class ClassifyLatentEngine(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = self.hparams.auto_optim
        
        self.phenotypes = self.hparams.DS_phenotypes['cols'].split(",") if isinstance(self.hparams.DS_phenotypes['cols'], str) else self.hparams.DS_phenotypes['cols']
        self.n_phenotypes = len(self.phenotypes)

        match self.hparams.modelID:
            case -1:
                self.net = nn.Identity() #Added for debugging
            case 0:
                self.net = nn.Linear(self.hparams.in_channels, len(self.phenotypes))
            case _:
                logging.critical(f"Classify Latent Engine:  Invalid modelID: {self.hparams.modelID}")
                sys.exit(f"Classify Latent Engine:  Error: Invalid modelID: {self.hparams.modelID}")
                
        self.ema_net = copy.deepcopy(self.net)
        self.ema_net.requires_grad_(False)
        self.ema_net.eval()
        
        #if there is a custom_step inside the selected Warp Drive, then use that instead of the default self.shared_step provided here
        if "custom_step" in dir(self.net): 
            self.shared_step = self.custom_shared_step

        if "custom_optimisers" in dir(self.net):
            self.configure_optimizers = self.net.custom_optimisers

        if  bool(self.hparams.preweights_path):
            logging.debug("Classify Latent Engine: Pre-weights found, loding...")
            chk = torch.load(self.hparams.preweights_path, map_location='cpu')
            self.net.load_state_dict(chk['state_dict'])
        
        match self.hparams.lossID:
            case 0:
                self.loss_func = nn.L1Loss()
            case 1:
                self.loss_func = nn.MSELoss()
            case 2:
                self.loss_func = nn.SmoothL1Loss()
            case _:
                logging.critical(f"Classify Latent Engine: Invalid lossID: {self.hparams.lossID}") 
                sys.exit(f"Classify Latent Engine: Error: Invalid lossID: {self.hparams.lossID}") 
                
        self.example_input_array = torch.empty(self.hparams.batch_size, self.hparams.in_channels).float()     
        
        self.val_res = []
        self.val_res_ema = []
        self.val_gt = [] 
        
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
                logging.debug(f"Classify Latent Engine: n_optimisers was set to 0 or -1, which has been updated to {self.hparams.n_optimisers} (number of optimisers required by the model, as specified with the model_optims and loss_optims attributes inside the selected Warp Drive)")
            else:
                self.hparams.n_optimisers = 1
                logging.debug(f"Classify Latent Engine: n_optimisers was set to 0 or -1, which has been updated to 1 (as model_optims and loss_optims attributes are not present inside the selected Warp Drive or they are not of equal lengths)")
            
        if self.hparams.n_optimisers == 1: #one optimiser for the whole model
            if "model_optims" in dir(self.net) and "loss_optims" in dir(self.net) and (len(self.net.model_optims) > 1 or len(self.net.loss_optims) > 1):
                logging.warning("Classify Latent Engine: It seems the model requires multiple optimisers (more than one element in model_optims and/or loss_optims attributes inside the selected Warp Drive), but n_optimisers is set to 1. Using 1 optimiser for the whole model. This could be by mistake! So, be careful!")
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
                logging.critical("Classify Latent Engine: Multiple optimisers requested (n_optimisers > 1), but model does not have a model_optims and loss_optims attributes or n_optimisers is not euqal to the length of the model_optims or loss_optims - to guide which parameters and loss are optimised by which optimiser.")
                sys.exit("Classify Latent Engine: Error: Multiple optimisers requested (n_optimisers > 1), but model does not have a model_optims and loss_optims attributes or n_optimisers is not euqal to the length of the model_optims or loss_optims - to guide which parameters and loss are optimised by which optimiser.")

    #The main forward pass
    def forward(self, x):
        return self.net(x)

    #Training 
    # def training_step(self, batch, batch_idx, **kwargs):
    def training_step(self, *args):
        batch, _ = args[0], args[1]
        out = self.net(batch['latent'])
        out_ema = self.ema_net(batch['latent'])
        loss = self.loss_func(out, batch['phenotypes'])
        loss_ema = self.loss_func(out_ema, batch['phenotypes'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        self.log('train_loss_ema', loss_ema, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        return loss  

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        ema(self.net, self.ema_net, self.hparams.ema_decay)

    #Validation  
    def validation_step(self, *args):
        batch, _ = args[0], args[1]
        out = self.net(batch['latent'])
        out_ema = self.ema_net(batch['latent'])
        loss = self.loss_func(out, batch['phenotypes'])
        loss_ema = self.loss_func(out_ema, batch['phenotypes'])
        self.log('val_loss', loss, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        self.log('val_loss_ema', loss_ema, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        
        self.val_res.append(out.detach().cpu().numpy())
        self.val_res_ema.append(out_ema.detach().cpu().numpy())
        self.val_gt.append(batch['phenotypes'].detach().cpu().numpy())
        
    def on_validation_epoch_end(self):
        self.val_res = np.concatenate(self.val_res, axis=0)
        self.val_res_ema = np.concatenate(self.val_res_ema, axis=0)
        self.val_gt = np.concatenate(self.val_gt, axis=0)
        
        r2 = score_each(r2_score, self.val_gt, self.val_res, "val_r2")
        r2_ema = score_each(r2_score, self.val_gt, self.val_res_ema, "val_r2_ema")
        r2s = {**r2, **r2_ema}
        self.log_dict(r2s, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=self.val_res.shape[0])
        
        self.val_res = []
        self.val_res_ema = []
        self.val_gt = []

    #Testing
    def on_test_start(self):
        # self.add_res_holder = AdditionalResHandler(ds_flags={"multi_channel": self.hparams.DS_multi_channel, "split_volume": self.hparams.DS_split_volume, "split_time": self.hparams.DS_split_time})
        self.metrics_list = []
        
        self.test_res = []
        self.test_res_ema = []
        self.test_gt = []

    def test_step(self, batch, batch_idx):
        out = self.net(batch['latent'])
        out_ema = self.ema_net(batch['latent'])
        loss = self.loss_func(out, batch['phenotypes'])
        loss_ema = self.loss_func(out_ema, batch['phenotypes'])
        self.log('test_loss', loss, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        self.log('test_loss_ema', loss_ema, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['latent'].shape[0])
        
        self.test_res.append(out.detach().cpu().numpy())
        self.test_res_ema.append(out_ema.detach().cpu().numpy())
        self.test_gt.append(batch['phenotypes'].detach().cpu().numpy())
        
        # pheno_pred = out.detach().cpu()
        # pheno_pred = pheno_pred.numpy() if pheno_pred.dtype not in {torch.bfloat16, torch.float16} else pheno_pred.to(dtype=torch.float32).numpy()
        # self.add_res_holder.store_res(batch, pheno_pred, "pheno_pred") 
        
        # pheno_pred_ema = out_ema.detach().cpu()
        # pheno_pred_ema = pheno_pred_ema.numpy() if pheno_pred_ema.dtype not in {torch.bfloat16, torch.float16} else pheno_pred_ema.to(dtype=torch.float32).numpy()
        # self.add_res_holder.store_res(batch, pheno_pred_ema, "pheno_pred_ema") 
        
    def on_test_end(self) -> None:
        self.test_res = np.concatenate(self.test_res, axis=0)
        self.test_res_ema = np.concatenate(self.test_res_ema, axis=0)
        self.test_gt = np.concatenate(self.test_gt, axis=0)
        
        r2 = score_each(r2_score, self.test_gt, self.test_res, "test_r2") 
        r2_ema = score_each(r2_score, self.test_gt, self.test_res_ema, "test_r2_ema") 
        r2s = {**r2, **r2_ema}
        #self.log_dict(r2s, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=self.test_res.shape[0])
        print("Final testing results: ")
        print(r2s)

    #Prediction
    def predict_step(self, batch, batch_idx):
        out = self.net(batch['latent'])
        out = out.detach().cpu()
        return out.numpy() if out.dtype not in {torch.bfloat16, torch.float16} else out.to(dtype=torch.float32).numpy()

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))
        
def score_each(metric, y_true, y_pred, tag):
    return {
        f"{tag}_{i}": metric(y_true[:, i], y_pred[:, i])
        for i in range(y_true.shape[1])
    }