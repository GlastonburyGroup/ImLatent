"""
The main entry point of the pipeline.
"""



import contextlib
import json
import os
import sys
from os.path import join as pjoin
import json
import yaml
from argparse import ArgumentParser
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
from typing import Optional, Union
import importlib.util
import tempfile

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch import optim
from torchvision import transforms as tvTransforms
from lightning.pytorch.tuner import Tuner
from transformers import AutoModel

from .AuxiliaryEngines.ReconEngine import ReconEngine

with contextlib.suppress(Exception):
    from .AuxiliaryEngines.DiffAEEngine import DiffAEEngine
    
with contextlib.suppress(Exception):
    from .AuxiliaryEngines.ClassifyLatentEngine import ClassifyLatentEngine
    from ..data.latent_datamodule import LatentDataModule
    
from ..data.datamodule import UKBBImgDataModule
from ..Science.losses import IS_NEG_LOSS
from ..transforms import transforms as pytTransforms, augmentations as pytAugmentations
from ..Science.stardate import ComputeSD
from ..utilities import hook_exception_log, process_unknown_args

class Engine(object):

    def __init__(self, parser: ArgumentParser, sys_params: Optional[Union[dict, None]]):
        _temp_args, _ = parser.parse_known_args()

        if "matmul_precision" in _temp_args.__dict__ and _temp_args.matmul_precision != "highest" and torch.cuda.is_bf16_supported():
            torch.set_float32_matmul_precision(_temp_args.matmul_precision)
        
        if _temp_args.taskID == 0:
            parser = ReconEngine.add_model_specific_args(parser)
        elif _temp_args.taskID == 1:
            parser = DiffAEEngine.add_model_specific_args(parser)
        elif _temp_args.taskID == 2:
            parser = ClassifyLatentEngine.add_model_specific_args(parser)
        else:
            sys.exit(
                "Only ReconEngine has been implemented yet, so only Undersampled Recon or MoCo is possible.")

        hparams, unknown_args = parser.parse_known_args()

        unknown_args = process_unknown_args(unknown_args)
        
        with open(hparams.configyml_path, 'r') as f:
            cfg = yaml.full_load(f)
        for k, v in unknown_args.items():
            if k.startswith("json§"): continue
            keys = k.split("§")
            reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], cfg)[keys[-1]] = v
        hparams.config_params = cfg

        with open(hparams.datajson_path, 'r') as json_file:
            json_data = json.load(json_file)
        for k, v in unknown_args.items():
            if not k.startswith("json§"): continue
            json_data.update({k.split("§")[-1]: v})
        hparams.__dict__.update(json_data)

        hparams.__dict__.update({'sys_params': sys_params})

        if "fulltrainID" in hparams and bool(hparams.fulltrainID):
            logging.debug("Main Engine: fulltrainID, won't be generated automatically using the supplied trainID!")
            hparams.trainID = hparams.fulltrainID
            hparams.run_name = hparams.fulltrainID
        else:
            hparams.trainID += f"fold{hparams.foldID}_prec{hparams.ampmode}" 
            if hparams.taskID == 0:
                hparams.trainID += (f"_model{hparams.modelID}" if hparams.modelID > 0 else f"_pythaemodel-{hparams.pythae_model}")
            elif hparams.taskID == 1:
                hparams.trainID += "_DiffAE"
            elif hparams.taskID == 2:
                hparams.trainID += "_LtCls"
                hparams.parent_train_path = pjoin(hparams.save_path, hparams.parent_trainID)
                hparams.save_path = pjoin(hparams.parent_train_path, "ClassifyLatent")

            if "interp_fact" in hparams and hparams.interp_fact != 1:
                hparams.trainID += f"_interp{hparams.interp_fact}"

            hparams.run_name = (
                f"{hparams.run_prefix}_{hparams.trainID}"
                if bool(hparams.run_prefix)
                else hparams.trainID
            )

        hparams.res_path = pjoin(
            hparams.save_path, hparams.run_name, f"Output"+(f"_{hparams.output_suffix}" if bool(hparams.output_suffix) else ""))
        os.makedirs(hparams.res_path, exist_ok=True)        
        
        logging.basicConfig(filename=pjoin(hparams.save_path, hparams.run_name, "CaptainsLog.log"), level=logging.DEBUG)
        init_msg = "\n--------------------->>>>>> NEW MISSION <<<<<---------------------"
        init_msg += f"\nStardate: {ComputeSD.nowstardate()}"
        init_msg += f"\nEarthdate: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        init_msg += f"\n---------------------<<<<< NEW MISSION >>>>>---------------------\n"
        logging.debug(init_msg)
        sys.excepthook = hook_exception_log # To log exceptions to the log file

        hparams.n_train_batches = int(hparams.n_train_batches) if hparams.n_train_batches > 1 else hparams.n_train_batches
        hparams.n_val_batches = int(hparams.n_val_batches) if hparams.n_val_batches > 1 else hparams.n_val_batches

        if not hparams.non_deter:
            seed_everything(hparams.config_params['seed'], workers=True)

        self.hparams = hparams

        self.__update_default_params()
        logging.debug("Main Engine: Default params updated!")
        self.__sanity_check()
        logging.debug("Main Engine: Sanity check passed!")
        self.__prep_datamodule()
        logging.debug("Main Engine: Data Module is ready!")
        self.__prep_trainer()

        logging.debug("Main Engine: Engine has been initialised!")

    def __update_default_params(self):
        if sys.platform.startswith('win') and self.hparams.num_workers > 0:
            logging.warning(f"Main Engine: Windows does not support multi-processing properly. num_workers has been now set to 0 (even though it was originally set to {self.hparams.num_workers}!")
            self.hparams.num_workers = 0
        if self.hparams.num_workers == 0:
            self.hparams.nbatch_prefetch = None
        elif self.hparams.num_workers > (self.hparams.sys_params['cpus_avail']-1):
            logging.warning(f"Main Engine: It is recommended to use at most {self.hparams.sys_params['cpus_avail']-1} workers as that's one less than the number of CPUs avaialble (set by SLURM or 90% of the total CPUs available on a non-SLURM system), but you have set num_workers to {self.hparams.num_workers}! Changing it to {self.hparams.sys_params['cpus_avail']-1}!")
            self.hparams.num_workers = self.hparams.sys_params['cpus_avail']-1
        elif self.hparams.num_workers < (self.hparams.sys_params['cpus_avail']-1):
            logging.warning(f"Main Engine: It is recommended to use {self.hparams.sys_params['cpus_avail']-1} workers as that's one less than the number of CPUs avaialble (set by SLURM or 90% of the total CPUs available on a non-SLURM system) to obtain the best possible performance in terms of data loading, but you have set num_workers to {self.hparams.num_workers}! Not updating the value, it's just a warning!")

    #TODO: fix the bugs inside the temporary sanity check function
    # A place for temporary sanity checks, to be removed after the bug is fixed
    def __temp_insanity(self):
        insane = True
        if self.hparams.lr_decay_type == 1: # To be resolved from Lightning
            logging.error("Main Engine: StepLR (lr_decay_type=1) is throwing a weird error from Lighning! Avoid it ultil resolved!")        
        elif self.hparams.complie_model: # To be resolved from Lightning or PyTorch, I'm not sure yet
            logging.error("Main Engine: Compile is throwing a weird error! Avoid it ultil resolved!")  
        elif self.hparams.lr_decay_type != 0:
            logging.error("Main Engine: Currently, lr_decay is not yet supported! Avoid it ultil resolved!")
        else:
            insane = False
        return insane

    def __sanity_check(self):
        insane = True
        if self.__temp_insanity():
            pass
        elif self.hparams.auto_lr and self.hparams.n_optimisers > 1: #TODO add checks for DDP for both autos
            logging.error("Main Engine: AutoLR is not supported for multi-optimiser setups!")
        elif self.hparams.auto_optim and self.hparams.n_optimisers > 1:
            logging.error("Main Engine: auto_optim is not supported for multi-optimiser setups!")
        elif "use_concurrent_dataloader" in self.hparams and self.hparams.use_concurrent_dataloader and (spec := importlib.util.find_spec("concurrent_dataloader")) is None:
            logging.error("Main Engine: concurrent_dataloader is not installed, but use_concurrent_dataloader param is set to True! Please install it using pip install git+https://github.com/soumickmj/concurrent-dataloader")
        else:
            insane = False
        if insane:
            logging.critical("Main Engine: Sanity check failed!")
            sys.exit("Main Engine: Sanity check failed!")

    def __prep_datamodule(self):
        # Prepare dataset params
        DSParams = {
            "mergeTrainVal": not self.hparams.run_mode in [1, 4],
            "fetchMinMax": "vol" in self.hparams.norm_type or "vol" in self.hparams.zscore_prenorm, #Params directly supplied as a command line argument (defined in the main file)
            **self.hparams.config_params['data']['dataset'], #Params defined in the config yaml file
        } | {
            k.replace("DS_", ""): v
            for k, v in self.hparams.__dict__.items() 
            if k.startswith("DS_") #Params defined in the dataNPath json file
        }

        # Prepare phenotypes and confounders
        if 'fetch_phenotypes' in DSParams and DSParams['fetch_phenotypes'] and 'phenotypes' in DSParams:
            DSParams['phenotypes']['cols'] = DSParams['phenotypes']['cols'].split(',')
            self.hparams.n_phenotypes = len(DSParams['phenotypes']['cols'])
        if 'fetch_confounders' in DSParams and DSParams['fetch_confounders'] and 'confounders' in DSParams:
            DSParams['confounders']['cols'] = DSParams['confounders']['cols'].split(',')
            self.hparams.n_confounders = len(DSParams['confounders']['cols'])
            DSParams['confounders']['bincat_cols'] = DSParams['confounders']['bincat_cols'].split(',')
            self.hparams.n_confounders_bincat = len(DSParams['confounders']['bincat_cols'])
            DSParams['confounders']['mulcat_cols'] = DSParams['confounders']['mulcat_cols'].split(',')
            self.hparams.n_confounders_mulcat = len(DSParams['confounders']['mulcat_cols'])
            DSParams['confounders']['cont_cols'] = [c for c in DSParams['confounders']['cols'] if c not in DSParams['confounders']['mulcat_cols'] and c not in DSParams['confounders']['bincat_cols']]
            self.hparams.n_confounders_cont = len(DSParams['confounders']['cont_cols'])

        # Prepare transforms (TODO: move to a separate function later)
        if self.hparams.taskID != 2:
            trans = []
            if "interp_fact" in self.hparams and self.hparams.interp_fact != 1:
                trans.append(pytTransforms.Interpolate(factor=self.hparams.interp_fact))
            if self.hparams.croppad:
                trans.append(pytTransforms.CropOrPad(size=self.hparams.input_shape))
            if self.hparams.norm_type != "zscore":
                trans.extend(
                    (
                        pytTransforms.IntensityNorm(type=self.hparams.norm_type),
                        pytTransforms.ToTensor(dim=3 if self.hparams.is3D else 2),
                    )
                )
            else:
                self.hparams.zscore_mean = tuple(float(i) for i in self.hparams.zscore_mean.split(','))
                self.hparams.zscore_std = tuple(float(i) for i in self.hparams.zscore_std.split(','))
                trans.extend(
                    (
                        pytTransforms.ToTensor(dim=3 if self.hparams.is3D else 2),
                        pytTransforms.ZScoreNorm(dim=3 if self.hparams.is3D else 2, mean=self.hparams.zscore_mean, std=self.hparams.zscore_std, prenorm=self.hparams.zscore_prenorm if "zscore_prenorm" in self.hparams else ""), 
                    )
                )
            
            if "grey2RGB" in self.hparams and self.hparams.grey2RGB > -1:
                trans.append(pytTransforms.Grey2RGB(ch_dim=0, mode=self.hparams.grey2RGB))

            if self.hparams.is_complexDS and self.hparams.config_params['data']['complex_mode'] != 0:
                trans.append(pytTransforms.ComplexModeConverter(complex_mode=self.hparams.config_params['data']['complex_mode']))
            transform = tvTransforms.Compose(trans)
            ###############
            # Prepare augmentation (TODO: move to a separate function later)

            if "p_augment" in self.hparams and self.hparams.p_augment > 0:
                augmentations = []
                if self.hparams.p_aug_horflip > 0:
                    augmentations.append(pytAugmentations.TVTransformWrapper(torchV_trans_obj=tvTransforms.RandomHorizontalFlip(p=1), p=self.hparams.p_aug_horflip))
                augmentations = tvTransforms.Compose(augmentations) if augmentations else None
            else:
                augmentations = None
            ###############
        else:
            transform = None
            augmentations = None
            print("Main Engine: No transforms or augmentations are applied for the classification task engine!")
        
        if self.hparams.taskID == 2:
            self.hparams.foldCSV = f"{self.hparams.datadir}/{self.hparams.foldCSV}" if bool(self.hparams.foldCSV) else ""
            self.hparams.datadir = f"{self.hparams.parent_train_path}/{self.hparams.parent_outfolder}"
            self.hparams.n_save_recon_subs = 1
            self.hparams.save_recon_subs = False
            DSParams['fetch_phenotypes'] = True
            self.datamodule = LatentDataModule(data_dir=self.hparams.datadir, foldCSV=self.hparams.foldCSV, foldID=self.hparams.foldID,DSParams=DSParams,
                                                transform=transform, augmentations=augmentations, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, data_prefetch=self.hparams.nbatch_prefetch, n_save_recon_subs=self.hparams.n_save_recon_subs, save_recon_subs=self.hparams.save_recon_subs,
                                                use_concurrent_dataloader=self.hparams.use_concurrent_dataloader if "use_concurrent_dataloader" in self.hparams else False,)
        else:
            data_file = self.hparams.datafile if "datafile" in self.hparams else "data.h5"                
            self.datamodule = UKBBImgDataModule(data_dir=self.hparams.datadir, data_file=data_file, foldCSV=self.hparams.foldCSV, foldID=self.hparams.foldID,DSParams=DSParams,
                                                transform=transform, augmentations=augmentations, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, data_prefetch=self.hparams.nbatch_prefetch, n_save_recon_subs=self.hparams.n_save_recon_subs, save_recon_subs=self.hparams.save_recon_subs,
                                                use_concurrent_dataloader=self.hparams.use_concurrent_dataloader if "use_concurrent_dataloader" in self.hparams else False,)
        

    def __prep_trainer(self):  # sourcery skip: low-code-quality
        self.hparams.IsNegLoss = IS_NEG_LOSS[self.hparams.lossID] 
        self.hparams.accumulate_gradbatch = self.hparams.effective_batch_size // self.hparams.batch_size
        if self.hparams.accumulate_gradbatch == 0:
            self.hparams.accumulate_gradbatch = 1
        self.amp = (self.hparams.ampmode != "32")

        self.hparams.do_val = self.hparams.run_mode in [1, 4]
        
        if self.hparams.taskID == 2:
            self.hparams.input_shape = (len(self.hparams.DS_phenotypes['cols'].split(",")) ,)
        else:
            self.hparams.input_shape = tuple(int(i) for i in self.hparams.input_shape.split(','))

        if self.hparams.lr_decay_type == 1:
            self.hparams.lrScheduler_func = optim.lr_scheduler.StepLR
            self.hparams.lrScheduler_param_dict = {"step_size": int(self.hparams.config_params['training']['LRDecay']['type1']['decay_nepoch']),
                                                   "gamma": float(self.hparams.config_params['training']['LRDecay']['type1']['decay_rate'])}
        elif self.hparams.lr_decay_type == 2:
            self.hparams.lrScheduler_func = optim.lr_scheduler.ReduceLROnPlateau
            self.hparams.lrScheduler_param_dict = {"factor": self.hparams.config_params['training']['LRDecay']['type2']['decay_rate'],
                                                   # We can keep it always as min, because we are always minimising the loss (even if the loss is negative, using IsNegLoss param).
                                                   "mode": "min",
                                                   "patience": int(self.hparams.config_params['training']['LRDecay']['type2']['patience']),
                                                   "threshold": float(self.hparams.config_params['training']['LRDecay']['type2']['threshold']),
                                                   "cooldown": int(self.hparams.config_params['training']['LRDecay']['type2']['cooldown']),
                                                   "min_lr": float(self.hparams.config_params['training']['LRDecay']['type2']['min_lr'])}

        if self.hparams.resume:
            if bool(self.hparams.load_model_4test):
                logging.debug("Main Engine: Loading trained model for testing...")
                path2chk = self.hparams.load_model_4test
            else:
                path2chk = pjoin(self.hparams.save_path, self.hparams.run_name)
            checkpoint_dir = pjoin(path2chk, "Checkpoints")
            if self.hparams.load_best:                
                available_checkpoints = {int(c.split("epoch=")[1].split(
                    "-")[0]): c for c in [x for x in os.listdir(checkpoint_dir) if "epoch" in x]}                
            else:
                available_checkpoints={0 if c == 'last.ckpt' else int(c.split("last-")[1].replace(".ckpt", "").replace("v", "")): c for c in [x for x in os.listdir(checkpoint_dir) if "last" in x]}
            self.chkpoint = pjoin(checkpoint_dir, available_checkpoints[sorted(
                    list(available_checkpoints.keys()))[-1]])
        else:
            self.chkpoint = None

        match self.hparams.taskID:
            case 0:
                self.model = ReconEngine(**vars(self.hparams))
            case 1:
                self.hparams.modelID = -13
                self.model = DiffAEEngine(**vars(self.hparams))
            case 2:
                self.model = ClassifyLatentEngine(**vars(self.hparams))
            case _:
                logging.error("Main Engine: Invalid taskID")
                sys.exit("Main Engine: Invalid taskID")
            
        if bool(self.hparams.load_hf):
            print("Main Engine: Loading pretrained model from HuggingFace...")
            modelHF = AutoModel.from_pretrained(self.hparams.load_hf, trust_remote_code=True)
            self.model.load_state_dict(modelHF.state_dict())

        if self.hparams.run_mode == 2 and bool(self.chkpoint):
            # TODO (from NCC1701) ckpt_path is not working during testing if not trained in the same run. So loading explicitly. check why
            logging.debug("Main Engine: Loading existing checkpoint... [Hugging Face model (if loaded) will be ignored]")
            state_dict = torch.load(self.chkpoint)
            if self.hparams.taskID == 1 and state_dict['state_dict']['x_T'].shape != self.model.x_T.shape:
                logging.debug("Main Engine (DiffAE): Input shape mismatch between the trained model and the current model. Ignoring the buffer while loading the weights...")
                state_dict['state_dict']['x_T'] = self.model.x_T                
                with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as temp_file:
                    torch.save(state_dict, temp_file.name)    
                    self.chkpoint = temp_file.name         
            self.model.load_state_dict(state_dict['state_dict'])
                            
        self.model.lr = self.hparams.lr #This is required for the Auto LR finder to work

        loggers = []
        if self.hparams.wnbactive:
            loggers.append(WandbLogger(name=self.hparams.run_name, id=self.hparams.run_name, project=self.hparams.wnbproject,
                                       group=self.hparams.wnbgroup, entity=self.hparams.wnbentity, config=self.hparams))
            if bool(self.hparams.wnbmodellog) and self.hparams.wnbmodellog != "None":
                loggers[-1].watch(self.model, log=self.hparams.wnbmodellog,
                                  log_freq=self.hparams.wnbmodelfreq)
        else:
            os.environ["WANDB_MODE"] = "dryrun"
        if self.hparams.tbactive:
            # TODO (from NCC1701) log_graph as True making it crash due to backward hooks
            os.makedirs(self.hparams.tblog_path, exist_ok=True)
            loggers.append(TensorBoardLogger(self.hparams.tblog_path,
                           name=self.hparams.run_name, log_graph=False))

        if self.hparams.complie_model:
            self.model = torch.compile(self.model)

        checkpoint_callback = ModelCheckpoint(
            dirpath=pjoin(self.hparams.save_path,
                          self.hparams.run_name, "Checkpoints"),
            monitor='val_loss',
            save_last=True,
        )

        try:
            precision = int(self.hparams.ampmode)
        except:
            precision = self.hparams.ampmode #This is in the case of bf16
            
        trainer_params = {
            "accelerator": self.hparams.accelerator,
            "callbacks": [checkpoint_callback],
            "check_val_every_n_epoch": 1 if self.hparams.do_val else self.hparams.num_epochs+1,
            "detect_anomaly": self.hparams.check_anomalies,
            "deterministic": not self.hparams.non_deter,
            "devices": self.hparams.ndevices if bool(self.hparams.ndevices) else 1,
            "limit_train_batches": self.hparams.n_train_batches,
            "limit_val_batches": self.hparams.n_val_batches,
            "logger": loggers,
            "log_every_n_steps": self.hparams.config_params['training']['log_freq'],
            "precision": precision,
            "num_nodes": self.hparams.nnodes,
            "max_epochs": self.hparams.num_epochs,
            "fast_dev_run": abs(self.hparams.dev_run) if bool(abs(self.hparams.dev_run)) else False,
            "sync_batchnorm": self.hparams.sync_batchnorm
        }
        if self.hparams.auto_optim:
            trainer_params.update({
                "accumulate_grad_batches": self.hparams.accumulate_gradbatch,
                "gradient_clip_val": self.hparams.grad_clip_val if bool(self.hparams.grad_clip_val) and bool(self.hparams.grad_clip_algo) else None,
                "gradient_clip_algorithm": self.hparams.grad_clip_algo if bool(self.hparams.grad_clip_algo) else None,            
            })
        if not self.hparams.do_val:
            trainer_params["num_sanity_val_steps"] = 0
        if bool(self.hparams.training_strategy) and self.hparams.training_strategy != "default":
            trainer_params["strategy"] = self.hparams.training_strategy
        if bool(self.hparams.profiler):
            trainer_params["profiler"] = self.hparams.profiler
        if self.hparams.dev_run < 0:
            trainer_params["barebones"] = True
        self.trainer = Trainer(**trainer_params)

        if not self.hparams.non_deter:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                torch.use_deterministic_algorithms(True)

        self.train_done = self.hparams.run_mode == 2 and bool(
            self.hparams.preweights_path
        )

        if self.hparams.auto_bs or self.hparams.auto_lr:
            self.align()

    def align(self):
        tuner = Tuner(self.trainer)
        
        if self.hparams.auto_bs:
            tuner.scale_batch_size(self.model, mode="binsearch", datamodule=self.datamodule)
                        
            # The tune method updated the batch size param inside the data module, so we need to update the hparams and trainer accordingly (not neccesary, but cleaner). Updating accumulate_grad_batches is required
            self.hparams.batch_size = self.datamodule.batch_size
            self.hparams.accumulate_gradbatch = self.hparams.effective_batch_size // self.datamodule.batch_size
            if self.hparams.accumulate_gradbatch == 0: #If it's 0, it means that the batch size is bigger than the effective batch size, so we set it to None (the default value)
                self.hparams.accumulate_gradbatch = 1
            self.trainer.accumulate_grad_batches = self.hparams.accumulate_gradbatch
            logging.debug(f"Main Engine: Auto Batch Size found and set the batch_size to be: {self.hparams.batch_size}")
            
        if self.hparams.auto_lr:
            lr_finder = tuner.lr_find(self.model)
            fig = lr_finder.plot(suggest=True)
            plt.savefig(pjoin(self.hparams.save_path, self.hparams.run_name, "auto_lr_graph.png"))
            self.hparams.lr = lr_finder.suggestion()
            logging.debug(f"Main Engine: Auto Learning Rate found and set the learning rate to be: {self.hparams.lr}")

        torch.cuda.empty_cache()

    def train(self):
        self.trainer.fit(model=self.model,
                         ckpt_path=self.chkpoint, datamodule=self.datamodule)
        self.train_done = True

    def test(self):
        os.makedirs(self.hparams.res_path, exist_ok=True)

        if self.train_done:
            if self.hparams.do_val:
                self.trainer.test(datamodule=self.datamodule)
            else:
                self.trainer.test(model=self.model, datamodule=self.datamodule)
        else:
            self.trainer.test(
                model=self.model, ckpt_path=self.chkpoint, datamodule=self.datamodule)

    def predict(self):
        os.makedirs(self.hparams.res_path, exist_ok=True)
        self.trainer.predict(
            model=self.model, ckpt_path=self.chkpoint, datamodule=self.datamodule)

    def engage(self):
        if self.hparams.run_mode in [0, 1, 3, 4]:
            self.train()
        if self.hparams.run_mode in [2, 3, 4]:
            self.test()
