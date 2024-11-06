#Renames: self.model to self.net, self.ema_model to self.ema_net

import copy
import json
import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.lib.function_base import flip
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
# from torch.cuda import amp
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torch.utils.data.dataset import ConcatDataset, TensorDataset
from torchvision.utils import make_grid, save_image

from ..WarpDrives.DiffAE.config import *
from ..WarpDrives.DiffAE.dist_utils import *
# from ..WarpDrives.DiffAE.lmdb_writer import *
# from ..WarpDrives.DiffAE.metrics import *
from ..WarpDrives.DiffAE.renderer import *
from ..WarpDrives.DiffAE.templates import *

from .ReconEngine import ReconEngine

from ...utilities import PlasmaConduit, ResHandler, AdditionalResHandler, Res2H5, get_nested_attribute, MetricsSave, filter_dict_keys
from Engineering.utilities import (CustomInitialiseWeights, DataHandler,
                                   DataSpaceHandler, ResSaver, Evaluator, fetch_vol_subds, fetch_vol_subds_fastMRI, getSSIM,
                                   log_images, process_slicedict, process_testbatch)

class DiffAEEngine(ReconEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        conf = ukbb_autoenc(n_latents=kwargs['config_params']['model']['latent_dim']) 
        conf.__dict__.update(kwargs) #update the detault ones with the supplied ones
        conf.__dict__.update(kwargs['config_params']['model']['DiffAE']) #update the supplied DiffAE params 
        
        if self.hparams.test_with_TEval:
            conf.T_inv = conf.T_eval
            conf.T_step = conf.T_eval
            
        if self.hparams.test_emb_only:
            self.hparams.config_params['training']['save_recon'] = False
            
        conf.fp16 = self.hparams.ampmode not in ["32", "32-true"]
            
        conf.refresh_values()
        conf.make_model_conf()

        self.save_hyperparameters(conf.as_dict_jsonable())
        self.conf = conf
        
        self.net = conf.make_model_conf().make_model()
        self.ema_net = copy.deepcopy(self.net)
        self.ema_net.requires_grad_(False)
        self.ema_net.eval()

        model_size = sum(param.data.nelement() for param in self.net.parameters())
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()

        # this is shared for both model and latent
        self.T_sampler = conf.make_T_sampler()

        if conf.train_mode.use_latent_net():
            self.latent_sampler = conf.make_latent_diffusion_conf(
            ).make_sampler()
            self.eval_latent_sampler = conf.make_latent_eval_diffusion_conf(
            ).make_sampler()
        else:
            self.latent_sampler = None
            self.eval_latent_sampler = None

        # initial variables for consistent sampling
        self.register_buffer('x_T', torch.randn(conf.sample_size, conf.in_channels, *conf.input_shape))

        if conf.pretrain is not None: #TODO: make this work, to load pretrained model
            print(f'loading pretrain ... {conf.pretrain.name}')
            state = torch.load(conf.pretrain.path, map_location='cpu')
            print('step:', state['global_step'])
            self.load_state_dict(state['state_dict'], strict=False)

        if conf.latent_infer_path is not None:
            print('loading latent stats ...')
            state = torch.load(conf.latent_infer_path)
            self.conds = state['conds']
            self.register_buffer('conds_mean', state['conds_mean'][None, :])
            self.register_buffer('conds_std', state['conds_std'][None, :])
        else:
            self.conds_mean = None
            self.conds_std = None
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nothing_yet', type=str)
        return parser
    
    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def sample(self, N, device, T=None, T_latent=None):
        if T is None:
            sampler = self.eval_sampler
            latent_sampler = self.latent_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
            latent_sampler = self.conf._make_latent_diffusion_conf(T_latent).make_sampler()

        noise = torch.randn(N,
                            self.conf.in_channels,
                            *self.conf.input_shape,
                            device=device)
        pred_img = render_uncondition(
            self.conf,
            self.ema_net,
            noise,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conds_mean=self.conds_mean,
            conds_std=self.conds_std,
        )
        pred_img = (pred_img + 1) / 2
        return pred_img

    def render(self, noise, cond=None, T=None, use_ema=True):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()

        if cond is not None:
            pred_img = render_condition(self.conf,
                                        self.ema_net if use_ema else self.net,
                                        noise,
                                        sampler=sampler,
                                        cond=cond)
        else:
            pred_img = render_uncondition(self.conf,
                                          self.ema_net if use_ema else self.net,
                                          noise,
                                          sampler=sampler,
                                          latent_sampler=None)
        pred_img = (pred_img + 1) / 2
        return pred_img

    def encode(self, x, use_ema=True):
        # TODO: What? Why? From the original authors!
        assert self.conf.model_type.has_autoenc()
        return self.ema_net.encoder.forward(x) if use_ema else self.net.encoder.forward(x)

    def encode_stochastic(self, x, cond, T=None, use_ema=True):
        if T is None:
            sampler = self.eval_sampler
        else:
            sampler = self.conf._make_diffusion_conf(T).make_sampler()
        out = sampler.ddim_reverse_sample_loop(self.ema_net if use_ema else self.net,
                                               x,
                                               model_kwargs={'cond': cond})
        return out['sample']

    def forward(self, x_start=None, noise=None, ema_model: bool = False):
        with amp.autocast(False):
            model = self.ema_net if ema_model else self.net
            return self.eval_sampler.sample(
                model=model,
                noise=noise,
                x_start=x_start,
                shape=noise.shape if noise is not None else x_start.shape,
            )

    # def get_inferred_latent_DS(self):
    #     """
    #     if latent mode => return the inferred latent dataset
    #     """
    #     print('on train dataloader start ...')
    #     if not self.conf.train_mode.require_dataset_infer():
    #         return None
    #     if self.conds is None:
    #         # usually we load self.conds from a file
    #         # so we do not need to do this again!
    #         self.conds = self.infer_whole_dataset()
    #         # need to use float32! unless the mean & std will be off!
    #         # (1, c)
    #         self.conds_mean.data = self.conds.float().mean(dim=0,
    #                                                        keepdim=True)
    #         self.conds_std.data = self.conds.float().std(dim=0,
    #                                                      keepdim=True)
    #     print('mean:', self.conds_mean.mean(), 'std:',
    #           self.conds_std.mean())

    #     # return the dataset with pre-calculated conds
    #     conf = self.conf.clone()
    #     conf.batch_size = self.batch_size
    #     data = TensorDataset(self.conds)
    #     return conf.make_loader(data, shuffle=True)
    
    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop? 
        used with gradient_accum > 1 and to see if the optimizer will perform "step" in this iteration or not
        """
        return (batch_idx + 1) % self.conf.accum_batches == 0
    
# def infer_whole_dataset(self,
    #                         with_render=False,
    #                         T_render=None,
    #                         render_save_path=None):    #TODO: not ready
    #     """
    #     predicting the latents given images using the encoder

    #     Args:
    #         both_flips: include both original and flipped images; no need, it's not an improvement
    #         with_render: whether to also render the images corresponding to that latent
    #         render_save_path: lmdb output for the rendered images
    #     """
    #     data = self.conf.make_dataset()
    #     data.transform = self.make_transform(self.conf.img_size, flip_prob=0)

    #     # data = SubsetDataset(data, 21)

    #     loader = self.conf.make_loader(
    #         data,
    #         shuffle=False,
    #         drop_last=False,
    #         batch_size=self.conf.batch_size_eval,
    #         parallel=True,
    #     )
    #     model = self.ema_net
    #     model.eval()
    #     conds = []

    #     if with_render:
    #         sampler = self.conf._make_diffusion_conf(
    #             T=T_render or self.conf.T_eval).make_sampler()

    #         if self.global_rank == 0:
    #             writer = LMDBImageWriter(render_save_path,
    #                                      format='webp',
    #                                      quality=100)
    #         else:
    #             writer = nullcontext()
    #     else:
    #         writer = nullcontext()

    #     with writer:
    #         for batch in tqdm(loader, total=len(loader), desc='infer'):
    #             with torch.no_grad():
    #                 # (n, c)
    #                 # print('idx:', batch['index'])
    #                 cond = model.encoder(batch['img'].to(self.device))

    #                 # used for reordering to match the original dataset
    #                 idx = batch['index']
    #                 idx = self.all_gather(idx)
    #                 if idx.dim() == 2:
    #                     idx = idx.flatten(0, 1)
    #                 argsort = idx.argsort()

    #                 if with_render:
    #                     noise = torch.randn(len(cond),
    #                                         self.conf.in_channels,
    #                                         *self.conf.input_shape,
    #                                         device=self.device)
    #                     render = sampler.sample(model, noise=noise, cond=cond)
    #                     render = (render + 1) / 2
    #                     # print('render:', render.shape)
    #                     # (k, n, c, h, w)
    #                     render = self.all_gather(render)
    #                     if render.dim() == 5:
    #                         # (k*n, c)
    #                         render = render.flatten(0, 1)

    #                     # print('global_rank:', self.global_rank)

    #                     if self.global_rank == 0:
    #                         writer.put_images(render[argsort])

    #                 # (k, n, c)
    #                 cond = self.all_gather(cond)

    #                 if cond.dim() == 3:
    #                     # (k*n, c)
    #                     cond = cond.flatten(0, 1)

    #                 conds.append(cond[argsort].cpu())
    #             # break
    #     model.train()
    #     return torch.cat(conds).float()
    
    def training_step(self, batch, batch_idx):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with amp.autocast(False):
            # forward
            if self.conf.train_mode.require_dataset_infer():
                # this mode as pre-calculated cond
                cond = batch[0]
                if self.conf.latent_znormalize:
                    cond = (cond - self.conds_mean.to(
                        self.device)) / self.conds_std.to(self.device)
            else:
                imgs, idxs = batch['inp']['data'], batch_idx
                # print(f'(rank {self.global_rank}) batch size:', len(imgs))
                x_start = imgs

            if self.conf.train_mode == TrainMode.diffusion:
                """
                main training mode!!!
                """
                # with numpy seed we have the problem that the sample t's are related!
                t, weight = self.T_sampler.sample(len(x_start), x_start.device)
                losses = self.sampler.training_losses(model=self.net,
                                                        x_start=x_start,
                                                        t=t)
            elif self.conf.train_mode.is_latent_diffusion():
                """
                training the latent variables!
                """
                # diffusion on the latent
                t, weight = self.T_sampler.sample(len(cond), cond.device)
                latent_losses = self.latent_sampler.training_losses(
                    model=self.net.latent_net, x_start=cond, t=t)
                # train only do the latent diffusion
                losses = {
                    'latent': latent_losses['loss'],
                    'loss': latent_losses['loss']
                }
            else:
                raise NotImplementedError()

            loss = losses['loss'].mean()
            loss_dict = {"train_loss": loss}
            for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    loss_dict[f'train_{key}'] = losses[key].mean()
            self.log_dict(loss_dict, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['inp']['data'].shape[0])

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.net.latent_net, self.ema_net.latent_net,
                    self.conf.ema_decay)
            else:
                ema(self.net, self.ema_net, self.conf.ema_decay)

            # logging
            imgs = None if self.conf.train_mode.require_dataset_infer() else batch['inp']['data']
            self.log_sample(x_start=imgs, batch_idx=(self.current_epoch*self.hparams.batch_size)+batch_idx) #Sampling the learnt models and saving images to TB. 
            # self.evaluate_scores() #TODO: (Soumick) Calculates FID and LPIPS. But our code isn't ready for it, also we don't need it most likely.

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        if self.conf.grad_clip > 0:
            # from trainer.params_grads import grads_norm, iter_opt_params
            params = [
                p for group in optimizer.param_groups for p in group['params']
            ]
            # print('before:', grads_norm(iter_opt_params(optimizer)))
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=self.conf.grad_clip)
            # print('after:', grads_norm(iter_opt_params(optimizer)))
    
    #Validation  
    def validation_step(self, batch, batch_idx):
        _, prediction_ema = self.inference_pass(batch['inp']['data'], T_inv=self.conf.T_eval, T_step=self.conf.T_eval, use_ema=True)
        _, prediction_base = self.inference_pass(batch['inp']['data'], T_inv=self.conf.T_eval, T_step=self.conf.T_eval, use_ema=False)        

        inp = batch['inp']['data'].cpu() 
        inp = (inp + 1) / 2
        
        _, val_ssim_ema = self._eval_prediction(inp, prediction_ema)
        _, val_ssim_base = self._eval_prediction(inp, prediction_base)
        
        self.log_dict({"val_ssim_ema": val_ssim_ema, "val_ssim_base": val_ssim_base, "val_loss": -val_ssim_ema}, on_step=True, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=batch['inp']['data'].shape[0])
        self.img_logger("val_ema", batch_idx, inp, prediction_ema)
        self.img_logger("val_base", batch_idx, inp, prediction_base)
        
    def _eval_prediction(self, inp, prediction):
        prediction = prediction.detach().cpu() 
        prediction = prediction.numpy() if prediction.dtype not in {torch.bfloat16, torch.float16} else prediction.to(dtype=torch.float32).numpy()
        if self.hparams.grey2RGB in [0, 2]:
            inp = inp[:, 1, ...].unsqueeze(1)
            prediction = np.expand_dims(prediction[:, 1, ...], axis=1)
        val_ssim = getSSIM(inp.numpy(), prediction, data_range=1) 
        return prediction, val_ssim
        
    def inference_pass(self, inp, T_inv, T_step, use_ema=True):
        semantic_latent = self.encode(inp, use_ema=use_ema) 
        if self.hparams.test_emb_only:
            return semantic_latent, None
        stochastic_latent = self.encode_stochastic(inp, semantic_latent, T=T_inv) 
        prediction = self.render(stochastic_latent, semantic_latent, T=T_step, use_ema=use_ema) 
        return semantic_latent, prediction
    
    # Testing
    # on_test_start is inherited from ReconEngine
    def test_step(self, batch, batch_idx):
        emb, recon = self.inference_pass(batch['inp']['data'], T_inv=self.conf.T_inv, T_step=self.conf.T_step, use_ema=self.hparams.test_ema)

        emb = emb.detach().cpu()
        emb = emb.numpy() if emb.dtype not in {torch.bfloat16, torch.float16} else emb.to(dtype=torch.float32).numpy()

        if self.hparams.test_emb_only:
            self.res_holder.store_res(batch, emb)
        else:
            recon = recon.detach().cpu()
            recon = recon.numpy() if recon.dtype not in {torch.bfloat16, torch.float16} else recon.to(dtype=torch.float32).numpy()

            gt = batch['inp']['data'].cpu().numpy()
            gt = (gt + 1) / 2
            
            if self.hparams.grey2RGB in [0, 2]:
                gt = np.expand_dims(gt[:, 1, ...], axis=1)
                recon = np.expand_dims(recon[:, 1, ...], axis=1)

            scores = self.evaluator.get_batch_scores(gt=gt, out=recon, keys=batch['key'], tag="") 
            self.metrics_list.append(scores)
            
            ssim = getSSIM(gt, recon, data_range=1)        
            self.res_holder.store_res(batch, emb, recon)   
            
            self.log('test_ssim'+(f"_{self.hparams.output_suffix}" if bool(self.hparams.output_suffix) else ""), ssim, on_step=False, on_epoch=True, reduce_fx="mean", sync_dist=True, batch_size=gt.shape[0])
        
    # on_test_end is inherited from ReconEngine

    #Prediction
    def predict_step(self, batch, batch_idx):
        emb = self.encode(batch['inp']['data']).detach().cpu()
        return emb.numpy() if emb.dtype not in {torch.bfloat16, torch.float16} else emb.to(dtype=torch.float32).numpy()
    
    def log_sample(self, x_start, batch_idx):
        """
        put images to the tensorboard
        """
        def do(model,
                   postfix,
                   use_xstart,
                   save_real=False,
                   no_latent_diff=False,
                   interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.split_tensor(self.x_T)
                batch_size = min(len(all_x_T), self.hparams.batch_size)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    _xstart = x_start[:len(x_T)] if use_xstart else None
                    if self.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.eval_sampler,
                            latent_sampler=self.eval_latent_sampler,
                            conds_mean=self.conds_mean,
                            conds_std=self.conds_std)
                    else:
                        if not use_xstart and self.conf.model_type.has_noise_to_cond(
                        ):
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T),
                                               self.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.eval_sampler.sample(model=model,
                                                       noise=x_T,
                                                       cond=cond,
                                                       x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                
                if self.hparams.is3D:
                    central_slice = gen.shape[-3] // 2
                    gen = gen[..., central_slice, :, :]
                    
                if self.hparams.grey2RGB in [0, 2]:
                    gen = gen[:, 1, ...].unsqueeze(1)
                    
                if (gen.dim() == 5): 
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                
                    if self.hparams.is3D:
                        central_slice = real.shape[-3] // 2
                        real = real[..., central_slice, :, :]
                    
                    if self.hparams.grey2RGB in [0, 2]:
                        real = real[:, 1, ...].unsqueeze(1)
                    
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        grid_real = (make_grid(real) + 1) / 2
                        self.loggers[-1].experiment.add_image(f'sample{postfix}/real', grid_real, batch_idx)

                if self.global_rank == 0:
                    # save samples to the tensorboard
                    grid = (make_grid(gen) + 1) / 2
                    # sample_dir = os.path.join(self.conf.logdir,
                    #                           f'sample{postfix}')
                    # if not os.path.exists(sample_dir):
                    #     os.makedirs(sample_dir)
                    # path = os.path.join(sample_dir,
                    #                     '%d.png' % batch_idx)
                    # save_image(grid, path)
                    self.loggers[-1].experiment.add_image(f'sample{postfix}/generated', grid, batch_idx)
            model.train()

        if self.hparams.tbactive and self.conf.sample_every_batches > 0 and batch_idx % self.conf.sample_every_batches == 0:

            if self.conf.train_mode.require_dataset_infer():
                do(self.net, '', use_xstart=False)
                do(self.ema_net, '_ema', use_xstart=False)
            else:
                if self.conf.model_type.has_autoenc(
                ) and self.conf.model_type.can_sample():
                    do(self.net, '', use_xstart=False)
                    do(self.ema_net, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.net, '_enc', use_xstart=True, save_real=True)
                    do(self.ema_net,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                elif self.conf.train_mode.use_latent_net():
                    do(self.net, '', use_xstart=False)
                    do(self.ema_net, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.net, '_enc', use_xstart=True, save_real=True)
                    do(self.net,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.ema_net,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.net, '', use_xstart=True, save_real=True)
                    do(self.ema_net, '_ema', use_xstart=True, save_real=True)

    # def evaluate_scores(self):
    #     """
    #     evaluate FID and other scores during training (put to the tensorboard)
    #     For, FID. It is a fast version with 5k images (gold standard is 50k).
    #     Don't use its results in the paper!
    #     """
    #     def fid(model, postfix):
    #         score = evaluate_fid(self.eval_sampler,
    #                              model,
    #                              self.conf,
    #                              device=self.device,
    #                              train_data=self.train_data,
    #                              val_data=self.val_data,
    #                              latent_sampler=self.eval_latent_sampler,
    #                              conds_mean=self.conds_mean,
    #                              conds_std=self.conds_std)
    #         if self.global_rank == 0:
    #             self.loggers[-1].experiment.add_scalar(f'FID{postfix}', score,
    #                                               self.num_samples)
    #             if not os.path.exists(self.conf.logdir):
    #                 os.makedirs(self.conf.logdir)
    #             with open(os.path.join(self.conf.logdir, 'eval.txt'),
    #                       'a') as f:
    #                 metrics = {
    #                     f'FID{postfix}': score,
    #                     'num_samples': self.num_samples,
    #                 }
    #                 f.write(json.dumps(metrics) + "\n")

    #     def lpips(model, postfix):
    #         if self.conf.model_type.has_autoenc(
    #         ) and self.conf.train_mode.is_autoenc():
    #             # {'lpips', 'ssim', 'mse'}
    #             score = evaluate_lpips(self.eval_sampler,
    #                                    model,
    #                                    self.conf,
    #                                    device=self.device,
    #                                    val_data=self.val_data,
    #                                    latent_sampler=self.eval_latent_sampler)

    #             if self.global_rank == 0:
    #                 for key, val in score.items():
    #                     self.loggers[-1].experiment.add_scalar(
    #                         f'{key}{postfix}', val, self.num_samples)

    #     if self.conf.eval_every_samples > 0 and self.num_samples > 0 and is_time(
    #             self.num_samples, self.conf.eval_every_samples,
    #             self.conf.batch_size_effective):
    #         print(f'eval fid @ {self.num_samples}')
    #         lpips(self.net, '')
    #         fid(self.net, '')

    #     if self.conf.eval_ema_every_samples > 0 and self.num_samples > 0 and is_time(
    #             self.num_samples, self.conf.eval_ema_every_samples,
    #             self.conf.batch_size_effective):
    #         print(f'eval fid ema @ {self.num_samples}')
    #         fid(self.ema_net, '_ema')
    #         # it's too slow
    #         # lpips(self.ema_net, '_ema')

    def configure_optimizers(self):
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(self.net.parameters(),
                                     lr=self.conf.lr,
                                     weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(self.net.parameters(),
                                      lr=self.conf.lr,
                                      weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out = {'optimizer': optim}
        if self.conf.warmup > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                      lr_lambda=WarmupLR(
                                                          self.conf.warmup))
            out['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
            }
        return out

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding "worker" in the batch dimension

        Args:
            x: (n, c)

        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank:(rank + 1) * per_rank]

    # def test_step_original(self, batch, *args, **kwargs):
    #     """
    #     for the "eval" mode. 
    #     We first select what to do according to the "conf.eval_programs". 
    #     test_step will only run for "one iteration" (it's a hack!).
        
    #     We just want the multi-gpu support. 
    #     """
    #     # make sure you seed each worker differently!

    #     # it will run only one step!
    #     print('global step:', self.global_step)
    #     """
    #     "infer" = predict the latent variables using the encoder on the whole dataset
    #     """
    #     if 'infer' in self.conf.eval_programs:
    #         print('infer ...')
    #         conds = self.infer_whole_dataset().float()
    #         # NOTE: always use this path for the latent.pkl files
    #         save_path = f'checkpoints/{self.conf.name}/latent.pkl'
    #         if self.global_rank == 0:
    #             conds_mean = conds.mean(dim=0)
    #             conds_std = conds.std(dim=0)
    #             if not os.path.exists(os.path.dirname(save_path)):
    #                 os.makedirs(os.path.dirname(save_path))
    #             torch.save(
    #                 {
    #                     'conds': conds,
    #                     'conds_mean': conds_mean,
    #                     'conds_std': conds_std,
    #                 }, save_path)
    #     """
    #     "infer+render" = predict the latent variables using the encoder on the whole dataset
    #     THIS ALSO GENERATE CORRESPONDING IMAGES
    #     """
    #     # infer + reconstruction quality of the input
    #     for each in self.conf.eval_programs:
    #         if each.startswith('infer+render'):
    #             m = re.match(r'infer\+render([0-9]+)', each)
    #             if m is not None:
    #                 T = int(m[1])
    #                 self.setup()
    #                 print(f'infer + reconstruction T{T} ...')
    #                 conds = self.infer_whole_dataset(
    #                     with_render=True,
    #                     T_render=T,
    #                     render_save_path=
    #                     f'latent_infer_render{T}/{self.conf.name}.lmdb',
    #                 )
    #                 save_path = f'latent_infer_render{T}/{self.conf.name}.pkl'
    #                 conds_mean = conds.mean(dim=0)
    #                 conds_std = conds.std(dim=0)
    #                 if not os.path.exists(os.path.dirname(save_path)):
    #                     os.makedirs(os.path.dirname(save_path))
    #                 torch.save(
    #                     {
    #                         'conds': conds,
    #                         'conds_mean': conds_mean,
    #                         'conds_std': conds_std,
    #                     }, save_path)

    #     # evals those "fidXX"
    #     """
    #     "fid<T>" = unconditional generation (conf.train_mode = diffusion).
    #         Note:   Diff. autoenc will still receive real images in this mode.
    #     "fid<T>,<T_latent>" = unconditional generation for latent models (conf.train_mode = latent_diffusion).
    #         Note:   Diff. autoenc will still NOT receive real images in this made.
    #                 but you need to make sure that the train_mode is latent_diffusion.
    #     """
    #     for each in self.conf.eval_programs:
    #         if each.startswith('fid'):
    #             m = re.match(r'fid\(([0-9]+),([0-9]+)\)', each)
    #             clip_latent_noise = False
    #             if m is not None:
    #                 # eval(T1,T2)
    #                 T = int(m[1])
    #                 T_latent = int(m[2])
    #                 print(f'evaluating FID T = {T}... latent T = {T_latent}')
    #             else:
    #                 m = re.match(r'fidclip\(([0-9]+),([0-9]+)\)', each)
    #                 if m is not None:
    #                     # fidclip(T1,T2)
    #                     T = int(m[1])
    #                     T_latent = int(m[2])
    #                     clip_latent_noise = True
    #                     print(
    #                         f'evaluating FID (clip latent noise) T = {T}... latent T = {T_latent}'
    #                     )
    #                 else:
    #                     # evalT
    #                     _, T = each.split('fid')
    #                     T = int(T)
    #                     T_latent = None
    #                     print(f'evaluating FID T = {T}...')

    #             self.train_dataloader()
    #             sampler = self.conf._make_diffusion_conf(T=T).make_sampler()
    #             if T_latent is not None:
    #                 latent_sampler = self.conf._make_latent_diffusion_conf(
    #                     T=T_latent).make_sampler()
    #             else:
    #                 latent_sampler = None

    #             conf = self.conf.clone()
    #             conf.eval_num_images = 50_000
    #             score = evaluate_fid(
    #                 sampler,
    #                 self.ema_net,
    #                 conf,
    #                 device=self.device,
    #                 train_data=self.train_data,
    #                 val_data=self.val_data,
    #                 latent_sampler=latent_sampler,
    #                 conds_mean=self.conds_mean,
    #                 conds_std=self.conds_std,
    #                 remove_cache=False,
    #                 clip_latent_noise=clip_latent_noise,
    #             )
    #             if T_latent is None:
    #                 self.log(f'fid_ema_T{T}', score)
    #             else:
    #                 name = 'fid'
    #                 if clip_latent_noise:
    #                     name += '_clip'
    #                 name += f'_ema_T{T}_Tlatent{T_latent}'
    #                 self.log(name, score)
    #     """
    #     "recon<T>" = reconstruction & autoencoding (without noise inversion)
    #     """
    #     for each in self.conf.eval_programs:
    #         if each.startswith('recon'):
    #             self.net: BeatGANsAutoencModel
    #             _, T = each.split('recon')
    #             T = int(T)
    #             print(f'evaluating reconstruction T = {T}...')

    #             sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

    #             conf = self.conf.clone()
    #             # eval whole val dataset
    #             conf.eval_num_images = len(self.val_data)
    #             # {'lpips', 'mse', 'ssim'}
    #             score = evaluate_lpips(sampler,
    #                                    self.ema_net,
    #                                    conf,
    #                                    device=self.device,
    #                                    val_data=self.val_data,
    #                                    latent_sampler=None)
    #             for k, v in score.items():
    #                 self.log(f'{k}_ema_T{T}', v)
    #     """
    #     "inv<T>" = reconstruction with noise inversion
    #     """
    #     for each in self.conf.eval_programs:
    #         if each.startswith('inv'):
    #             self.net: BeatGANsAutoencModel
    #             _, T = each.split('inv')
    #             T = int(T)
    #             print(
    #                 f'evaluating reconstruction with noise inversion T = {T}...'
    #             )

    #             sampler = self.conf._make_diffusion_conf(T=T).make_sampler()

    #             conf = self.conf.clone()
    #             # eval whole val dataset
    #             conf.eval_num_images = len(self.val_data)
    #             # {'lpips', 'mse', 'ssim'}
    #             score = evaluate_lpips(sampler,
    #                                    self.ema_net,
    #                                    conf,
    #                                    device=self.device,
    #                                    val_data=self.val_data,
    #                                    latent_sampler=None,
    #                                    use_inverted_noise=True)
    #             for k, v in score.items():
    #                 self.log(f'{k}_inv_ema_T{T}', v)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup

