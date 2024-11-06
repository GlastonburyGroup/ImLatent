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

from .ivaes_config import iVAEsConfig
from .ivaes_utils import Label_Prior, Label_Decoder, compute_posterior, kl_criterion

from ......utilities import PlasmaConduit

class iVAEs(BaseAE):
    """Identifiable VAE models.
    Contains three distinct models: iVAE, IDVAE, and CI-iVAE.

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
        model_config: iVAEsConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = f"iVAEs_{model_config.ivae_mode}"

        assert model_config.aggressive_post == False, "Aggressive posterior not implemented"
        assert model_config.kl_annealing == False, "KL annealing not implemented"

        self.kl_annealing_coeff = 1.0 #as it's not used

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

        self.beta = model_config.beta
        self.method = model_config.ivae_mode
        self.alpha_step = model_config.alpha_step

        self.dim_u = model_config.n_phenotypes
        self.dims_to_reduce = [1, 2, 3] if len(model_config.input_dim) == 2 else [1, 2, 3, 4]

        self.label_prior = Label_Prior(model_config.latent_dim, self.dim_u, hidden_nodes=model_config.hidden_nodes_label_prior)
        if self.method == 'IDVAE':
            self.label_decoder = Label_Decoder(self.dim_u, model_config.latent_dim, hidden_nodes=model_config.hidden_nodes_label_decoder)

        default_training_config_path = "Engineering/Engines/WarpDrives/pythaeDrive/configs/originals/celeba/base_training_config.json"
        self.default_training_config = BaseTrainerConfig.from_json_file(default_training_config_path)

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The forward pass of the iVAE models

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """        
        if not self.training:
            self.kl_annealing_coeff = 1.0

        x = inputs["data"]
        if "phenotypes" in inputs:
            x_pheno = inputs["phenotypes"]
        else:
            x_pheno = torch.randn(x.shape[0], self.dim_u).to(x.device)
            print("Warning: phenotypes were not received, initialising with random. It's fine for the initial forward pass check, but not for training.")
        
        #Add random noise to the input data to prevent posterior collapse (from CI-iVAE code)
        x += torch.randn_like(x)*1e-2 

        encoder_output = self.encoder(x)
        z_mean, z_log_var = encoder_output.embedding, encoder_output.log_covariance
        
        lam_mean, lam_log_var = self.label_prior(x_pheno) #The original code one-hot encodes the class lables. But we have continuous phenotypes

        post_mean, post_log_var = compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
        
        post_sample, _ = self._sample_gauss_logvar(post_mean, post_log_var)        
        z_sample, _ = self._sample_gauss_logvar(z_mean, z_log_var)
        
        recon_x = self.decoder(post_sample)["reconstruction"]
        if self.model_config.orig_recon_loss:
            obs_loglik_post = -torch.mean((recon_x - x)**2, dim=self.dims_to_reduce)    
        else:
            obs_loglik_post = -F.l1_loss(
                                recon_x.reshape(x.shape[0], -1),
                                x.reshape(x.shape[0], -1),
                                reduction="none",
                            ).sum(dim=-1)
        kl_post_prior = kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
        elbo_iVAE = obs_loglik_post - self.kl_annealing_coeff*self.beta*kl_post_prior
                
        if self.method == 'iVAE':
            loss = -elbo_iVAE
        elif self.method == 'IDVAE':
            u_sample, _ = self._sample_gauss_logvar(lam_mean, lam_log_var)
            recon_u = self.label_decoder(u_sample)
            obs_loglik_cond = -torch.mean((recon_u - x_pheno)**2, dim=[1])
            kl_cond = kl_criterion(lam_mean, lam_log_var,
                                    torch.zeros_like(lam_mean),
                                    torch.ones_like(lam_log_var))
            elbo_cond = obs_loglik_cond - self.kl_annealing_coeff*self.beta*kl_cond
            loss = -elbo_iVAE-elbo_cond
        elif self.method == 'CI-iVAE':
            recon_data_encoded = self.decoder(z_sample)["reconstruction"]
            if self.model_config.orig_recon_loss:
                obs_loglik_encoded = -torch.mean((recon_data_encoded - x)**2, dim=self.dims_to_reduce)
            else:
                obs_loglik_encoded = -F.l1_loss(
                                        recon_data_encoded.reshape(x.shape[0], -1),
                                        x.reshape(x.shape[0], -1),
                                        reduction="none",
                                    ).sum(dim=-1)
            kl_encoded_prior = kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
            elbo_VAE_with_label_prior = obs_loglik_encoded - self.kl_annealing_coeff*self.beta*kl_encoded_prior
            
            epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], self.model_config.M)).to(x.device)
            z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, self.model_config.M])
            z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, self.model_config.M])
            z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

            epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], self.model_config.M)).to(x.device)
            post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, self.model_config.M])
            post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, self.model_config.M])
            post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon
            
            log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
            log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
            
            alpha_list = self.alpha_step + np.arange(0.0, 1.0-self.alpha_step,
                                                        self.alpha_step)
            loss = torch.zeros((elbo_iVAE.shape[0], len(alpha_list)+2))
            loss[:, 0], loss[:, -1] = -elbo_iVAE, -elbo_VAE_with_label_prior
            i = 1
            for alpha in alpha_list:
                ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                skew_kl_iVAE = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                skew_kl_iVAE = torch.abs(torch.mean(skew_kl_iVAE, dim=-1))
                skew_kl_VAE_with_label_prior = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                skew_kl_VAE_with_label_prior = torch.abs(torch.mean(skew_kl_VAE_with_label_prior, dim=-1))
                loss[:, i] = -alpha*elbo_VAE_with_label_prior-(1.0-alpha)*elbo_iVAE+alpha*skew_kl_VAE_with_label_prior+(1.0-alpha)*skew_kl_iVAE
                i += 1
            del(i)
            loss, _ = torch.min(loss, dim = 1)
        
        loss = torch.mean(loss)
            
        out = ModelOutput(
            loss=loss,
            loss_elbo_iVAE = torch.mean(elbo_iVAE),
            recon_x=recon_x,
            z=z_sample,
            post_sample=post_sample,
        )

        if self.method == 'IDVAE':
            out.loss_kl_cond = torch.mean(kl_cond)
            out.pheno_pred = recon_u
        elif self.method == 'CI-iVAE':
            out.recon_data_encoded = recon_data_encoded
            out.loss_elbo_VAE_with_label_prior = torch.mean(elbo_VAE_with_label_prior)

        return out
    
    def _sample_gauss_logvar(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return self._sample_gauss(mu, std)
        
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
        raise "Not implemented yet!"

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
        if 'phenotypes' in batch:
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
        if not self.model_config.predict_reparameterised:
            z = self.encoder(inputs).embedding
            recon_x = self.decoder(z)["reconstruction"]

            return PlasmaConduit(
                recon_x=recon_x,
                embedding=z,
            )
        else:
            encoder_output = self.encoder(inputs)
            z_mean, z_log_var = encoder_output.embedding, encoder_output.log_covariance
            z_sample, _ = self._sample_gauss_logvar(z_mean, z_log_var)
            recon_x = self.decoder(z_sample)["reconstruction"]

            return PlasmaConduit(
                recon_x=recon_x,
                embedding=z_sample,
            )

    #The following three functions are from the original pipeline. But not considered yet. So, they aren't also implemented according to our pipeline.
    
    # def make_plots(self):
    #     if self.dataset in ['EMNIST', 'FashionMNIST']:
    #         os.makedirs(os.path.join(self.save_dir, 'figures', f'epoch_{self.epoch+1:03d}'))
    #         self.set_mu_sig()
    #         sig_rms = np.sqrt(np.mean((self.decoder.sig**2).detach().cpu().numpy(), axis=0))
    #         plot_samples(self.decoder, n_rows=20, dataset=self.dataset, method=self.method)
    #         plot_spectrum(self, sig_rms, dims_to_plot=self.dim_z, dataset=self.dataset)
    #         n_dims_to_plot = 40
    #         top_sig_dims = np.flip(np.argsort(sig_rms))
    #         dims_to_plot = top_sig_dims[:n_dims_to_plot]
    #         plot_variation_along_dims(self.decoder, dims_to_plot, method=self.method)
    #         plot_t_sne(self, dataset=self.dataset)
    #     else:
    #         raise RuntimeError("Check dataset name. Doesn't match.")
    
    # def set_mu_sig(self, n_batches=40):        
    #     examples = iter(self.test_loader)
    #     n_batches = min(n_batches, len(examples))
    #     self.lam_mean, self.lam_log_var, self.z_mean, self.z_log_var, self.target = [], [], [], [], []
    #     for _ in range(n_batches):
    #         data, targ = next(examples)
    #         data += torch.randn_like(data)*1e-2
    #         onehot_targ = nn.functional.one_hot(targ, num_classes=self.dim_u)
    #         onehot_targ = torch.tensor(onehot_targ, dtype=torch.float32, device=self.device)
    #         self.eval()
            
    #         lam_mean, lam_log_var = self.label_prior(onehot_targ)
    #         z_mean, z_log_var = self.encoder(data.to(self.device))
    #         lam_mean, lam_log_var, z_mean, z_log_var = lam_mean.detach().cpu().numpy(), lam_log_var.detach().cpu().numpy(), z_mean.detach().cpu().numpy(), z_log_var.detach().cpu().numpy()
            
    #         self.lam_mean.append(lam_mean)
    #         self.lam_log_var.append(lam_log_var)
    #         self.z_mean.append(z_mean)
    #         self.z_log_var.append(z_log_var)
    #         self.target.append(targ)
        
    #     self.lam_mean = np.concatenate(self.lam_mean, 0)
    #     self.lam_log_var = np.concatenate(self.lam_log_var, 0)
    #     self.z_mean = np.concatenate(self.z_mean, 0)
    #     self.z_log_var = np.concatenate(self.z_log_var, 0)
    #     self.target = np.concatenate(self.target, 0)
        
    #     self.decoder.mu = torch.tensor([self.lam_mean[self.target == i].mean(0) for i in range(self.dim_u)]).to(self.device)
    #     self.decoder.sig = torch.tensor([np.exp(0.5*self.lam_log_var[self.target == i]).mean(0) for i in range(self.dim_u)]).to(self.device)
        
    #     sst = np.sum((self.z_mean - np.mean(self.z_mean, axis=0))**2)
    #     ssw = np.sum([np.sum((self.z_mean[self.target==i]-np.mean(self.z_mean[self.target==i], axis=0))**2) for i in range(self.dim_u)])
    #     print('ssw/sst: %.3f' % (ssw/sst))
    #     self.ssw_over_sst = ssw/sst
    
    # def calculate_knn_accuracy(self, k=5, test=True):
    #     model_dir = './results/%s/%s/%d' % (self.method, self.dataset, self.seed)
    #     model_dir = os.path.join(model_dir, os.listdir(model_dir)[0], 'model_save')
    #     model_dir = os.path.join(model_dir, np.sort(os.listdir(model_dir))[-1])
    #     self.load(model_dir)
        
    #     latent_train = []; latent_test = []
    #     target_train = []; target_test = []
    #     for batch_idx, (data, target) in enumerate(self.train_loader):
    #         data = data.to(self.device)
    #         target = nn.functional.one_hot(target, num_classes=self.dim_u)
    #         target = torch.tensor(target, dtype=torch.float32, device=self.device)

    #         z_mean, _ = self.encoder(data)
    #         latent_train.append(z_mean.cpu().detach().numpy())
    #         target_train.append(target.cpu().numpy())
    #     latent_train = np.concatenate(latent_train)
    #     target_train = np.concatenate(target_train)
            
    #     for batch_idx, (data, target) in enumerate(self.test_loader):
    #         data = data.to(self.device)
    #         target = nn.functional.one_hot(target, num_classes=self.dim_u)
    #         target = torch.tensor(target, dtype=torch.float32, device=self.device)

    #         z_mean, _ = self.encoder(data)
    #         latent_test.append(z_mean.cpu().detach().numpy())
    #         target_test.append(target.cpu().numpy())
    #     latent_test = np.concatenate(latent_test)
    #     target_test = np.concatenate(target_test)
        
    #     if test:
    #         neigh = KNeighborsClassifier(n_neighbors=k)
    #         neigh.fit(latent_train, target_train)
            
    #         target_pred = np.array(np.argmax(neigh.predict(latent_test), axis=1))
    #         target_test = np.array(np.argmax(target_test, axis=1))
    #         accuracy = np.mean(np.equal(target_pred, target_test))
    #     else:
    #         neigh = KNeighborsClassifier(n_neighbors=k+1)
    #         neigh.fit(latent_train, target_train)
    #         neighbor_indices = neigh.kneighbors(latent_train, return_distance=False)
    #         neighbor_indices = neighbor_indices[:, 1:]
            
    #         target_train = np.array(np.argmax(target_train, axis=1))
    #         target_pred = mode(target_train[neighbor_indices], axis=1)[0][:, 0]
    #         accuracy = np.mean(np.equal(target_pred, target_train))
    #     print('accuracy: ', accuracy)