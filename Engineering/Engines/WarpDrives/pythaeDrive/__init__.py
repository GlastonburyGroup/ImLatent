# This package acts as a wrapper for the pythae package from the following paper presented at NeurIPS 2022: https://openreview.net/forum?id=w7VPQWgnn3s
# The code is available at: https://github.com/clementchadebec/benchmark_VAE

# Choices updated on 20230822
PYTHAEMODELID = {
    "ae": "Autoencoder (AE)",
    "vae": "Variational Autoencoder (VAE)",
    "beta_vae": "Beta Variational Autoencoder (BetaVAE)",
    "iwae": "Importance Weighted Autoencoder (IWAE)",
    "wae": "Wasserstein Autoencoder (WAE)",
    "info_vae": "Info Variational Autoencoder (INFOVAE_MMD)",
    "rae_gp": "Regularized AE with gradient penalty (RAE_GP)",
    "rae_l2": "Regularized AE with L2 decoder param (RAE_L2)",
    "vamp": "VAMP Autoencoder (VAMP)",
    "hvae": "Hamiltonian VAE (HVAE)",
    "rhvae": "Riemannian Hamiltonian VAE (RHVAE)",
    "aae": "Adversarial Autoencoder (Adversarial_AE)",
    "vaegan": "Variational Autoencoder GAN (VAEGAN)",
    "vqvae": "Vector Quantized VAE (VQVAE)",
    "msssim_vae": "VAE with perceptual metric similarity (MSSSIM_VAE)",
    "svae": "Hyperspherical VAE (SVAE)",
    "disentangled_beta_vae": "Disentangled Beta Variational Autoencoder (DisentangledBetaVAE)",
    "factor_vae": "Disentangling by Factorising (FactorVAE)	",
    "beta_tc_vae": "Beta-TC-VAE (BetaTCVAE)	",
    "vae_iaf": "VAE with Inverse Autoregressive Flows (VAE_IAF)",
    "vae_lin_nf": "VAE with Linear Normalizing Flows (VAE_LinNF)",

    #The one's which are not implemented in the examples of the package
    "miwae": "Multiply Importance Weighted Autoencoder (MIWAE)",
    "piwae": "Partially Importance Weighted Autoencoder (PIWAE)",
    "ciwae": "Combination Importance Weighted Autoencoder (CIWAE)",
    "pvae": "Poincaré Disk VAE (PoincareVAE)",

    "ultra_factor_vae": "Extended FactorVAE for Phenotypes and Confounders (UltraFactorVAE)",
    "ultra_vae": "Extended VAE for Phenotypes and Confounders (UltraVAE)",
    "ultra_cevae": "Context-encoding VAE from StRegA (CE-VAE)",
    
    "complex_factor_vae": "Complex-valued FactorVAE (CV-FactorVAE)",
    "complex_vae": "Complex-valued Vanila VAE (CV-VAE)",
    "complex_ae": "Complex-valued Vanila AE (CV-AE)",
    
    "ivaes": "Identifiable VAE Models (iVAEs): Baseline iVAE, IDVAE, and CI-iVAE",
}

# Model classes and the corresponding config classes - to be used for importing
PYTHAEMODELCLASS = {
    "ae": "AE, AEConfig",
    "vae": "VAE, VAEConfig",
    "beta_vae": "BetaVAE, BetaVAEConfig",
    "iwae": "IWAE, IWAEConfig",
    "wae": "WAE_MMD, WAE_MMD_Config",
    "info_vae": "INFOVAE_MMD, INFOVAE_MMD_Config",
    "rae_gp": "RAE_GP, RAE_GP_Config",
    "rae_l2": "RAE_L2, RAE_L2_Config",
    "vamp": "VAMP, VAMPConfig",
    "hvae": "HVAE, HVAEConfig",
    "rhvae": "RHVAE, RHVAEConfig",
    "aae": "Adversarial_AE, Adversarial_AE_Config",
    "vaegan": "VAEGAN, VAEGANConfig",
    "vqvae": "VQVAE, VQVAEConfig",
    "msssim_vae": "MSSSIM_VAE, MSSSIM_VAEConfig",
    "svae": "SVAE, SVAEConfig",
    "disentangled_beta_vae": "DisentangledBetaVAE, DisentangledBetaVAEConfig",
    "factor_vae": "FactorVAE, FactorVAEConfig",
    "beta_tc_vae": "BetaTCVAE, BetaTCVAEConfig",
    "vae_iaf": "VAE_IAF, VAE_IAF_Config",
    "vae_lin_nf": "VAE_LinNF, VAE_LinNF_Config",
    "miwae": "MIWAE, MIWAEConfig",
    "piwae": "PIWAE, PIWAEConfig",
    "ciwae": "CIWAE, CIWAEConfig",
    "pvae": "PoincareVAE, PoincareVAEConfig",

    "ultra_factor_vae": "UltraFactorVAE, UltraFactorVAEConfig",
    "ultra_vae": "UltraVAE, UltraVAEConfig",
    "ultra_cevae": "UltraCEVAE, UltraCEVAEConfig",
    
    "ultra_cevae": "UltraCEVAE, UltraCEVAEConfig",
    "complex_factor_vae": "CVFactorVAE, CVFactorVAEConfig",
    "complex_vae": "CVVAE, CVVAEConfig",
    "complex_ae": "CVAE, CVAEConfig",
    
    "ivaes": "iVAEs, iVAEsConfig",
}