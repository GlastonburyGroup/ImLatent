# It is an extension to the existing benchmark NNs available in the repo: https://github.com/clementchadebec/benchmark_VAE/tree/main/src/pythae/models/nn/benchmarks/
# In theory, this should beccome the 4th of the benchmarks. Waiting for the author's response over email and then will make a PR to the repo.

"""A collection of Neural nets used to perform the benchmark on CELEBA"""

from .convnets import *
from .cconvnets import *
from .resnets import *
from .cresnets import *

__all__ = [
    "Encoder_Conv_AE_MED",
    "Encoder_Conv_VAE_MED",
    "Encoder_Conv_SVAE_MED",
    "Decoder_Conv_AE_MED",
    "Discriminator_Conv_MED",
    "Encoder_ResNet_AE_MED",
    "Encoder_ResNet_VAE_MED",
    "Encoder_ResNet_SVAE_MED",
    "Encoder_ResNet_VQVAE_MED",
    "Decoder_ResNet_AE_MED",
    "Decoder_ResNet_VQVAE_MED",
    
    "Encoder_CVConv_AE_MED",
    "Encoder_CVConv_VAE_MED",
    "Encoder_CVConv_SVAE_MED",
    "Decoder_CVConv_AE_MED",
    "Discriminator_CVConv_MED",
    "Encoder_CVResNet_AE_MED",
    "Encoder_CVResNet_VAE_MED",
    "Encoder_CVResNet_SVAE_MED",
    "Encoder_CVResNet_VQVAE_MED",
    "Decoder_CVResNet_AE_MED",
    "Decoder_CVResNet_VQVAE_MED",
]
