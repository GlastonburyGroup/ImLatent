#This package is adapated from the code of the paper "Covariate-informed Representation Learning to Prevent Posterior Collapse of iVAE"
#From the GitHub repo: https://github.com/kyg0910/CI-iVAE/
#following: https://github.com/kyg0910/CI-iVAE/tree/main/experiments/EMNIST_and_FashionMNIST
#taking only the VAE part (dropping the GIN part as we are not interested in that)

#The code makes it possible to use three models: iVAE, IDVAE, and the proposed CI-iVAE
#Original papers of the models:
#iVAE (Identifiable VAE): https://arxiv.org/abs/1907.04809
#IDVAE (Identifiable Double VAE): https://arxiv.org/abs/2010.09360

#For now, aggressive posterior and KL annealing (both present in the original repo) are not implemented - for simplicity, and also because we are not interested (in my opinion)
#Both these were implmeneted on top of the original iVAE model as baselines to compare with the CI-iVAE model. But not part of any of the proposed methods mentioned earlier.

#TODO: LR scheduler (maybe)

from .ivaes_config import iVAEsConfig
from .ivaes_model import iVAEs

__all__ = ["iVAEs", "iVAEsConfig"]