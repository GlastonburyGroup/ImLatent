import argparse
import contextlib
import math
import os
import sys
if os.name == "posix":
    import resource
import logging
import psutil
import math

sys.path.insert(0, os.getcwd()) #to handle the sub-foldered structure of the "Executors" folder

import torch
from Engineering.Engines.MainEngine import Engine
from Engineering.constants import *
from Engineering.utilities import get_SLURM_envs
from lightning.pytorch import seed_everything

with contextlib.suppress(Exception):
    torch.multiprocessing.set_start_method("fork")
seed_everything(1701, workers=True)

def getARGSParser():
    parser = argparse.ArgumentParser()

    #Basic params
    parser.add_argument('--taskID', action="store", type=int, default=1, help="0: Reconstruction-based latent extraction using AEs and VAEs, 1: Latent extraction using diffusion autoencoder, 2: Latent classification models (utilised for latent manipulation for DiffAE)")
    parser.add_argument('--trainID', action="store", default="prova3D_diffAE", help="The name of the training session. run_prefix (from datainfo json) and foldID will be added as prefix and suffix")
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False, help="To resume training from the last checkpoint")
    parser.add_argument('--load_best', action=argparse.BooleanOptionalAction, default=False, help="To resume training from the last checkpoint")
    parser.add_argument('--load_model_4test', action="store", default="", help="To load checkpoint for testing, a speicifc one which has nothing to do with the current run name")
    parser.add_argument('--num_workers', action="store", default=8, type=int, help="Number of workers for dataloaders. Each (train, validation, test) will get this value.")
    parser.add_argument('--use_concurrent_dataloader', action=argparse.BooleanOptionalAction, default=False, help="Whether to use concurrent dataloader")
    parser.add_argument('--batch_size', action="store", default=2, type=int, help="Actual batch size for the model. This can be modified as required, just control the effective_batch_size for comparibility")
    parser.add_argument('--effective_batch_size', action="store", default=2, type=int, help="To make runs with different BS comparible, keep this constant. accumulate_gradbatch would be this divided by BS. (Original paper: same as batch size)")
    parser.add_argument('--auto_optim', action=argparse.BooleanOptionalAction, default=True, help="Whether to use Lightning's auto-optimisation or not")
    parser.add_argument('--n_train_batches', action="store", default=0.25, type=float, help="How much of training dataset to use, can be float between 0.0 and 1.0, or an integer (float = fraction, int = num_batches)")
    parser.add_argument('--n_val_batches', action="store", default=0.025, type=float, help="How much of validation dataset to use, can be float between 0.0 and 1.0, or an integer (float = fraction, int = num_batches)")
    parser.add_argument('--nbatch_prefetch', action="store", default=20, type=int, help=" The number of batches to prefetch for the dataloaders by each worker.")
    parser.add_argument('--configyml_path', action="store", default="Configs/config_v1.yaml", help="Relative path, from the project root, to the configuration yaml file")
    parser.add_argument('--datajson_path', action="store", default="Configs/UKBB208/dataNpath_F20208_time2slc_4ChTransverse_Crop128.json", help="Relative path, from the project root, to the dataNpath json file, containing the paths and other data related info")
    parser.add_argument('--foldID', action="store", type=int, default=0, help="Which fold ID to use, from the foldCSV specified in the datainfo json")
    parser.add_argument("--output_suffix", action="store", default="", help="Suffix to add to the output folder name (useful for multiple tests with same config)")
    parser.add_argument('--accelerator', action="store", default="gpu", help="gpu (default), cpu, tpu, ipu, hpu, mps, auto, custom (for custom accelerator - not implemented yet)")
    parser.add_argument('--ampmode', action="store", default="16-mixed", help="The precision to use. 16-mixed is default, as per original DiffAE. 32 or 32-true: full precision, 16/16-mixed: half precision (noAMP/AMP), bf16/bf16-mixed: bfloat16 (noAMP/AMP). 64 is also possible, but why to use?")
    parser.add_argument('--matmul_precision', action="store", default="highest", help="Running float32 matrix multiplications in lower precision [will only be considered if bfloat16 is available]. highest (default): float32 matrix multiplications, high: TensorFloat32 or bfloat16_3x datatypes, medium: bfloat16 ")
    parser.add_argument('--run_mode', action="store", default=4, type=int, help='0: Train, 1: Train and Validate, 2:Test, 3: Train followed by Test, 4: Train and Validate followed by Test')
    parser.add_argument('--dev_run', action="store", default=0, type=int, help='0: Train normally. 1 to n: runs only n training and n validation batch and the program ends. Set it to negative value to make it run in barebone mode.')
    parser.add_argument('--non_deter', action=argparse.BooleanOptionalAction, default=False, help="Whether to use non-deterministic algorithms or not. Usually, never!")
    
    #This is following the original work of DiffAE. If any of the paramters (except for the prenorm) is modified, code of the DiffEngine needs to be modified accordingly
    parser.add_argument('--norm_type', action="store", default="zscore", help='Currently "zscore" normalisation and 2 (min-)max modes and their volumetric versions are supported. minmax, divbymax. Volumetric versions: minmaxvol, divbymaxvol')
    parser.add_argument('--zscore_mean', action="store", default="0.5", help="Coma-seperated channel-wise mean values (to be used for zscore normalisation)")
    parser.add_argument('--zscore_std', action="store", default="0.5", help="Coma-seperated channel-wise std values (to be used for zscore normalisation)")
    parser.add_argument('--zscore_prenorm', action="store", default="minmaxvol", help="Integer or float value, to be used as pre-normalisation for zscore normalisation (typically, 255 for 8-bit images). Can also be a string, containing one of the possible values of norm_type (except zscore)")
    
    #DiffAE specific
    parser.add_argument("--test_ema", action=argparse.BooleanOptionalAction, default=True, help="Whether to use EMA model for testing or the base model")
    parser.add_argument("--test_with_TEval", action=argparse.BooleanOptionalAction, default=True, help="Typically, test should be performed with the default values of T_inv and T_step (200 and 100, respectively), instead of the default value of T_eval (20). But only effects the recon quality, not the latents. However, slows the execution down a lot")
    parser.add_argument("--test_emb_only", action=argparse.BooleanOptionalAction, default=False, help="To make the execution faster, we can skip completely the recon steps.")

    #Loading from Hugging Face
    parser.add_argument('--load_hf', action="store", default="", help="Model tag from Hugging Face to load a pre-trained model. Keep it blank if not desired. Please note that the model should be compatible with the specified taskID and modelID.")

    #Multi-GPU/Node and training strategy params
    parser.add_argument('--training_strategy', action="store", default="", help=f'Choose a custom training strategy. Set to "default" or blank for the default training strategy. Other options: {str(STRATEGIES)}')
    parser.add_argument('--ndevices', action="store", default=0, type=int, help='Use multi-GPU (or even TPU) for execution. Set it to 2 or more for multi-GPU execution. (Number of GPUs per node during multi-node execution)')
    parser.add_argument('--nnodes', action="store", default=1, type=int, help='Use multi-Node for execution. Set it to 2 or more for multi-node execution. (Set it to 1 for single-node or non-cluster execution)')
    parser.add_argument('--sync_batchnorm', action=argparse.BooleanOptionalAction, default=False, help='Sync batchnorms across all devices. Only works with multi-GPU execution. (Default: False)')

    #Training params
    parser.add_argument('--num_epochs', action="store", default=200, type=int, help="Total number of epochs. If resuming, then it will continue till a total number of epochs set by this.")
    parser.add_argument('--lr', action="store", default=0.0001, type=float)
    parser.add_argument('--lossID', action="store", default=0, type=int, help=f"Loss IDs: {str(LOSSID)}")
    parser.add_argument('--optimiserID', action="store", default=0, help=f"Optimiser IDs: {str(OPTIMID)}")
    parser.add_argument('--n_optimisers', action="store", default=-1, type=int, help="Number of optimisers to use (Default: -1). If it's set to 0 or -1, the number of optimiser is determined by the length of the model_optims array that must be present inside the selected Warp Drive (Model)")
    parser.add_argument('--ploss_model', action="store", default="med1ch3D.UNetMSSDS6", help="Model to use for the perceptual losses. -1: traditional perceptual loss, -2: LPIPS")
    parser.add_argument('--ploss_level', action="store", default=math.inf, type=int)
    parser.add_argument('--ploss_type', action="store", default="L1")
    parser.add_argument('--grad_clip_algo', action="store", default="norm", help="Which gradient clipping algorithm to use: value or norm. Leave it blank if not desired")
    parser.add_argument('--grad_clip_val', action="store", default=1, type=float, help="The value to clip the gradients to, using the specified algo. Only used if grad_clip_algo is not blank")
    parser.add_argument('--lr_decay_type', action="store", default=0, type=int, help='0: No Decay, 1: StepLR, 2: ReduceLROnPlateau')
    parser.add_argument('--interp_fact', type=float, default=1, help="If not set to 1, the input will be interpolated to the given factor before sending it to the CropOrPad function (if enabled)")
    parser.add_argument('--input_shape', action="store", default="50,128,128", help="length, width, depth (to be used if patch_size is not given)")
    parser.add_argument('--croppad', action=argparse.BooleanOptionalAction, default=True, help="If True, then it will crop or pad the volume/slice to the given input_shape")
    parser.add_argument('--n_save_recon_subs', action="store", default=13, type=float, help="How much of testing dataset to be used for saving reconstructions (for the rest, only embeddings will be storred), can be float between 0.0 and 1.0, or an integer (float = fraction, int = num_batches)")
    parser.add_argument('--save_recon_subs', action="store", default="", help="If supplied, will override the n_save_recon_subs parameter. A comma separated list of subject IDs to be used for saving reconstructions (for the rest, only embeddings will be storred)")
    
    #Data augmentation params
    parser.add_argument('--p_augment', action="store", default=1, type=float, help="Probability of applying augmentation")
    parser.add_argument('--p_aug_horflip', action="store", default=0.5, type=float, help="Probability of applying Random Horizontal Flip augmentation")
    
    #Additional data param
    parser.add_argument('--grey2RGB', action="store", default=-1, type=float, help="-1 (default): disabled, 0: data in central channel and other two only zeros, 1: repeat the greyscale data")
    
    #Model Params
    parser.add_argument('--modelID', action="store", default=0, type=int, help=f"Model IDs: {str(MODELID)}")
    parser.add_argument('--pythae_model', action="store", default="factor_vae", help=f"pythae: Model name for the pythae package, from the list: {str(PYTHAEMODELID)} (Only used if modelID is 0)")
    parser.add_argument('--pythae_wrapper_mode', action="store", default=0, type=int, help="Model Wrapper mode: 0: No Wrapper, 1: LSTM Wrapped Model")
    parser.add_argument('--pythae_config', action="store", default="", help="pythae: Model config json file for the pythae pacakge. If blank, default one will be used. (Only used if modelID is 0)")
    parser.add_argument('--preweights_path', action="store", default="", help="Checkpoint path for pre-loading weights before starting the pipeline (Can be used for transfer learning)")
    parser.add_argument('--is3D', action="store", default=1, type=int, help="Is it a 3D model?")
    parser.add_argument('--in_channels', action="store", default=1, type=int, help="Number of input channels")
    parser.add_argument('--out_channels', action="store", default=1, type=int, help="Number of input channels (Should be identical to in_channels, but still possible to change)")
    parser.add_argument('--complie_model', action=argparse.BooleanOptionalAction, default=False, help="Apply torch.compile on the model for speed-up (requires PyTorch 2.0+)")
    parser.add_argument('--check_anomalies', action=argparse.BooleanOptionalAction, default=False, help="Check for anomalies, like nan loss during training (Disable for speedup)")

    #Model tunes with lightning
    parser.add_argument('--auto_bs', action=argparse.BooleanOptionalAction, default=False, help="Automatically find the batch size to fit best")
    parser.add_argument('--auto_lr', action=argparse.BooleanOptionalAction, default=False, help="Automatically find the LR")
    parser.add_argument('--profiler', action="store", default="", help="Whether to profile the steps during training to adentify bottlenecks. Blank for no profiling, or 'simple' for simple profiling, or 'advanced' for advanced profiling")

        
    #Logging params    
    parser.add_argument("-tba", "--tbactive", action=argparse.BooleanOptionalAction, default=True, help="Use Tensorboard")
    parser.add_argument("-wnba", "--wnbactive", action=argparse.BooleanOptionalAction, default=True, help="Use WandB")
    parser.add_argument("-wnbp", "--wnbproject", help="WandB: Name of the project")
    parser.add_argument("-wnbe", "--wnbentity", help="WandB: Name of the entity")
    parser.add_argument("-wnbg", "--wnbgroup", help="WandB: Name of the group")
    parser.add_argument("-wnbpf", "--wnbprefix", default='', help="WandB: Prefix for TrainID")
    parser.add_argument("-wnbml", "--wnbmodellog", default=None, help="WandB: While watching the model, what to save: gradients, parameters, all, None")
    parser.add_argument("-wnbmf", "--wnbmodelfreq", type=int, default=100, help="WandB: The number of steps between logging gradients")

    return parser

def helm(sys_params=None):
    #Set basic limits: CPU and RAM
    if sys_params is not None:
        if "cpus_avail" in sys_params:
            torch.set_num_threads(max(1, sys_params["cpus_avail"] // 2))
        if "mem_limit" in sys_params:
            if os.name == "posix":
                resource.setrlimit(resource.RLIMIT_AS, (sys_params["mem_limit"], sys_params["mem_limit"]))
            else:
                print("WARNING: Memory limit cannot be set for non-POSIX OS (Windows)!")
    
    parser = getARGSParser()
    engine = Engine(parser=parser, sys_params=sys_params)
    if engine.hparams.auto_bs or engine.hparams.auto_lr:
        logging.debug("Executor: Engine alignment initiating...")
        engine.align()
        logging.debug("Executor: Engine alignment finished!")
    engine.engage()

# @profile
def main():
    sys_params = {}
    if 'SLURM_JOB_ID' in os.environ:
        print("We are in the SLURM space!")
        sys_params["slurmENV"] = get_SLURM_envs()
        sys_params["mem_limit"] = sys_params["slurmENV"]['COMPUTED_MEM_PER_TASK'] * 1024 * 1024  # Converts to bytes (TODO: check if using COMPUTED_MEM_PER_NODE is better suited, but I don't think so - Soumick)
        sys_params["cpus_avail"] = int(sys_params["slurmENV"]['SLURM_CPUS_PER_TASK']) #TODO: check if it's better to use SLURM_JOB_CPUS_PER_NODE or SLURM_CPUS_ON_NODE instead
    else:
        sys_params["mem_limit"] = math.floor(0.9 * psutil.virtual_memory().available) #Use up to 80% of the available memory (TODO: make this a parameter, maybe?)
        sys_params["cpus_avail"] = math.floor(0.9 * psutil.cpu_count()) #Maybe use only the number of physical cores? Then use 0.8 * psutil.cpu_count(logical=False)
    helm(sys_params=sys_params)

if __name__ == '__main__':
    main()
