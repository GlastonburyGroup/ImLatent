# Unsupervised latent representation learning using 2D and 3D diffusion and other autoencoders

This repository provides a pipeline for training and evaluating 2D and 3D diffusion autoencoders, traditional autoencoders, and various variational autoencoders for unsupervised latent representation learning from 2D and 3D images, primarily focusing on MRIs. This repository was developed as part of the paper titled [Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/) and was utilised to learn and infer latent representations from cardiac MRIs (CINE) using a 3D diffusion autoencoder.

- [Unsupervised latent representation learning using 2D and 3D diffusion and other autoencoders](#unsupervised-latent-representation-learning-using-2d-and-3d-diffusion-and-other-autoencoders)
  - [Pipeline](#pipeline)
    - [Structure](#structure)
    - [Executing the pipeline](#executing-the-pipeline)
      - [Running inference on a trained model](#running-inference-on-a-trained-model)
  - [Dataset](#dataset)
    - [Data Dimensions](#data-dimensions)
  - [Trained Weights from Hugging Face](#trained-weights-from-hugging-face)
  - [Citation](#citation)
  - [Credits](#credits)
    - [DiffAE: Diffusion Autoencoder](#diffae-diffusion-autoencoder)
    - [pythae: Unifying Generative Autoencoders in Python](#pythae-unifying-generative-autoencoders-in-python)


## Pipeline 

### Structure
**Inside the _Executors_ package**, you will find the actual execution scripts. For different problems, the relevant *main* files can be placed here. Sub-packages may also be created within this package to better organise different experiments.

- **_main*.py_:** This is the main script used to run the pipeline. It contains all the default values for various command-line arguments, which can be supplied when invoking this script.  

Currently, three distinct main files are available:  
1. `main_recon.py`: For 2D and 3D autoencoders (including VAEs but excluding diffusion autoencoders).  
2. `main_diffAE.py`: For 2D and 3D diffusion autoencoders.  
3. `main_latentcls.py`: For training a classifier on the latent space.  

**Inside the *Configs* folder,** there are two types of files required by the *main* scripts. Sub-folder structures can be created within this folder to better organise different experiments.  

- **_config*.yaml_:** These files contain configuration parameters, typically specific to different aspects of the pipeline (e.g., learning rate scheduler parameters). As these parameters are less likely to change frequently, they are defined here in a hierarchical format. However, these values can also be overridden using command-line arguments, which will be discussed later.  
- **_dataNpath*.json_:** As the name suggests, these files include dataset-specific parameters, such as the name of the foldCSV file and the run prefix. They also define necessary paths, such as the data directory (where the `data.h5` file, the supplied CSV for folds, and the processed dataset files are stored).

**Inside the *Engineering* package,** all the files related to the actual implementation of the pipeline are stored. 


### Executing the pipeline

A conda environment can be created using the provided `environment.yml` file. 

To execute, the call should be from the root directory of the pipeline. For example:
```python
    python Executors/main_recon.py --batch_size 32 --lr 0.0001 --training§LRDecay§type1§decay_rate 0.15 --training§prova testing --json§save_path /myres/toysets/Results
```

Instead of create a conda environment, [Poetry](https://python-poetry.org/) can also be used. . Once Poetry is installed, this pipeline can be launched from its root directory without manually installing any dependencies manually or using the yml file by adding `poetry run` before calling python. For example:
```python
    poetry run python Executors/main_recon.py --batch_size 32 --lr 0.0001 --training§LRDecay§type1§decay_rate 0.15 --training§prova testing --json§save_path /myres/toysets/Results
```
For continuous access in the terminal without adding the `poetry run` prefix to all commands, `poetry shell` (It must be installed additionally: https://github.com/python-poetry/poetry-plugin-shell.git) can be executed to activate the environment. The other Python commands can then be executed normally.

Here, _main_recon.py_ is the main file that is to be executed, _batch_size_ and _lr_ are arguments which will be replacing the default values for those parameters supplied within that main file, _training§LRDecay§type1§decay_rate_ and _training§prova_ (arguments not specified inside the main file or any of the Engines) are going to override the values present inside the config.yaml file mentioned inside the main file (or supplied as a command line argument) - following the path splitting the key with dollars, and finally, _json§save_path_ (same as the earlier one, but arguments starting with _json§_) will override the _save_path_ value of the parameter inside the _dataNpath.json_ specified inside the main file (or supplied as a command line argument).

Please note:  
_training§LRDecay§type1§decay_rate_ will try to find the dictonary path _training/LRDecay/type1/decay_rate_ inside the yaml file. If the path is found, the value will be updated (for the current run only) with the supplied one. If it's not found, like in this example training§prova, a new path will be created, and the value will be added (for the current run only). Any command line argument that is not found inside the main file, or any of the Engines or Warp Drives, will be treated as an "unknown" parameter and will be treated in this manner - unless they start with "_json§". In that case, it is used to update the value of _save_path_ present inside the _dataNpath.json_ for the current run. 


For complete list of command line arguments, please refer to the main files or execute the main file with the `--help` flag. For example:
```python
    python Executors/main_diffAE.py --help
```
and check the files inside *Configs* folder.


#### Running inference on a trained model
Add the following command line arguments to the ones used during training:
```python
    --resume --load_best --run_mode 2
```
To run inference of a model currently in training (i.e. interm inference), add the following:
```python
    --resume --load_best --run_mode 2 --output_suffix XYZ
```
where XYZ is a suffix that would be added to the Output directory. 
To run inference on the whole dataset (ignoring the splits), add the following:
```python
    --resume --load_best --run_mode 2 --output_suffix fullDS --json§foldCSV 0
```
(can also change fullDS to something else)

## Dataset 
This pipeline expects an HDF5 file containing the dataset as input, following the structure described below.

The groups should follow the path:
```python
patientID/fieldID/instanceID
```
Groups may (optionally) include the following attributes: DICOM Patient ID, DICOM Study ID, description of the study, AET (model of the scanner), Host, date of the study, number of series in the study:
```python
    patientDICOMID, studyDICOMID, studyDesc, aet, host, date, n_series
```
Each series present is stored as a separate dataset, and the key for the datasets can be:

- **`primary`:** To store the primary data (i.e., the series description matches one of the values of the `primary_data` attribute).
- **`primary_*`:** (Optional) If the `multi_primary` attribute is set to `True`, then instead of a single `primary` key, there will be multiple keys in the form `primary_*`, where `*` is replaced by the corresponding tag supplied using the `primary_data_tags` attribute.
- **`auxiliary_*`:** (Optional) To store auxiliary data (e.g., T1 maps in the case of ShMoLLI). Here, `*` is replaced by the corresponding tag supplied using the `auxiliary_data_tags` attribute.

Additional type information may be appended to the dataset key:

1. **`_sagittal`, `_coronal`, or `_transverse`:**  
   If the `default_plane` attribute is not present, or the acquisition plane (determined using the DICOM header tag `0020|0037`) of the series differs from the value specified in the `default_plane` attribute, the plane is appended to the key.

2. **`_0` to `_n`:**  
   If the `repeat_acq` attribute is set to `True`, "_0" is appended to the first acquisition with the key created following the above rules. Subsequent acquisitions with the same key will have suffixes like `_1`, `_2`, ..., `_n`. If `repeat_acq` is set to `False` (or not supplied), "_0" is not appended, and any repeated occurrence of the same key is ignored (after logging an error).

**The value of each dataset must be the data itself.**

Each dataset may (optionally) include the following attributes: series ID, DICOM header, description of the series, the min and max intensity values of the series (of the magnitude, in case of complex-valued):
```python
    seriesID, DICOMHeader, seriesDesc, min_val, max_val
```

For volumetric normalisation modes (e.g., `norm_type = divbymaxvol` or `norm_type = minmaxvol`), the `min_val` and `max_val` attributes are required.


### Data Dimensions
The dataset must be 5D, with the following shape:
```python
Channel : Time : Slice : X : Y 
```

- **`Channel`:** This dimension is used to stack different MRIs from multi-echo or multi-TIeff MRI acquisitions, referred to as "Channels." In the case of multi-contrast MRIs, this dimension can also be used, but only if the images are co-registered. If there is only one channel, the shape of this dimension is `1`.
- **`Time`:** For dynamic MRIs (and other dynamic acquisitions), the different time points should be concatenated along this dimension. If there is only one time point, the shape of this dimension is `1`.
- **`Slice`:** For 3D MRIs, this dimension stores the different slices. For 2D acquisitions, the shape of this dimension will be `1`.
- **`X` and `Y`:** These represent the in-plane spatial dimensions.

For any other type of data, the dimensions can be reshaped to fit this structure (i.e., unnecessary dimensions can be set to have a shape of `1`).

**Note:**
In [this research](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/), the UK Biobank MRI ZIP files were processed using the script available at [https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS/blob/master/preprocess/createH5s/createH5_MR_DICOM.py](https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS/blob/master/preprocess/createH5s/createH5_MR_DICOM.py) to create the corresponding HDF5 file.

## Trained Weights from Hugging Face
The 3D DiffAE models trained on the CINE Cardiac Long Axis MRIs from UK Biobank as part of the research [Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/) are available on [Hugging Face](https://huggingface.co/collections/soumickmj/cardiacdiffae-gwas-671b7595d09b0746b8fd0b72). 

To use the weights (without this pipeline), you can directly load them using the Hugging Face Transformers library or you can use them with this library using by supplying the `--load_hf` argument to the main files. For example,
```python
    python Executors/main_diffAE.py --load_hf GlastonburyGroup/UKBBLatent_Cardiac_20208_DiffAE3D_L128_S1701
```
After loading, the model can further be trained (treating our weights as pretrained weights) or used for inference (following the instructions in the previous section, except the `--resume --load_best` flags).

An [application]((https://huggingface.co/spaces/GlastonburyGroup/Live_UKBBLatent_Cardiac_20208_DiffAE3D_L128)) is also been hosted on [Hugging Face Spaces](https://huggingface.co/spaces/GlastonburyGroup/Live_UKBBLatent_Cardiac_20208_DiffAE3D_L128), where you can use your own MRIs to infer latent representations using the trained 3D DiffAE models. 


## Citation
If you find this work useful or utilise this pipeline (or any part of it) in your research, please consider citing us:
```bibtex
@article{Ometto2024.11.04.24316700,
            author       = {Ometto, Sara and Chatterjee, Soumick and Vergani, Andrea Mario and Landini, Arianna and Sharapov, Sodbo and Giacopuzzi, Edoardo and Visconti, Alessia and Bianchi, Emanuele and Santonastaso, Federica and Soda, Emanuel M and Cisternino, Francesco and Pivato, Carlo Andrea and Ieva, Francesca and Di Angelantonio, Emanuele and Pirastu, Nicola and Glastonbury, Craig A},
            title        = {Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture},
            elocation-id = {2024.11.04.24316700},
            year         = {2024},
            doi          = {10.1101/2024.11.04.24316700},
            publisher    = {Cold Spring Harbor Laboratory Press},
            url          = {https://www.medrxiv.org/content/early/2024/11/05/2024.11.04.24316700},
            journal      = {medRxiv}
          }  
```

## Credits
This pipeline is developed by [Dr Soumick Chatterjee](https://github.com/soumickmj) (as part of the [Glastonbury Group](https://humantechnopole.it/en/research-groups/glastonbury-group/), [Human Technopole, Milan, Italy](https://humantechnopole.it/en/)) based on the [NCC1701 pipeline](https://github.com/soumickmj/NCC1701) from the paper [*ReconResNet: Regularised residual learning for MR image reconstruction of Undersampled Cartesian and Radial data*](https://doi.org/10.1016/j.compbiomed.2022.105321). Special thanks to [Dr Domenico Iuso](https://github.com/snipdome) for collaborating on enhancing the NCC1701 and this pipeline with the latest PyTorch (and related) features, including DeepSpeed, and to [Rupali Khatun](https://github.com/rupaliasma) for her contributions to the pipeline with the complex-valued autoencoders.

### DiffAE: Diffusion Autoencoder
The 2D DiffAE model in this repository is based on the paper [*Diffusion Autoencoders: Toward a Meaningful and Decodable Representation*](https://diff-ae.github.io/). The repository adapts the code from the original [DiffAE repository](https://github.com/phizaz/diffae) to work with non-RGB images (e.g., MRIs) and extends it to 3D for processing volumetric images. 

If you are using the DiffAE model from this repository, in addition to citing our paper mentioned above, please also cite the original paper:
```bibtex
@inproceedings{preechakul2021diffusion,
      title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation}, 
      author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2022},
}
```

### pythae: Unifying Generative Autoencoders in Python
For non-diffusion autoencoders (including VAEs), this pipeline utilises and extends (e.g. additional models, including complex-valued models) the [pythae](https://github.com/clementchadebec/benchmark_VAE) package. This package has been integrated into our pipeline, and to use models from this package, one must supply `0` as the `modelID`, the model name (from the list in the package's `__init__.py` file located inside `Engineering/Engines/WarpDrives/pythaeDrive`) as `pythae_model`, and the relative path (from `Engineering/Engines/WarpDrives/pythaeDrive/configs`) to the configuration JSON file for pythae as `pythae_config`. This is optional; if left blank, the default configuration will be used. 

For example, for the Factor VAE's configuration file intended for the CelebA dataset, `originals/celeba/factor_vae_config.json` must be supplied. Default configurations can also be found in the same `__init__.py` file. The `__init__.py` file must be updated whenever a new model is added to the package or a new one is introduced. 

The original configuration files (including the entire package) are intended for a few "toy" image datasets (binary_mnist, celeba, cifar10, dsprites, and mnist). Since CelebA is the most complex dataset among them, we have chosen those configurations as defaults. However, these configurations may need to be modified to suit our specific tasks.

If you are using any of the non-diffusion autoencoder (including VAEs) models from this repository, in addition to citing our paper mentioned above, please also cite the original paper:
```bibtex
@inproceedings{chadebec2022pythae,
        author = {Chadebec, Cl\'{e}ment and Vincent, Louis and Allassonniere, Stephanie},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
        pages = {21575--21589},
        publisher = {Curran Associates, Inc.},
        title = {Pythae: Unifying Generative Autoencoders in Python - A Benchmarking Use Case},
        volume = {35},
        year = {2022}
}
```

