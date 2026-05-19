# Unsupervised latent representation learning using 2D and 3D diffusion and other autoencoders

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20204053.svg)](https://doi.org/10.5281/zenodo.20204053)
[![Preprint](https://img.shields.io/badge/medRxiv-2024.11.04.24316700-b31b1b.svg)](https://doi.org/10.1101/2024.11.04.24316700)
[![Project page](https://img.shields.io/badge/Project-page-blue.svg)](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow.svg)](https://huggingface.co/collections/soumickmj/cardiacdiffae-gwas-671b7595d09b0746b8fd0b72)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository provides a pipeline for training and evaluating 2D and 3D diffusion autoencoders, traditional autoencoders, and various variational autoencoders for unsupervised latent representation learning from 2D and 3D images, primarily focusing on MRIs. It is the deep learning pipeline used in the paper [*Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture*](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/) (accepted in *Nature Communications*; preprint at [medRxiv](https://doi.org/10.1101/2024.11.04.24316700)), in which it was used to learn and infer latent representations from cardiac CINE MRIs with a 3D diffusion autoencoder. Note that this repository contains only the DL pipeline; the full set of scripts accompanying the paper (preprocessing, GWAS, downstream analyses) lives separately under the same [GitHub organisation](https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS).

The snapshot of this codebase at the time of acceptance is archived on Zenodo: [10.5281/zenodo.20204053](https://doi.org/10.5281/zenodo.20204053).

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
  - [Contact](#contact)


## Pipeline

### Structure
**Inside the _Executors_ package**, you will find the actual execution scripts. For different problems, the relevant *main* files can be placed here. Sub-packages may also be created within this package to better organise different experiments.

- **_main*.py_:** The main script used to run the pipeline. It contains the default values for the various command-line arguments, which can be overridden when invoking the script.

Currently, three distinct main files are available:
1. `main_recon.py`: For 2D and 3D autoencoders (including VAEs, but excluding diffusion autoencoders).
2. `main_diffAE.py`: For 2D and 3D diffusion autoencoders.
3. `main_latentcls.py`: For training a classifier on the latent space.

**Inside the *Configs* folder,** there are two types of files required by the *main* scripts. Sub-folder structures can be created within this folder to better organise different experiments.

- **_config*.yaml_:** These files contain configuration parameters, typically specific to different aspects of the pipeline (e.g., learning rate scheduler parameters). As these parameters are less likely to change frequently, they are defined here in a hierarchical format. They can also be overridden using command-line arguments, as described below.
- **_dataNpath*.json_:** As the name suggests, these files include dataset-specific parameters, such as the name of the foldCSV file and the run prefix. They also define the necessary paths, such as the data directory (where the `data.h5` file, the supplied CSV for folds, and the processed dataset files are stored).

**Inside the *Engineering* package,** all the files related to the actual implementation of the pipeline are stored.


### Executing the pipeline

A conda environment can be created using the provided `environment.yml` file.

To execute, the call should be made from the root directory of the pipeline. For example:
```bash
python Executors/main_recon.py --batch_size 32 --lr 0.0001 --training§LRDecay§type1§decay_rate 0.15 --training§prova testing --json§save_path /myres/toysets/Results
```

[Poetry](https://python-poetry.org/) can be used instead of a conda environment. Once Poetry is installed, this pipeline can be launched from its root directory without manually installing any dependencies (or using the yml file) by prefixing the command with `poetry run`. For example:
```bash
poetry run python Executors/main_recon.py --batch_size 32 --lr 0.0001 --training§LRDecay§type1§decay_rate 0.15 --training§prova testing --json§save_path /myres/toysets/Results
```
For continuous use without prefixing every command with `poetry run`, `poetry shell` (which must be installed separately: https://github.com/python-poetry/poetry-plugin-shell.git) can be executed to activate the environment, after which Python commands can be run normally.

In the example above, `main_recon.py` is the script being executed. `batch_size` and `lr` override the default values for those parameters defined inside the main file. `training§LRDecay§type1§decay_rate` and `training§prova` (arguments not declared in the main file or any of the Engines) override the corresponding values inside the config.yaml file specified by the main file (or supplied as a command-line argument), with the key split on the `§` delimiter to traverse the nested dictionary. Finally, `json§save_path` (any argument starting with `json§`) overrides the corresponding key inside the `dataNpath.json` specified by the main file (or supplied as a command-line argument).

A note on behaviour: `training§LRDecay§type1§decay_rate` will look up the dictionary path `training/LRDecay/type1/decay_rate` inside the yaml file. If the path exists, the value is updated for the current run; if not (as with `training§prova` above), the path is created and the value is added for the current run. Any command-line argument that is not declared in the main file or any of the Engines or Warp Drives is treated as "unknown" and handled in this manner — unless it starts with `json§`, in which case it updates a value inside the `dataNpath.json` for the current run.


For the complete list of command-line arguments, please refer to the main files or run the main file with the `--help` flag, e.g.:
```bash
python Executors/main_diffAE.py --help
```
and check the files inside the *Configs* folder.


#### Running inference on a trained model
Add the following command-line arguments to those used during training:
```bash
--resume --load_best --run_mode 2
```
To run inference of a model currently in training (i.e. interim inference), add:
```bash
--resume --load_best --run_mode 2 --output_suffix XYZ
```
where `XYZ` is a suffix that will be appended to the output directory.

To run inference on the whole dataset (ignoring the splits), use:
```bash
--resume --load_best --run_mode 2 --output_suffix fullDS --json§foldCSV 0
```
(`fullDS` can of course be replaced with any other suffix.)

## Dataset
This pipeline expects an HDF5 file containing the dataset as input, following the structure described below.

The groups should follow the path:
```
patientID/fieldID/instanceID
```
Groups may (optionally) include the following attributes: DICOM Patient ID, DICOM Study ID, description of the study, AET (model of the scanner), Host, date of the study, number of series in the study:
```
patientDICOMID, studyDICOMID, studyDesc, aet, host, date, n_series
```
Each series present is stored as a separate dataset, and the key for the datasets can be:

- **`primary`:** To store the primary data (i.e., the series description matches one of the values of the `primary_data` attribute).
- **`primary_*`:** (Optional) If the `multi_primary` attribute is set to `True`, then instead of a single `primary` key, there will be multiple keys in the form `primary_*`, where `*` is replaced by the corresponding tag supplied via the `primary_data_tags` attribute.
- **`auxiliary_*`:** (Optional) To store auxiliary data (e.g., T1 maps in the case of ShMoLLI). Here, `*` is replaced by the corresponding tag supplied via the `auxiliary_data_tags` attribute.

Additional type information may be appended to the dataset key:

1. **`_sagittal`, `_coronal`, or `_transverse`:**
   If the `default_plane` attribute is not present, or if the acquisition plane (determined using the DICOM header tag `0020|0037`) of the series differs from the value specified in the `default_plane` attribute, the plane is appended to the key.

2. **`_0` to `_n`:**
   If the `repeat_acq` attribute is set to `True`, `_0` is appended to the first acquisition with the key created following the above rules. Subsequent acquisitions with the same key will have suffixes `_1`, `_2`, ..., `_n`. If `repeat_acq` is set to `False` (or not supplied), `_0` is not appended, and any repeated occurrence of the same key is ignored (after logging an error).

**The value of each dataset must be the data itself.**

Each dataset may (optionally) include the following attributes: series ID, DICOM header, description of the series, the min and max intensity values of the series (of the magnitude, in the case of complex-valued data):
```
seriesID, DICOMHeader, seriesDesc, min_val, max_val
```

For volumetric normalisation modes (e.g., `norm_type = divbymaxvol` or `norm_type = minmaxvol`), the `min_val` and `max_val` attributes are required.


### Data Dimensions
The dataset must be 5D, with the following shape:
```
Channel : Time : Slice : X : Y
```

- **`Channel`:** Used to stack different MRIs from multi-echo or multi-TIeff acquisitions ("Channels"). For multi-contrast MRIs, this dimension can also be used, but only if the images are co-registered. If there is only one channel, the shape of this dimension is `1`.
- **`Time`:** For dynamic MRIs (and other dynamic acquisitions), the different time points should be concatenated along this dimension. If there is only one time point, the shape of this dimension is `1`.
- **`Slice`:** For 3D MRIs, this dimension stores the different slices. For 2D acquisitions, the shape of this dimension is `1`.
- **`X` and `Y`:** These represent the in-plane spatial dimensions.

For any other type of data, the dimensions can be reshaped to fit this structure (i.e., unnecessary dimensions can be set to a shape of `1`).

**Note:**
In [this research](https://glastonburygroup.github.io/CardiacDiffAE_GWAS/), the UK Biobank MRI ZIP files were processed using the script available at [https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS/blob/master/preprocess/createH5s/createH5_MR_DICOM.py](https://github.com/GlastonburyGroup/CardiacDiffAE_GWAS/blob/master/preprocess/createH5s/createH5_MR_DICOM.py) to create the corresponding HDF5 file.

## Trained Weights from Hugging Face
The 3D DiffAE models trained on the CINE Cardiac Long Axis MRIs from UK Biobank as part of the paper are available on [Hugging Face](https://huggingface.co/collections/soumickmj/cardiacdiffae-gwas-671b7595d09b0746b8fd0b72).

The weights can be loaded directly using the Hugging Face Transformers library (without this pipeline), or used with this pipeline by supplying the `--load_hf` argument to the main files. For example:
```bash
python Executors/main_diffAE.py --load_hf GlastonburyGroup/UKBBLatent_Cardiac_20208_DiffAE3D_L128_S1701
```
Once loaded, the model can be trained further (treating our weights as pretrained) or used for inference (following the instructions in the previous section, except the `--resume --load_best` flags).

An [interactive application](https://huggingface.co/spaces/GlastonburyGroup/Live_UKBBLatent_Cardiac_20208_DiffAE3D_L128) is also hosted on [Hugging Face Spaces](https://huggingface.co/spaces/GlastonburyGroup/Live_UKBBLatent_Cardiac_20208_DiffAE3D_L128), where you can supply your own MRIs and infer latent representations using the trained 3D DiffAE models.

## Citation
If you use this pipeline (or any part of it) in your research, please cite the paper:

> Ometto, S.\*, Chatterjee, S.\*, Vergani, A. M., Landini, A., Sharapov, S., Giacopuzzi, E., Visconti, A., Bianchi, E., Santonastaso, F., Soda, E. M., Cisternino, F., Pivato, C. A., Ieva, F., Di Angelantonio, E., Pirastu, N., Glastonbury, C. A. *Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture*. **Nature Communications** (in press, 2026). Preprint: [medRxiv 2024.11.04.24316700](https://doi.org/10.1101/2024.11.04.24316700).
>
> \*Joint first authors.

BibTeX (preprint version; will be updated once the journal version is online):
```bibtex
@article{Ometto2024.11.04.24316700,
    author       = {Ometto, Sara and Chatterjee, Soumick and Vergani, Andrea Mario and Landini, Arianna and Sharapov, Sodbo and Giacopuzzi, Edoardo and Visconti, Alessia and Bianchi, Emanuele and Santonastaso, Federica and Soda, Emanuel M and Cisternino, Francesco and Pivato, Carlo Andrea and Ieva, Francesca and Di Angelantonio, Emanuele and Pirastu, Nicola and Glastonbury, Craig A},
    title        = {Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture},
    elocation-id = {2024.11.04.24316700},
    year         = {2024},
    doi          = {10.1101/2024.11.04.24316700},
    publisher    = {Cold Spring Harbor Laboratory Press},
    url          = {https://www.medrxiv.org/content/early/2024/11/05/2024.11.04.24316700},
    journal      = {medRxiv},
    note         = {Ometto and Chatterjee contributed equally. Accepted at Nature Communications.}
}
```

Please also cite the archived codebase if you refer to a specific snapshot (if you use the version used in the cardiac MRI paper):
```bibtex
@software{ImLatent_Zenodo_NatCommCardiacMRI,
    title     = {GlastonburyGroup/ImLatent: Hundreds of cardiac MRI traits derived using 3D diffusion autoencoders share a common genetic architecture},
    year      = {2026},
    publisher = {Zenodo},
    version   = {NatCommCardiacMRI},
    doi       = {10.5281/zenodo.20204053},
    url       = {https://doi.org/10.5281/zenodo.20204053}
}
```

## Credits
This pipeline was developed by [Dr Soumick Chatterjee](https://github.com/soumickmj) (as part of the [Glastonbury Group](https://humantechnopole.it/en/research-groups/glastonbury-group/), [Human Technopole, Milan, Italy](https://humantechnopole.it/en/)) and builds on the [NCC1701 pipeline](https://github.com/soumickmj/NCC1701) from the paper [*ReconResNet: Regularised residual learning for MR image reconstruction of Undersampled Cartesian and Radial data*](https://doi.org/10.1016/j.compbiomed.2022.105321). Special thanks to [Dr Domenico Iuso](https://github.com/snipdome) for collaborating on enhancing NCC1701 and this pipeline with newer PyTorch features (including DeepSpeed), and to [Rupali Khatun](https://github.com/rupaliasma) for her contributions on the complex-valued autoencoders.

### DiffAE: Diffusion Autoencoder
The 2D DiffAE model in this repository is based on the paper [*Diffusion Autoencoders: Toward a Meaningful and Decodable Representation*](https://diff-ae.github.io/). The code is adapted from the original [DiffAE repository](https://github.com/phizaz/diffae) to work with non-RGB images (e.g., MRIs) and extended to 3D for volumetric inputs.

If you use the DiffAE model from this repository, in addition to the paper above, please also cite the original work:
```bibtex
@inproceedings{preechakul2021diffusion,
    title     = {Diffusion Autoencoders: Toward a Meaningful and Decodable Representation},
    author    = {Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```

### pythae: Unifying Generative Autoencoders in Python
For non-diffusion autoencoders (including VAEs), this pipeline uses and extends (with additional models, including complex-valued variants) the [pythae](https://github.com/clementchadebec/benchmark_VAE) package. The package is integrated into the pipeline, and to use models from it one must supply `0` as the `modelID`, the model name (from the list in `Engineering/Engines/WarpDrives/pythaeDrive/__init__.py`) as `pythae_model`, and the relative path (from `Engineering/Engines/WarpDrives/pythaeDrive/configs`) to the configuration JSON file as `pythae_config`. The configuration is optional; if left blank, the default is used.

For example, for the Factor VAE configuration intended for CelebA, supply `originals/celeba/factor_vae_config.json`. Default configurations are also listed in the same `__init__.py` file, which must be updated whenever a model is added or modified.

The original configuration files (and the package itself) target a few "toy" image datasets (binary_mnist, celeba, cifar10, dsprites, mnist). Since CelebA is the most complex of these, we use those configurations as defaults, though they will typically need to be adjusted for specific tasks.

If you use any of the non-diffusion autoencoder (including VAE) models from this repository, in addition to the paper above, please also cite the original work:
```bibtex
@inproceedings{chadebec2022pythae,
    author    = {Chadebec, Cl\'{e}ment and Vincent, Louis and Allassonniere, Stephanie},
    booktitle = {Advances in Neural Information Processing Systems},
    editor    = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
    pages     = {21575--21589},
    publisher = {Curran Associates, Inc.},
    title     = {Pythae: Unifying Generative Autoencoders in Python - A Benchmarking Use Case},
    volume    = {35},
    year      = {2022}
}
```

## Contact

For questions about the code or the analyses, please open an issue or get in touch with Soumick Chatterjee (<soumick.chatterjee@fht.org>, <contact@soumick.com>).
