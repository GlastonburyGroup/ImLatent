# UKBBLatent: Unsupervised phenotyping and genetic discovery in 71,021 cardiac MRIs

The official code for the paper "Unsupervised phenotyping and genetic discovery in 71,021 cardiac MRIs" (https://arxiv.org/abs/XXXX.XXXX).

## Pipeline 
This pipeline is based on the NCC1701 pipeline (of Soumick): https://github.com/soumickmj/NCC1701 from the paper: https://doi.org/10.1016/j.compbiomed.2022.105321
Special thanks to Domenico Iuso for working together with me (Soumick) on improving the NCC1701 with the latest features, including DeepSpeed!

### pythae
A really nice package, a benchmarking platform, for training autoencoders (mainly VAEs). This package has been integrated into our pipeline. 
To use models from this package, you would need to supply 0 as the modelID, model name (from the list: Engineering/Engines/WarpDrives/pythaeDrive/__init__.py) as pythae_model, and the relative path (from the Engineering/Engines/WarpDrives/pythaeDrive/configs) to the config JSON file for pythae as pythae_config (optional, if blank, default one will be used). For example, for the Factor VAE's config file meant for the celeba dataset, "originals/celeba/factor_vae_config.json" must be supplied (defaults can also be found inside the same _init__.py file). The init file must be updated when a new model is added to the package, or we have introduced a new one. The original config files (including the whole package) are meant for a few "toy" image datasets (binary_mnist, celeba, cifar10, dsprites, and mnist). As celeba is the most complex one among them, we are choosing those configs as defaults. But we might need to modify them for our tasks.

### Engaging (i.e. Executing) the pipeline 
Inside the _Bridge_ package, there are the actual execution scripts. For different problems, we can put the _main_ files here. We can have different sub-packages here to organise the different experiments better.
_main*.py_ : is the actual main file that is to be used to engage the pipeline. (_prova*_ file is similar to the main one but to be used for debugging purposes on a dummy dataset). This file contains all the default values for different command line arguments - these can be supplied while calling this script.

Inside the _Cargo_ folder, there are the two types files required by the _main_ file. Sub-folder structures can be created for organising the different experiments better.
_config*.yaml_: The configurations, more specific ones to the different aspects of the pipeline (e.g. LR scheduler parameters), which won't be modified on a regular basis, are kept here in this file - in a hierarchical fashion. These values can also be overridden using command line arguments, discussed later.
_dataNpath*.json_: As the name suggests, this file contains dataset-specific parameters, including the name of the foldCSV, run_prefix, etc., as well as the required paths - data directory (where the data.h5, supplied .csv file for the folds can be found - and also where the processed dataset files will be stored). 

To engage, the call should be from the root directly of the pipeline. For example:
```python
    python Bridge/main_recon.py --batch_size 32 --lr 0.0001 --training§LRDecay§type1§decay_rate 0.15 --training§prova testing --json§save_path /myres/toysets/Results
```
Here, _main_recon.py_ is the main file that is to be executed, _batch_size_ and _lr_ are arguments which will be replacing the default values for those parameters supplied within that main file, _training§LRDecay§type1§decay_rate_ and _training§prova_ (arguments not specified inside the main file or any of the Engines) are going to override the values present inside the config.yaml file mentioned inside the main file (or supplied as a command line argument) - following the path splitting the key with dollars, and finally, _json§save_path_ (same as the earlier one, but arguments starting with _json§_) will override the _save_path_ value of the parameter inside the _dataNpath.json_ specified inside the main file (or supplied as a command line argument).
Notes:  
_training§LRDecay§type1§decay_rate_ will try to find the dictonary path _training/LRDecay/type1/decay_rate_ inside the yaml file. If the path is found, the value will be updated (for the current run only) with the supplied one. If it's not found, like in this example training§prova, a new path will be created, and the value will be added (for the current run only). Any command line argument that is not found inside the main file, or any of the Engines or Warp Drives, will be treated as an "unknown" parameter and will be treated in this manner - unless they start with "_json§". In that case, it is used to update the value of _save_path_ present inside the _dataNpath.json_ for the current run. 

#### Running inference of a trained model
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
The original dataset is first processed using https://gitlab.fht.org/glastonburygroup/tricorder/-/blob/main/preprocess/createMRH5.py to create the corresponding HDF5 file.

Structure of the HDF5:
Groups with the path 
```python
    patientID/fieldID/instanceID
```
Groups have the following attributes: DICOM Patient ID, DICOM Study ID, description of the study, AET (not sure what it means, but it seems to have the model of the scanner), 
Host (some ID, not sure!!), date of the study, number of series in the study:
```python
    patientDICOMID, studyDICOMID, studyDesc, aet, host, date, n_series
```
Inside each group, each series present is stored as a different dataset (except if it is specified in the https://gitlab.fht.org/glastonburygroup/tricorder/-/blob/main/preprocess/createH5s/meta.yaml to stack some dim). The key for the datasets can be:

**primary**: to store the primary data (i.e. the series description matches with one of the values of the primary_data attribute).

**primary_***: (optional) In case the multi_primary attribute is set to True, then inside of just "primary", there will be multiple "primary_*", where "*" will be replaced by the corresponding tag supplied using the primary_data_tags attribute.

**auxiliary_***: (optional) to store auxiliary data [e.g. T1 maps in case of ShMoLLI] (i.e. the series description matches with one of the values of the auxiliary_data attribute). "*" will be replaced by the corresponding tag supplied using the auxiliary_data_tags attribute.

Additionally, some additional type info might be added to the key.

**_sagittal** or **_coronal** or **_transverse**: If the default_plane attribute is not present or the acquisition plane (computed using the DICOM header tag 0020|0037) of the series is different from the one specified in the default_plane attribute, then the plane will be concatenated with the key.

**_0** to **_n**: If the repeat_acq attribute is set to True, then "_0" will be concatenated with the first acquisition with the already created key (following the rules mentioned so far). The subsequent acquisitions with the same key will have suffixes like "_1, _2,...,_n". If repeat_acq is set to False (or not supplied), then "_0" won't be concatenated, and any repeat occurrence of the same key will be ignored (i.e. the series will be ignored after logging an error). 

[Note: all the mentioned attributes mentioned in this section refer to the attributes present inside the previously-mentioned meta.yaml file]

The value of the dataset is the data itself.

Each dataset additionally contains five attributes - series ID, DICOM header (ignoring Siemens' CSA header - tags starting with 0029 - as they contain unnecessary things and are also large), description of the series, the min and max intensity values of the series (of the magnitude, in case of complex-valued):
```python
    seriesID, DICOMHeader, seriesDesc, min_val, max_val
```
There should be one of each of these attributes (for group and dataset). If for some reason, there are multiple, then only the first one is taken, and a warning is logged for the same.

As complex-valued data is created (explained later) by combining the magnitude and phase images, there will always be 2 of each - seriesID and DICOMHeader. Hence, within these two attributes, there will be "mag_0" and "phase_0", and the value of these keys will be the actual series ID and DICOM header. In the case of real-valued data, there will be only "mag_0". 

Moreover, there can be dimensional concatenation if specified by the stack_dim attribute inside the meta.yaml file. In that case, there will be even more series IDs, and DICOM headers need to be stored. Hence, they are stored as "mag_0" to "mag_n" (additionally, "phase_0" to "phase_n" for complex-valued data). 

If an error is encountered while creating any group or dataset, that group is removed, and the error is logged.

### DIXON Dataset (F20201)
Neck-to-knee data was acquired using 6 different series. Those needed to be stitched to obtain the complete 3D volume, but they had different numbers of slices, contrast differences, etc. - making it difficult to combine them. The pipeline provided by the Research Centre for Optimal Health of the University of Westminster takes care of all these issues and does further analysis of the data (e.g. calculating fat and water percentages and finding bone joints). We have initially processed the UKBB provided zip files (6 series, 4 channels each - water, fat, in-phase, opposed-phase) using this pipeline (cf. https://github.com/recoh/pipeline) - referred to here as the RECOH pipeline, then created the HDF5 file using the different outputs of the pipeline. In each of the groups inside the HDF5 of this dataset, two additional attributes can be found: _n_DICOM_series_ (typically, 24) and _DICOMSeriesIDs_ (seriesIDs of the individual DICOMs). _n_series_ attributed in this case tells us whether multiple sets of acquisitions were performed. Each group contains 4 datasets and 2 more attributes - all these 6 things were obtained from the RECOH pipeline. The 4 datasets are: 
```python
    primary_WaterFat_i, primary_InOpp_i, auxiliary_WaterFatPercent_i, meta_mask_i
```
where _i_ denotes the acquisition number starting with 0. These datasets are 2-channel water-fat volumes, 2-channel in-phase opposed-phase volumes, 2-channel water-fat percentage (computed), and a 3D mask of the whole body (might be required as there is background noise in the images), respectively. Each dataset contains the same attributes as mentioned above, except for the two missing ones: _DICOMHeader_ and _seriesDesc_. Moreover, the 2 attributes obtained from the RECOH pipeline are:
```python
    meta_bone_joints_i, meta_RECOHPipe_i
```
containing JSON data - x-y-z coordinates of the bone joints and the metadata provided as a summary by the pipeline, respectively.

### Complex-valued Data
The dataset can be real-valued or complex-valued. The dataset contains individual magnitude and phase images, and the corresponding complex images are created using the following formula:
```python
complex_image = magnitude_data * (np.cos(phase_data) + 1j * np.sin(phase_data))
```
or rather, using the more computationally efficient expression:
```python
complex_image = magnitude_data * np.exp(1j * phase_data)
```
Whether a specific series is a magnitude or phase image is determined by the value of the DICOM header 0008|0008 - if it contains "ORIGINAL\\PRIMARY\\M" or "ORIGINAL\\PRIMARY\\P". While combining the magnitude and phase images, the 0020|0012 DICOM header of the magnitude and phase images are compared - to make sure the correct pairs are being combined.

To fetch the magnitude and phase images, the following can be applied to the dataset:
```python
magnitude = abs(complex_image)
phase = np.angle(complex_image)
```

### Data Dimensions
The dataset is 5D, having the following shape:
```python
    Channel : Time : Slice : X : Y 
```
Channel: Different MRIs from multi-echo or multi-TIeff MRI acquisitions will use this dimension to stack them, which will be referred to as "Channels". In the case of multi-contrast MRIs, this dimension can also be used - but only if they are co-registered. If there's only one, then the shape of this dim is 1.
Time: For dynamic MRIs (and other dynamic acquisitions), the different time-points are to be concatenated in this dim. If only one TP, the shape of this dim is 1.
Slice: For 3D MRIs, this dim will store the different slices. For 2D acquisitions, this dim will have a shape of 1.
X and Y: The actual in-plane spatial dimensions.
