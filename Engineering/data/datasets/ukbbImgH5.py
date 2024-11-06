from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
import hashlib
import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class UKBBImgH5(Dataset):
    def __init__(self, h5_file, transform=None, patientIDs=None, fetchMinMax=True, #These params were also present in the version for the V0 Dataset. fetch2D and expand_ch params were there in V0, have been removed in V1.
                        every_primary=True, selected_primaries=None, selected_auxiliaries=None, include_repeatAcq=True, combine_acquisitions=False, sanity_filters=None, #New params added for filtering the V1 dataset better
                        multi_channel=True, split_volume=True, split_time=True, #New params added for handlling the 5D data structure of the V1 dataset
                        channel_dim=0, slice_dim=2, data_mask_mode=0, #Params to make any dim as the channel and slice.
                        fetch_phenotypes=False, phenotypes=None, #Params to fetch phenotypes
                        fetch_confounders=False, confounders=None, #Params to fetch confounders
                        squeeze_slice=True): #param to decide whether to squeeze the obtained slice or not
        self.h5_file = h5_file
        self.transform = transform
        self.fetchMinMax = fetchMinMax
        self.data_mask_mode = data_mask_mode
        self.squeeze_slice = squeeze_slice

        self.fetch_phenotypes = fetch_phenotypes
        if self.fetch_phenotypes:
            self.phenotypes = phenotypes
            try:
                self.phenotypes['data'] = pd.read_table(self.phenotypes['path'], index_col="FID")[phenotypes['cols']]
            except:
                self.phenotypes['data'] = pd.read_table(self.phenotypes['path'], index_col="f.eid")[phenotypes['cols']]
                self.phenotypes['data'].rename_axis("FID", axis='index', inplace=True)
            scaler = StandardScaler()
            self.phenotypes['scaler_mean'] = {}
            self.phenotypes['scaler_scale'] = {}
            for column in self.phenotypes['data'].columns:
                self.phenotypes['data'][column] = scaler.fit_transform(self.phenotypes['data'][[column]])
                self.phenotypes['scaler_mean'][column] = scaler.mean_
                self.phenotypes['scaler_scale'][column] = scaler.scale_

        self.fetch_confounders = fetch_confounders
        if self.fetch_confounders:
            self.confounders = confounders
            try:
                self.confounders['data'] = pd.read_table(self.confounders['path'], index_col="FID")[confounders['cols']]
            except:
                self.confounders['data'] = pd.read_table(self.confounders['path'], index_col="f.eid")[confounders['cols']]
                self.confounders['data'].rename_axis("FID", axis='index', inplace=True)
            scaler = StandardScaler()
            self.confounders['scaler_mean'] = {}
            self.confounders['scaler_scale'] = {}
            self.confounders['le_class_order'] = {}
            for column in confounders['cont_cols']:
                self.confounders['data'][column] = scaler.fit_transform(self.confounders['data'][[column]])
                self.confounders['scaler_mean'][column] = scaler.mean_
                self.confounders['scaler_scale'][column] = scaler.scale_
            le = LabelEncoder()
            for column in confounders['mulcat_cols'] + confounders['bincat_cols']:
                self.confounders['data'][column] = le.fit_transform(self.confounders['data'][column])
                self.confounders['le_class_order'][column] = le.classes_

        if selected_primaries is not None:
            for i in range(len(selected_primaries)):
                if not selected_primaries[i].startswith("primary"):
                    selected_primaries[i] = f"primary_{selected_primaries[i]}"

        if selected_auxiliaries is not None:
            for i in range(len(selected_auxiliaries)):
                if not selected_auxiliaries[i].startswith("auxiliary"):
                    selected_auxiliaries[i] = f"auxiliary_{selected_auxiliaries[i]}"

        self.multi_channel = multi_channel
        self.split_volume = split_volume
        self.split_time = split_time
        
        h5_init = h5py.File(h5_file, 'r', libver='latest', swmr=True) 
        self.allPatientIDs = list(h5_init.keys())

        self.search_criteria = {
            'patientIDs': patientIDs if bool(patientIDs) else self.allPatientIDs,
            'every_primary': every_primary,
            'selected_primaries': None if every_primary else selected_primaries,
            'selected_auxiliaries': selected_auxiliaries,
            'include_repeatAcq': include_repeatAcq,
            'combine_acquisitions': combine_acquisitions #TODO Not implemented yet
        }
        
        dim_list = [0,1,2,3,4]
        if channel_dim not in [0, -1, None]: #a custom channel_dim has been provided and it's not the same as the default one
            self.search_criteria['custom_channel_dim'] = channel_dim
            dim_list[0], dim_list[channel_dim] = dim_list[channel_dim], dim_list[0]
        if slice_dim not in [2, -1, None]: #a custom slice_dim has been provided and it's not the same as the default one
            self.search_criteria['custom_slice_dim'] = slice_dim
            dim_list[2], dim_list[slice_dim] = dim_list[slice_dim], dim_list[2]
        if np.array_equal(dim_list, [0,1,2,3,4]):
            self.do_transpose = False
        else:
            self.do_transpose = True
            self.transpose_dim_order = dim_list

        if sanity_filters is not None:
            self.search_criteria['sanity_filters'] = sanity_filters

        processed_name = f"DSPaths{'_mulChan' if multi_channel else ''}{'_splitVol' if split_volume else ''}{'_splitTime' if split_time else ''}"
        md5_patients = hashlib.md5(json.dumps(self.search_criteria, sort_keys=True).encode('utf-8')).hexdigest()
        sha_patients = hashlib.sha3_256(json.dumps(self.search_criteria, sort_keys=True).encode('utf-8')).hexdigest()
        processed_path = h5_file.replace(".h5", f"_processedmeta/{processed_name}_MD5{md5_patients}_SHA3{sha_patients}.json")
        if os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                self.DSPaths = json.load(f)
        else:    
            self.DSPaths = []
            def get_paths(name, obj):
                name_parts = name.split('/')
                #TODO: replace the current search with regex-based search
                if isinstance(obj, h5py.Dataset) and name_parts[0] in self.search_criteria['patientIDs']:
                    if 'primary' not in name_parts[-1] and 'auxiliary' not in name_parts[-1]:
                        return
                    elif (
                        'primary' in name_parts[-1]
                        and not every_primary
                        and (
                            self.search_criteria['selected_primaries'] is None 
                            or (name_parts[-1] not in self.search_criteria['selected_primaries']
                            and name_parts[-1].replace("_0", "") not in self.search_criteria['selected_primaries'])
                        )
                    ):
                        return
                    elif 'auxiliary' in name_parts[-1] and (
                        self.search_criteria['selected_auxiliaries'] is None
                        or (name_parts[-1] not in self.search_criteria['selected_auxiliaries']
                        and name_parts[-1].replace("_0", "") not in self.search_criteria['selected_auxiliaries'])
                    ):
                        return
                    elif sanity_filters is not None:
                        for filter_dim in [k for k in sanity_filters.keys() if k.startswith("dim")]:
                            if obj.shape[int(filter_dim.split("dim")[-1])] != sanity_filters[filter_dim]:
                                logging.debug(f"UKBBImgH5: Shape insanity found for patient ID {name_parts[0]} while checking {name_parts[-1]}. Shape of this datum is {obj.shape}, excepted {sanity_filters[filter_dim]} for {filter_dim}. Skipping this patient.")
                                return
                    obj_shape = list(obj.shape)
                    if channel_dim not in [0, -1, None]: #a custom channel_dim has been provided and it's not the same as the default one
                        obj_shape[0], obj_shape[channel_dim] = obj_shape[channel_dim], obj_shape[0]
                    if slice_dim not in [2, -1, None]: #a custom slice_dim has been provided and it's not the same as the default one
                        obj_shape[2], obj_shape[slice_dim] = obj_shape[slice_dim], obj_shape[2]

                    self.DSPaths += [(name, (i, j, k))
                                    for i in (range(1) if multi_channel or obj_shape[0] == 1 else range(obj_shape[0]))
                                    for j in (range(1) if not split_time or obj_shape[1] == 1 else range(obj_shape[1]))
                                    for k in (range(1) if not split_volume or obj_shape[2] == 1 else range(obj_shape[2]))]

            h5_init.visititems(get_paths)
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            with open(processed_path, 'w') as f:
                json.dump(self.DSPaths, f)

        if data_mask_mode: #if data_mask_mode is not 0, then we need to fetch the mask - either from the data.h5 or meta_mask.h5
            key, _ = self.DSPaths[0]
            if key.replace(key.split("/")[-1], "meta_mask_0") in h5_init or key.replace(key.split("/")[-1], "meta_mask") in h5_init: #TODO - check this
                self.mask_in_datah5 = True
            elif os.path.isfile(self.h5_file.replace(os.path.basename(self.h5_file), "meta_mask.h5")):
                self.mask_in_datah5 = False
                self.mask_file = self.h5_file.replace(os.path.basename(self.h5_file), "meta_mask.h5")
            else:
                logging.error(f"UKBBImgH5: data_mask_mode is {data_mask_mode} but the mask is neither present inside the {os.path.basename(self.h5_file)}, not the seperate meta_mask.h5 is available inside the folder.")
                sys.exit(1)

    def __len__(self):
        return len(self.DSPaths)

    def __getitem__(self, idx):
        key, (ds_i, ds_j, ds_k) = self.DSPaths[idx]
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as h5_local:
            ds = h5_local[key]
            attrs = ds.attrs

            if self.do_transpose:
                ds = np.transpose(ds, self.transpose_dim_order)

            ds_slice = ds[(ds_i if not self.multi_channel else slice(None)), 
                            (ds_j if self.split_time else slice(None)), 
                            (ds_k if self.split_volume else slice(None)),...].copy()
            if self.squeeze_slice:
                ds_slice = ds_slice.squeeze()
            if not np.iscomplexobj(ds_slice):
                ds_slice = ds_slice.astype(np.float32)
            else:
                ds_slice = ds_slice.astype(np.complex64)

            sample = {
                "inp": {
                    "data": ds_slice,
                },
                "key": key,
                "indices": (ds_i, ds_j, ds_k),
            }
            
            if self.fetchMinMax and ("min_val" in attrs and "max_val" in attrs):
                minval, maxval = attrs['min_val'], attrs['max_val']
                if np.issubdtype(type(minval), np.integer):
                    minval = float(minval)
                if np.issubdtype(type(maxval), np.integer):
                    maxval = float(maxval)
                sample["inp"]["volmax"] = maxval
                sample["inp"]["volmin"] = minval

            if self.fetch_phenotypes:
                sample["phenotypes"] = self.phenotypes['data'][self.phenotypes['data'].index==int(sample['key'].split('/')[0])].values.squeeze().astype(np.float32)
            if self.fetch_confounders:
                sample["confounders_continuous"] = self.confounders['data'][self.confounders['data'].index==int(sample['key'].split('/')[0])][self.confounders['cont_cols']].values.squeeze().astype(np.float32)
                sample["confounders_binary"] = self.confounders['data'][self.confounders['data'].index==int(sample['key'].split('/')[0])][self.confounders['bincat_cols']].values.squeeze().astype(int)
                sample["confounders_multicat"] = self.confounders['data'][self.confounders['data'].index==int(sample['key'].split('/')[0])][self.confounders['mulcat_cols']].values.squeeze().astype(int)

            if self.data_mask_mode:
                if self.mask_in_datah5:
                    pass #TODO - implement this
                else:
                    with h5py.File(self.mask_file, 'r', libver='latest', swmr=True) as mask_local:
                        msk = mask_local[key]

                        if self.do_transpose:
                            msk = np.transpose(msk, self.transpose_dim_order)

                        msk_slice = msk[ 0, #mask is always single channel 
                                        (ds_j if self.split_time else slice(None)) if msk.shape[1] > 1 else 0, 
                                        (ds_k if self.split_volume else slice(None)) if msk.shape[2] > 1 else 0,...].copy().squeeze()
                if self.data_mask_mode == 1: #use masked data only
                    sample["inp"]["data"] = sample["inp"]["data"] * msk_slice
                else:
                    sample["inp"]["mask"] = msk_slice.astype(np.float32)

        if bool(self.transform):
            sample = self.transform(sample)

        return sample