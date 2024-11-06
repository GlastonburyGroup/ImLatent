from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import json
import hashlib
import os
import logging

class UKBBImgZarr(Dataset):
    def __init__(self, zaar_root, transform=None, patientIDs=None, fetchMinMax=True, #These params were also present in the version for the V0 Dataset. fetch2D and expand_ch params were there in V0, have been removed in V1.
                        every_primary=True, selected_primaries=None, selected_auxiliaries=None, include_repeatAcq=True, combine_acquisitions=False, sanity_filters=None, #New params added for filtering the V1 dataset better
                        multi_channel=True, split_volume=True, split_time=True, #New params added for handlling the 5D data structure of the V1 dataset
                        channel_dim=0, slice_dim=2): #Params to make any dim as the channel and slice.
        self.zaar_root = zaar_root
        self.transform = transform
        self.zarr_group = zarr.open(zaar_root, 'r')
        self.fetchMinMax = fetchMinMax

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

        self.search_criteria = {
            'patientIDs': patientIDs if bool(patientIDs) else list(self.zarr_group.keys()),
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
        processed_path = zaar_root.replace(".zarr", f"_processedmeta/{processed_name}_MD5{md5_patients}_SHA3{sha_patients}.json")
        if os.path.exists(processed_path):
            with open(processed_path, 'r') as f:
                self.DSPaths = json.load(f)
        else:    

            def visititems(group, func, path=''):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, zarr.hierarchy.Group):
                        visititems(item, func, path + key + '/')
                    else:
                        func(path + key, item)  

            self.DSPaths = []
            def get_paths(name, obj):
                name_parts = name.split('/')
                #TODO: replace the current search with regex-based search
                if isinstance(obj, zarr.core.Array) and name_parts[0] in self.search_criteria['patientIDs']:
                    if 'primary' not in name_parts[-1] and 'auxiliary' not in name_parts[-1]:
                        return
                    elif (
                        'primary' in name_parts[-1]
                        and not every_primary
                        and (
                            self.search_criteria['selected_primaries'] is None 
                            or name_parts[-1].replace("_0", "")
                            not in self.search_criteria['selected_primaries']
                        )
                    ):
                        return
                    elif 'auxiliary' in name_parts[-1] and (
                        self.search_criteria['selected_auxiliaries'] is None
                        or name_parts[-1].replace("_0", "")
                        not in self.search_criteria['selected_auxiliaries']
                    ):
                        return
                    elif sanity_filters is not None:
                        for filter_dim in [k for k in sanity_filters.keys() if k.startswith("dim")]:
                            if obj.shape[int(filter_dim.split("dim")[-1])] != sanity_filters[filter_dim]:
                                logging.debug(f"UKBBImgZarr: Shape insanity found for patient ID {name_parts[0]} while checking {name_parts[-1]}. Shape of this datum is {obj.shape}, excepted {sanity_filters[filter_dim]} for {filter_dim}. Skipping this patient.")
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

            visititems(self.zarr_group, get_paths)
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            with open(processed_path, 'w') as f:
                json.dump(self.DSPaths, f)

    def __len__(self):
        return len(self.DSPaths)

    def __getitem__(self, idx):
        key = self.DSPaths[idx]
        key, (ds_i, ds_j, ds_k) = self.DSPaths[idx]
        ds = self.zarr_group[key]
        attrs = ds.attrs.asdict()

        if self.do_transpose:
            ds = np.transpose(ds, self.transpose_dim_order)

        ds_slice = ds[(ds_i if not self.multi_channel else slice(None)), 
                        (ds_j if self.split_time else slice(None)), 
                        (ds_k if self.split_volume else slice(None)),...].squeeze()
        sample = {
            "inp": {
                "data": ds_slice.astype(np.float32),
            },
            "key": key,
            "indices": (ds_i, ds_j, ds_k),
        }
        
        if self.fetchMinMax and ("min_val" in attrs and "max_val" in attrs):
            minval, maxval = attrs['min_val'], attrs['max_val']
            sample["inp"]["volmax"] = maxval
            sample["inp"]["volmin"] = minval

        if bool(self.transform):
            sample = self.transform(sample)

        return sample
