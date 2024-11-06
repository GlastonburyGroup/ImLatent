import os
import contextlib
import random
import sys
import logging
from typing import Union
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning.pytorch as pl
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torchvision import transforms

def h5_to_dict(h5_obj):
    """Convert an HDF5 object into a dictionary."""
    if isinstance(h5_obj, h5py.Dataset):
        return h5_obj[()]
    elif isinstance(h5_obj, h5py.Group):
        return {key: h5_to_dict(val) for key, val in h5_obj.items()}
    else:
        raise TypeError(f"Unsupported type: {type(h5_obj)}")
    
def flatten_dict(d, parent_key='', sep='/'):
    """Flatten a nested dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items |= flatten_dict(v, new_key, sep=sep)
        else:
            items[new_key] = [v]
    return items
    
def count_DSs(d):
    """Count the number of numpy arrays (originally, datasets) in a nested dictionary."""
    if isinstance(d, dict):
        return sum(count_DSs(val) for val in d.values())
    elif isinstance(d, np.ndarray):
        return 1
    else:
        return 0

class LatentDataset(Dataset):
    def __init__(self, emb_H5, phenotypes=None, confounders=None, patientIDs=None, **kwargs):
        super().__init__()
        
        with h5py.File(emb_H5, 'r', libver='latest', swmr=True) as f:
            data_dict = h5_to_dict(f)
        if bool(patientIDs):
            data_dict = {k: v for k, v in data_dict.items() if k in patientIDs}
            patientIDs = [int(p) for p in eval(patientIDs)]
            
        flat_data = flatten_dict(data_dict)
        data = pd.DataFrame.from_dict(flat_data, orient='index').reset_index()
        data = data.rename(columns={0: "latent"})
        df_splitcol = data['index'].str.split('/', expand=True)
        df_splitcol.columns = ["subID", "fieldID", "instanceID", "dsID"]
        data = data.drop('index', axis=1)
        self.data = pd.concat([data, df_splitcol], axis=1)
        self.data.subID = self.data.subID.astype(int)
        
        all_values = np.concatenate(self.data['latent'].to_list())
        self.latent_mean = np.mean(all_values)
        self.latent_std = np.std(all_values)
        self.data['latent'] = self.data['latent'].apply(lambda arr: (arr - self.latent_mean) / self.latent_std)
        
        if bool(phenotypes):
            try:
                self.phenotypes = pd.read_table(phenotypes['path'], index_col="FID")
            except:
                self.phenotypes = pd.read_table(phenotypes['path'], index_col="f.eid")
                self.phenotypes.rename_axis("FID", axis='index', inplace=True)
            if bool(patientIDs):
                self.phenotypes = self.phenotypes.loc[patientIDs]
            if bool(phenotypes['cols']):
                if isinstance(phenotypes['cols'], str):
                    phenotypes['cols'] = phenotypes['cols'].split(',')
                self.phenotypes = self.phenotypes[phenotypes['cols']]
            scaler = StandardScaler()
            self.phenotypes_scaler_mean = {}
            self.phenotypes_scaler_scale = {}
            for column in self.phenotypes.columns:
                self.phenotypes[column] = scaler.fit_transform(self.phenotypes[[column]])
                self.phenotypes_scaler_mean[column] = scaler.mean_
                self.phenotypes_scaler_scale[column] = scaler.scale_
            self.data = self.data.merge(self.phenotypes, left_on="subID", right_index=True)
        if bool(confounders):
            print("Confounder file not yet implemented! Will be ignored!")
            #try:
                # self.confounders = pd.read_table(confounders['path'], index="FID")
            #except:
                # self.confounders = pd.read_table(confounders['path'], index="f.eid")
                # self.confounders.rename_axis("FID", axis='index', inplace=True)
            # if bool(patientIDs):
            #     self.phenotypes = self.confounders.loc[patientIDs]
            # if bool(confounders['cols']):
            #     self.confounders = self.confounders[confounders['cols']]
            # self.data = self.data.merge(self.confounders, left_on="subID", right_index=True)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datum = self.data.iloc[idx]
        return {
            "latent": torch.tensor(datum.latent.squeeze(), dtype=torch.float32),
            "phenotypes": torch.tensor(datum[self.phenotypes.columns], dtype=torch.float32),
            "subID": datum.subID,
            "fieldID": datum.fieldID,
            "instanceID": datum.instanceID,
            "dsID": datum.dsID
        }

class LatentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, foldCSV: str = "", foldID: int = 0, DSParams: dict = {},
                 transform=None, augmentations=None, batch_size: int = 32, num_workers: int = 0, data_prefetch: int = 2, n_save_recon_subs: int = 1, save_recon_subs:str = "",
                 use_concurrent_dataloader: bool = False):
        super().__init__()
        self.h5_file = f"{data_dir}/emb.h5"
        self.foldID = foldID
        self.DSParams = DSParams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_prefetch = data_prefetch
        self.n_save_recon_subs = n_save_recon_subs
        self.save_recon_subs = save_recon_subs
        self.use_concurrent_dataloader = use_concurrent_dataloader

        self.mergeTrainVal = self.DSParams["mergeTrainVal"]
        del self.DSParams["mergeTrainVal"]

        if bool(foldCSV):
            self.folds = pd.read_csv(foldCSV)
            filter_lst = []
            if 'fetch_phenotypes' in DSParams and DSParams['fetch_phenotypes']:
                try:
                    filter_lst += list(pd.read_table(DSParams['phenotypes']['path']).FID)
                except:
                    filter_lst += list(pd.read_table(DSParams['phenotypes']['path'])["f.eid"])
            if 'fetch_confounders' in DSParams and DSParams['fetch_confounders']:
                try:
                    filter_lst += list(pd.read_table(DSParams['confounders']['path']).FID)
                except:
                    filter_lst += list(pd.read_table(DSParams['confounders']['path'])["f.eid"])
            if filter_lst:
                for i in range(len(self.folds)):
                    for split in ["train", "val", "test"]:
                        lst = eval(self.folds[split][i])
                        filtered = str([x for x in lst if int(x) in filter_lst])
                        self.folds[split][i] = filtered
        else:
            self.folds = None

    def prepare_data(self):
        pass  # we currently don't need to download or prepare anything

    def setup(self, stage: str):  # sourcery skip: switch
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if self.folds is not None:
                train_pIDs = self.folds.train[self.foldID]
                val_pIDs = self.folds.val[self.foldID]
                if self.mergeTrainVal:
                    train_pIDs += val_pIDs
                    val_pIDs = None
            else:
                train_pIDs = None
                val_pIDs = None
            self.trainDS = LatentDataset(self.h5_file, patientIDs=train_pIDs, **self.DSParams)
            if not self.mergeTrainVal:
                self.valDS = LatentDataset(self.h5_file, patientIDs=val_pIDs, **self.DSParams)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            DSParams_test = self.DSParams
            DSParams_test['fetch_phenotypes'] = False
            DSParams_test['fetch_confounders'] = False
            test_pIDs = self.folds.test[self.foldID] if self.folds is not None else None
            self.testDS = LatentDataset(self.h5_file, patientIDs=test_pIDs, **DSParams_test)
            test_pIDs = eval(test_pIDs) if bool(test_pIDs) else self.testDS.allPatientIDs #To make it easy to save the metadata during the creation of the dataset, we kept it as a string before.

        # Assign prediction dataset for use in dataloader(s). Predict is same as test, but doesn't care for folds and uses the whole dataset
        if stage == "predict":
            self.predictDS = LatentDataset(self.h5_file, **self.DSParams)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainDS, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)

    def val_dataloader(self) -> DataLoader:        
        return DataLoader(self.valDS, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)

    def test_dataloader(self) -> DataLoader:  
        return DataLoader(self.testDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)     

    def predict_dataloader(self) -> DataLoader: 
        return DataLoader(self.predictDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)    