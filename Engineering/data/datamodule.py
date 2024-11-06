import os
import contextlib
import random
import logging
from typing import Union
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lightning.pytorch as pl
import pandas as pd
from Engineering.data.datasets.ukbbImgH5 import UKBBImgH5

import torch
from torchvision import transforms

with contextlib.suppress(Exception):
    #This is an optional feature and the pipeline should be able to run without it.
    #Can be installed from https://github.com/iarai/concurrent-dataloader [or https://github.com/soumickmj/concurrent-dataloader]
    from concurrent_dataloader.dataloader_mod.dataloader import DataLoader as DataLoaderParallel
    from concurrent_dataloader.dataloader_mod.worker import _worker_loop as _worker_loop_parallel


class UKBBImgDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, data_file: str = "data.h5", foldCSV: str = "", foldID: int = 0, DSParams: dict = {},
                 transform=None, augmentations=None, batch_size: int = 32, num_workers: int = 0, data_prefetch: int = 2, n_save_recon_subs: int = 1, save_recon_subs:str = "",
                 use_concurrent_dataloader: bool = False):
        # data_dir: Path to the directory containing the UKBB H5 file, it must contain a file named "data.h5" or the name specified in data_file
        # foldCSV: Path to the CSV file containing the folds, it must be inside the data_dir
        # foldID: The fold ID to use for training, validation, and test
        # fetchMinMax: Whether to fetch the min and max values of the volumes directly from the dataset or to calculate on the fly for the individiual ones
        # transform: The transform to apply to the data TODO: add transforms for each of the splits
        # batch_size: The batch size to use for training, validation, and testing (This is the actual batchsize. The effective batchsize must be controlled from the main code)
        # num_workers: The number of workers to use for the dataloaders
        # data_prefetch: The number of batches to prefetch for the dataloaders by each worker.
        super().__init__()
        self.h5_file = f"{data_dir}/{data_file}"
        self.foldID = foldID
        self.DSParams = DSParams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_prefetch = data_prefetch
        self.n_save_recon_subs = n_save_recon_subs
        self.save_recon_subs = save_recon_subs
        self.use_concurrent_dataloader = use_concurrent_dataloader

        self.mergeTrainVal = self.DSParams.get("mergeTrainVal", False)
        with contextlib.suppress(Exception):
            del self.DSParams["mergeTrainVal"]

        if bool(foldCSV):
            self.folds = pd.read_csv(f"{data_dir}/{foldCSV}")
            filter_lst = []
            if 'fetch_phenotypes' in DSParams and DSParams['fetch_phenotypes']:
                filter_lst += list(pd.read_table(DSParams['phenotypes']['path']).FID)
            if 'fetch_confounders' in DSParams and DSParams['fetch_confounders']:
                filter_lst += list(pd.read_table(DSParams['confounders']['path']).FID)
            if filter_lst:
                for i in range(len(self.folds)):
                    for split in ["train", "val", "test"]:
                        lst = eval(self.folds[split][i])
                        filtered = str([x for x in lst if int(x) in filter_lst])
                        self.folds[split][i] = filtered
        else:
            self.folds = None

        if not bool(transform):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_transform = transforms.Compose([augmentations, transform]) if bool(augmentations) else transform
        self.val_transform = transform

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
            self.trainDS = UKBBImgH5(
                self.h5_file, transform=self.train_transform, patientIDs=train_pIDs, **self.DSParams)
            if not self.mergeTrainVal:
                self.valDS = UKBBImgH5(
                    self.h5_file, transform=self.val_transform, patientIDs=val_pIDs, **self.DSParams)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            DSParams_test = self.DSParams
            DSParams_test['fetch_phenotypes'] = False
            DSParams_test['fetch_confounders'] = False
            test_pIDs = self.folds.test[self.foldID] if self.folds is not None else None
            self.testDS = UKBBImgH5(
                self.h5_file, transform=self.val_transform, patientIDs=test_pIDs, **DSParams_test)
            test_pIDs = eval(test_pIDs) if bool(test_pIDs) else self.testDS.allPatientIDs #To make it easy to save the metadata during the creation of the dataset, we kept it as a string before.
            if bool(self.save_recon_subs):
                self.testDS.pIDs_save_recon = self.save_recon_subs.split(",")
            else:
                self.testDS.pIDs_save_recon = test_pIDs if self.n_save_recon_subs == 1 else random.sample(test_pIDs, int(len(test_pIDs)*self.n_save_recon_subs) if self.n_save_recon_subs < 1 else min(int(self.n_save_recon_subs), len(test_pIDs)))

        # Assign prediction dataset for use in dataloader(s). Predict is same as test, but doesn't care for folds and uses the whole dataset
        if stage == "predict":
            self.predictDS = UKBBImgH5(self.h5_file, transform=self.val_transform, **self.DSParams)

    def train_dataloader(self) -> DataLoader:
        if not self.use_concurrent_dataloader:
            return DataLoader(self.trainDS, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)
        dl = DataLoaderParallel(dataset=self.trainDS,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers//2, 
                                shuffle=True, 
                                prefetch_factor=self.data_prefetch, 
                                num_fetch_workers=self.num_workers//2, # parallel threads used to load data             
                                fetch_impl="threaded", # threaded | asyncio            
                                batch_pool=self.batch_size*2, # only for threaded implementation (pool of pre-loaded batches)             #TODO: parameterise this
                                pin_memory=False, # if using fork, it must be 0         
                                drop_last=True
                                )
        torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
        return dl

    def val_dataloader(self) -> DataLoader:        
        if not self.use_concurrent_dataloader:
            return DataLoader(self.valDS, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)
        dl = DataLoaderParallel(dataset=self.valDS,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers//2, 
                                shuffle=True, 
                                prefetch_factor=self.data_prefetch, 
                                num_fetch_workers=self.num_workers//2, # parallel threads used to load data             
                                fetch_impl="threaded", # threaded | asyncio            
                                batch_pool=self.batch_size*2, # only for threaded implementation (pool of pre-loaded batches)             
                                pin_memory=False, # if using fork, it must be 0         
                                drop_last=True
                                )
        torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
        return dl

    def test_dataloader(self) -> DataLoader:    
        if self.use_concurrent_dataloader:    
            logging.warning("Using concurrent dataloader during testing is currently not working (i.e. HDF5 saves are working). Using normal dataloader instead.")
        return DataLoader(self.testDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)      
        # if not self.use_concurrent_dataloader:
        #     return DataLoader(self.testDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)      
        # dl = DataLoaderParallel(dataset=self.testDS,
        #                         batch_size=self.batch_size,
        #                         num_workers=self.num_workers//2, 
        #                         shuffle=False, 
        #                         prefetch_factor=self.data_prefetch, 
        #                         num_fetch_workers=self.num_workers//2, # parallel threads used to load data             
        #                         fetch_impl="threaded", # threaded | asyncio            
        #                         batch_pool=self.batch_size*2, # only for threaded implementation (pool of pre-loaded batches)             
        #                         pin_memory=False, # if using fork, it must be 0  
        #                         )
        # torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
        # return dl

    def predict_dataloader(self) -> DataLoader: 
        if self.use_concurrent_dataloader:    
            logging.warning("Using concurrent dataloader during prediction is currently not working (i.e. HDF5 saves are working). Using normal dataloader instead.")
        return DataLoader(self.predictDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)     
        # if not self.use_concurrent_dataloader:
        #     return DataLoader(self.predictDS, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, prefetch_factor=self.data_prefetch)     
        # dl = DataLoaderParallel(dataset=self.predictDS,
        #                         batch_size=self.batch_size,
        #                         num_workers=self.num_workers//2, 
        #                         shuffle=False, 
        #                         prefetch_factor=self.data_prefetch, 
        #                         num_fetch_workers=self.num_workers//2, # parallel threads used to load data             
        #                         fetch_impl="threaded", # threaded | asyncio            
        #                         batch_pool=self.batch_size*2, # only for threaded implementation (pool of pre-loaded batches)             
        #                         pin_memory=False, # if using fork, it must be 0  
        #                         )
        # torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
        # return dl
