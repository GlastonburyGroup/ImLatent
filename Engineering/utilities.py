import argparse
import contextlib
import os
import re
from statistics import median
import pandas as pd
import logging
from collections import OrderedDict, defaultdict
from typing import Any, Tuple
import h5py
import ast
from multiprocessing import Pool, cpu_count

import importlib
import nibabel as nib
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import torch
import torchcomplex
import torchio as tio
import torchvision.utils as vutils
from async_timeout import sys
from sewar.full_ref import uqi as UQICalc, msssim as MSSSIMCalc
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)

from Engineering.Science.freq_trans import fftNc, ifftNc
from Engineering.Science.misc import minmax
from Engineering.transforms.tio.transforms import getDataSpaceTransforms

import sys
sys.excepthook

def BoolArgs(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Can be supplied as: ['yes', 'true', 't', 'y', '1'] or ['no', 'false', 'f', 'n', '0']")
    
def convert_to_int_float_bool(v):
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        try: 
            return BoolArgs(v)
        except argparse.ArgumentTypeError:
            return v

def process_unknown_args(unknown_args):
    unknown_args_dict = {}
    prev_arg = None
    for arg in unknown_args:
        # If the previous argument had no value, then this argument must be the value
        if prev_arg:
            if arg.startswith('-'): #the previous one was a boolean flag
                unknown_args_dict[prev_arg.lstrip('no-')] = not prev_arg.startswith('no-')
                prev_arg = None
            else:
                unknown_args_dict[prev_arg] = convert_to_int_float_bool(arg)
                prev_arg = None
                continue
            
        # Split the argument by the equals sign
        parts = arg.split('=')
        if len(parts) == 2:
            # This arg is a key-value pair
            key, value = parts
            unknown_args_dict[key.lstrip('-')] = convert_to_int_float_bool(value)
        else:
            # This is a flag or a key with no value
            prev_arg = arg.lstrip('-')

    if prev_arg: #If still a previous argument left, then it must be a boolean flag
        unknown_args_dict[prev_arg.lstrip('no-')] = not prev_arg.startswith('no-')
        
    return unknown_args_dict

def string2dict(str):
    try:
        return ast.literal_eval(str)
    except ValueError:
        pattern = r"(\w+):([^,}]+)" 
        matches = re.findall(pattern, str)

        dictionary = {}
        for k, v in matches:
            if v.isdigit():
                v = int(v)
            else:
                with contextlib.suppress(ValueError):
                    v = float(v)
            dictionary[k] = v
        return dictionary

def dict_deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            # get node or create one
            node = source.setdefault(key, {})
            dict_deep_update(node, value)
        else:
            source[key] = value

    return source

# function to import a function from a package using a string
def import_func_str(package_name, function_name):
    module = importlib.import_module(package_name)
    return getattr(module, function_name)

# a wrapper on the getattr function, as it does not work with nested attributes 
# For example, cannot use model.encoder to fetch this from the net object, rather getattr(getattr(net, 'model'), 'encoder') has to be used!
# This function acts as a drop-in replacement for the getattr function
def get_nested_attribute(obj, attribute_name):
    attribute_names = attribute_name.split('.')
    attribute_value = obj
    for name in attribute_names:
        attribute_value = getattr(attribute_value, name)
    return attribute_value

# overrides the default exception hook to log exceptions 
def hook_exception_log(*args):
  logging.getLogger().error('\nUncaught Exception:\n\n', exc_info=args, stack_info=True) 
  sys.__excepthook__(*args)

def get_SLURM_envs():
    prefix = "SLURM"
    pattern = re.compile(r'{prefix}\w+'.format(prefix=prefix))
    envvars = {key: val for key, val in os.environ.items() if pattern.match(key)}
    envvars['COMPUTED_MEM_PER_TASK'] = int(envvars['SLURM_MEM_PER_CPU']) * int(envvars['SLURM_CPUS_PER_TASK'])
    envvars['COMPUTED_MEM_PER_NODE'] = int(envvars['SLURM_MEM_PER_CPU']) * int(envvars['SLURM_JOB_CPUS_PER_NODE'].split("(x")[0]) #.split("(x")[0] is required as for multi-node SLURM, the value will be something like 16(x2)
    return envvars

class PlasmaConduit(OrderedDict):
    """This class is used to transport structured data, like the output from the model, from place to place. 
    This class is inspired from the ``ModelOutput`` class from hugginface transformers library"""

    def pop(self, key, default=-1):
        '''od.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.  If key is not found, d is returned if given, otherwise KeyError
        is raised.

        '''
        if key in self:
            result = self[key]
            del self[key]
            return result
        return default

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

def filter_dict_keys(dictionary, keycontains="", keyprefix="", keysuffix=""):
    filtered_dict = {f"{keyprefix}{key}{keysuffix}": value for key, value in dictionary.items() if keycontains in key}
    return filtered_dict

def sitkShow(data, slice_last=True):
    if issubclass(type(data), tio.Image):
        data = data['data']
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = data.squeeze()
    if slice_last and len(data.shape) == 3:
        data = np.transpose(data)
    img = sitk.GetImageFromArray(data)
    sitk.Show(img)


def getSSIM(gt, out, gt_flag=None, data_range=1):
    if gt_flag is None:  # all of the samples have GTs
        gt_flag = [True]*gt.shape[0]

    vals = []
    for i in range(gt.shape[0]):
        if not gt_flag[i]:
            continue
        vals.extend(
            structural_similarity(
                gt[i, j, ...], out[i, j, ...], data_range=data_range
            )
            for j in range(gt.shape[1])
        )
    return median(vals)

def pearson_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2)) + np.finfo(float).eps
    correlation = numerator / denominator
    return correlation


def calc_metircs(gt, out, tag, norm4diff=False):
    if gt.min() == 0 and gt.max() == 0: #blank image, skip calculating metrics
        return None, None, None
    ssim, ssimMAP = structural_similarity(gt, out, data_range=1, full=True)
    mssim = MSSSIMCalc(gt, out, MAX=1)
    if not np.iscomplexobj(gt) and not np.iscomplexobj(out) and np.iscomplexobj(mssim):
        mssim = mssim.real
    nrmse = normalized_root_mse(gt, out)
    psnr = peak_signal_noise_ratio(gt, out, data_range=1)
    uqi = UQICalc(gt, out)
    if norm4diff:
        gt = minmax(gt)
        out = minmax(out)
    diff = gt - out
    dif_std = np.std(diff)
    metrics = {
        "SSIM"+tag: ssim,
        "MSSSIM"+tag: mssim,
        "NRMSE"+tag: nrmse,
        "PSNR"+tag: psnr,
        "UQI"+tag: uqi,
        "SDofDiff"+tag: dif_std
    }
    return metrics, ssimMAP, abs(diff)

class Evaluator:
    def __init__(self, is3D, norm4diff=False, store_fullscale_evals=False, n_calc_processes=0, metrics2use=None):
        self.is3D = is3D
        self.norm4diff = norm4diff
        self.store_fullscale_evals = store_fullscale_evals

        if n_calc_processes == 0:
            self.get_batch_scores = self.get_batch_scores_single_process
        elif n_calc_processes == -1:
            self.n_calc_processes = cpu_count()
        else:
            self.n_calc_processes = n_calc_processes

        if metrics2use is None:
            self.metrics2use = {
                "ssim": True,
                "msssim": True,
                "nrmse": True,
                "psnr": True,
                "uqi": True,
                "sddiff": True,
            }
        else:
            self.metrics2use = metrics2use

    def calc_metrics(self, gt, out, tag):
        if gt.min() == 0 and gt.max() == 0: #blank image, skip calculating metrics
            return None, None, None
        ssim, ssimMAP = structural_similarity(gt, out, data_range=1, full=True) if self.metrics2use["ssim"] else (-1, None)
        if self.metrics2use["msssim"]:
            mssim = MSSSIMCalc(gt, out, MAX=1)
            if not np.iscomplexobj(gt) and not np.iscomplexobj(out) and np.iscomplexobj(mssim):
                mssim = mssim.real
        else:
            mssim = -1
        nrmse = normalized_root_mse(gt, out) if self.metrics2use["nrmse"] else -1
        psnr = peak_signal_noise_ratio(gt, out, data_range=1) if self.metrics2use["psnr"] else -1
        uqi = UQICalc(gt, out) if self.metrics2use["uqi"] else -1
        if self.metrics2use["sddiff"]:
            if self.norm4diff:
                gt = minmax(gt)
                out = minmax(out)
            diff = gt - out
            dif_std = np.std(diff)
        else:
            diff = None
            dif_std = -1
        metrics = {
            "SSIM"+tag: ssim,
            "MSSSIM"+tag: mssim,
            "NRMSE"+tag: nrmse,
            "PSNR"+tag: psnr,
            "UQI"+tag: uqi,
            "SDofDiff"+tag: dif_std
        }
        return metrics, ssimMAP, abs(diff) if diff is not None else None

    def get_single_score(self, args):
        gt, out, tag = args
        if not np.iscomplexobj(gt):
            return self.calc_metrics(gt, out, tag)
        else:
            scores, ssimMAP, diff = self.calc_metrics(abs(gt), abs(out), f"{tag}_mag")
            scores_phase, _, _ = self.calc_metrics(np.angle(gt), np.angle(out), f"{tag}_phase")
            scores.update(scores_phase)
            scores.update({
                f"Pearson{tag}_mag": pearson_correlation(abs(gt), abs(out)),
                f"Pearson{tag}_phase": pearson_correlation(np.angle(gt), np.angle(out)),
            })
            return scores, ssimMAP, abs(diff) if diff is not None else None
    
    def get_batch_scores_single_process(self, gt, out, keys, tag):
        res = []
        for i in range(gt.shape[0]): #batch
            for j in range(gt.shape[1]): #channel
                if self.is3D:
                    for k in range(gt.shape[2]): #slice
                        if not np.iscomplexobj(gt):
                            metrics, ssimMAP, diff = self.calc_metrics(gt[i,j,k], out[i,j,k], tag)
                        else:
                            metrics, ssimMAP, diff = self.calc_metrics(abs(gt[i,j,k]), abs(out[i,j,k]), f"{tag}_mag")
                            metrics_phase, _, _ = self.calc_metrics(np.angle(gt[i,j,k]), np.angle(out[i,j,k]), f"{tag}_phase")
                            metrics.update(metrics_phase)
                            metrics.update({
                                f"Pearson{tag}_mag": pearson_correlation(abs(gt[i,j,k]), abs(out[i,j,k])),
                                f"Pearson{tag}_phase": pearson_correlation(np.angle(gt[i,j,k]), np.angle(out[i,j,k])),
                            })
                        if metrics is None:
                            continue
                        datum = {
                            "key": keys[i],
                            "Channel": j,
                            "Slice": k,
                            **metrics,
                        }
                        if self.store_fullscale_evals:
                            datum["SSIMMAP"+tag] = ssimMAP
                            datum["Diff"+tag] = diff
                        res.append(datum)
                else:
                    if not np.iscomplexobj(gt):
                        metrics, ssimMAP, diff = self.calc_metrics(gt[i,j], out[i,j], tag)
                    else:
                        metrics, ssimMAP, diff = self.calc_metrics(abs(gt[i,j]), abs(out[i,j]), f"{tag}_mag")
                        metrics_phase, _, _ = self.calc_metrics(np.angle(gt[i,j]), np.angle(out[i,j]), f"{tag}_phase")
                        metrics.update(metrics_phase)
                        metrics.update({
                            f"Pearson{tag}_mag": pearson_correlation(abs(gt[i,j]), abs(out[i,j])),
                            f"Pearson{tag}_phase": pearson_correlation(np.angle(gt[i,j]), np.angle(out[i,j])),
                        })
                    if metrics is None:
                        continue
                    datum = {
                        "key": keys[i],
                        "Channel": j,
                        **metrics,
                    }
                    if self.store_fullscale_evals:
                        datum["SSIMMAP"+tag] = ssimMAP
                        datum["Diff"+tag] = diff
                    res.append(datum)
        return res

    def get_batch_scores(self, gt, out, keys, tag):
        res = []
        pool = Pool(self.n_calc_processes)  
        
        for i in range(gt.shape[0]):  # batch
            for j in range(gt.shape[1]):  # channel
                if self.is3D:
                    for k in range(gt.shape[2]):  # slice
                        args = (gt[i, j, k], out[i, j, k], tag)
                        res.append(pool.apply_async(self.get_single_score, (args,)))  # Apply the function asynchronously
                else:
                    args = (gt[i, j], out[i, j], tag,)
                    res.append(pool.apply_async(self.get_single_score, (args,)))  # Apply the function asynchronously

        pool.close()
        pool.join()

        # Retrieve the results from async processes
        res = [result.get() for result in res]
        res = [i for i in res if i!= (None, None, None)]

        # Organize the results into the desired format
        for i, result in enumerate(res):
            batch_index = i // (gt.shape[1] * gt.shape[2]) if self.is3D else i // gt.shape[1]
            channel_index = (i % (gt.shape[1] * gt.shape[2])) // gt.shape[2] if self.is3D else i % gt.shape[1]
            slice_index = i % gt.shape[2] if self.is3D else None
            
            datum = {
                "key": keys[batch_index],
                "Channel": channel_index,
                **result[0],
            }
            if slice_index is not None:
                datum["Slice"] = slice_index
                
            if self.store_fullscale_evals:
                datum["SSIMMAP" + tag] = result[1]
                datum["Diff" + tag] = result[2]
            res[i] = datum

        return res

def get_batch_scores_single_process(gt, out, keys, tag, is3D, norm4diff=False, store_fullscale_evals=False):
    res = []
    for i in range(gt.shape[0]): #batch
        for j in range(gt.shape[1]): #channel
            if is3D:
                for k in range(gt.shape[2]): #slice
                    metrics, ssimMAP, diff = calc_metircs(gt[i,j,k], out[i,j,k], tag, norm4diff)
                    if metrics is None:
                        continue
                    datum = {
                        "key": keys[i],
                        "Channel": j,
                        "Slice": k,
                        **metrics,
                    }
                    if store_fullscale_evals:
                        datum["SSIMMAP"+tag] = ssimMAP
                        datum["Diff"+tag] = diff
                    res.append(datum)
            else:
                metrics, ssimMAP, diff = calc_metircs(gt[i,j], out[i,j], tag, norm4diff)
                if metrics is None:
                    continue
                datum = {
                    "key": keys[i],
                    "Channel": j,
                    **metrics,
                }
                if store_fullscale_evals:
                    datum["SSIMMAP"+tag] = ssimMAP
                    datum["Diff"+tag] = diff
                res.append(datum)
    return res

def process_slice(args):
    gt, out, tag, norm4diff = args
    return calc_metircs(gt, out, tag, norm4diff)

def log_images(writer, inputs, outputs, targets, step, section='', imID=0, chID=0):
    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[imID, chID, ...],
                                      normalize=True,
                                      scale_each=True),
                     step)
    if inputs is not None:
        writer.add_image('{}/input'.format(section),
                         vutils.make_grid(inputs[imID, chID, ...],
                                          normalize=True,
                                          scale_each=True),
                         step)
    if targets is not None:
        writer.add_image('{}/target'.format(section),
                         vutils.make_grid(targets[imID, chID, ...],
                                          normalize=True,
                                          scale_each=True),
                         step)


def ReadNIFTI(file_path):
    """Read a NIFTI file using given file path to an array
    Using: NiBabel"""
    nii = nib.load(file_path)
    return np.array(nii.get_fdata())


def SaveNIFTI(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)


class DataSpaceHandler:
    def __init__(self, **kwargs) -> None:
        self.dataspace_inp = kwargs['dataspace_inp']
        self.model_dataspace_inp = kwargs['model_dataspace_inp']
        self.dataspace_gt = kwargs['dataspace_gt']
        self.model_dataspace_gt = kwargs['model_dataspace_gt']
        self.dataspace_out = kwargs['dataspace_out']
        self.model_dataspace_out = kwargs['model_dataspace_out']
        self.data_dim = kwargs['inplane_dims']
        self.fftnorm = kwargs['fftnorm']

    def getTransforms(self):
        return getDataSpaceTransforms(self.dataspace_inp, self.model_dataspace_inp, self.dataspace_gt, self.model_dataspace_gt)


class DataHandler:
    def __init__(self, dataspace_op: DataSpaceHandler, inp=None, gt=None, out=None, metadict=None, storeAsTensor=True):
        self.dataspace_op = dataspace_op
        self.storeAsTensor = storeAsTensor
        self.inp = self.__convert_type(inp)
        self.gt = self.__convert_type(gt)
        self.out = self.__convert_type(out)
        self.metadict = metadict
        self.inpK = None
        self.gtK = None
        self.outK = None
        self.outCorrectedK = None

    def __convert_type(self, x):
        if x is None or self.storeAsTensor == torch.is_tensor(x):
            return x
        elif self.storeAsTensor and not torch.is_tensor(x):
            return torch.from_numpy(x)
        else:
            return x.numpy()

    def setInpK(self, x):
        self.inpK = self.__convert_type(x)

    def setGTK(self, x):
        self.gtK = self.__convert_type(x)

    def setOutK(self, x):
        self.outK = self.__convert_type(x)

    def setOutCorrectedK(self, x):
        self.outCorrectedK = self.__convert_type(x)

    # Get kspace

    def __getK(self, x, dataspace, k, imnorm=False):
        if k is not None:
            return k
        elif dataspace == 1 or x is None:
            return x
        else:
            return fftNc(data=x if not imnorm else x/x.max(), dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    def getKInp(self, imnorm=False):
        return self.__getK(self.inp, self.dataspace_op.model_dataspace_inp, self.inpK, imnorm)

    def getKGT(self, imnorm=False):
        return self.__getK(self.gt, self.dataspace_op.model_dataspace_gt, self.gtK, imnorm)

    def getKOut(self, imnorm=False):
        return self.__getK(self.out, self.dataspace_op.dataspace_out, self.outK, imnorm)

    def getKOutCorrected(self):
        return self.outCorrectedK

    # Get Image space

    def __getIm(self, x, dataspace):
        if dataspace == 0 or x is None:
            return x
        else:
            return ifftNc(data=x, dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    def getImInp(self):
        return self.__getIm(self.inp, self.dataspace_op.model_dataspace_inp)

    def getImGT(self):
        return self.__getIm(self.gt, self.dataspace_op.model_dataspace_gt)

    def getImOut(self):
        return self.__getIm(self.out, self.dataspace_op.dataspace_out)

    def getImOutCorrected(self):
        if self.outCorrectedK is None:
            return None
        else:
            return ifftNc(data=self.outCorrectedK, dim=self.dataspace_op.data_dim, norm=self.dataspace_op.fftnorm)

    # all combo

    def getKAll(self):
        return (self.getKInp(), self.getKGT(), self.getKOut())

    def getImgAll(self):
        return (self.getImInp(), self.getImGT(), self.getImOut())

class ResHandler(defaultdict):
    #This class acts as defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) 
    # #for storing results as: H5 key -> i (Channels) -> j (Time) -> k (Slice)

    def __init__(self, save_recon=False, pIDs_save_recon=None, ds_flags=None):
        super().__init__(lambda: ResHandler())
        self.save_recon = save_recon
        self.pIDs_save_recon = str(pIDs_save_recon) #TODO: check if it holds always, a trick to work with different types of data
        self.ds_flags = ds_flags
    
    def __missing__(self, key):
        value = self[key] = self.default_factory()
        return value
    
    def store_res(self, batch, emb, recon=None):
        if not self.save_recon:
            for key, idx_0, idx_1, idx_2, em in zip(batch['key'], batch['indices'][0], batch['indices'][1], batch['indices'][2], emb):
                self[key][idx_0.item()][idx_1.item()][idx_2.item()] = em.tolist()
        else:
            for key, idx_0, idx_1, idx_2, em, r in zip(batch['key'], batch['indices'][0], batch['indices'][1], batch['indices'][2], emb, recon):
                if key.split("/")[0] in self.pIDs_save_recon:
                    self[key][idx_0.item()][idx_1.item()][idx_2.item()] = (em.tolist(), r.squeeze())
                else:
                    self[key][idx_0.item()][idx_1.item()][idx_2.item()] = em.tolist()
    
    def to_array(self):
        res_store = {}
        for key, sub_dict in self.items():
            max_i = max(sub_dict.keys())
            max_j = max(max(sub_dict[i].keys()) for i in sub_dict.keys())
            max_k = max(max(sub_dict[i][j].keys()) for i in sub_dict.keys() for j in sub_dict[i].keys())
            if not self.save_recon or not isinstance(sub_dict[0][0][0], tuple):
                emb = np.full((max_i+1, max_j+1, max_k+1, len(sub_dict[0][0][0])), np.nan)
                if np.iscomplexobj(sub_dict[0][0][0]):
                    emb = emb.astype(np.complex64)
                save_recon_sub = False
            else:
                emb = np.full((max_i+1, max_j+1, max_k+1, len(sub_dict[0][0][0][0])), np.nan)
                if np.iscomplexobj(sub_dict[0][0][0][0]):
                    emb = emb.astype(np.complex64)
                recon = np.full((max_i+1, max_j+1, max_k+1, *sub_dict[0][0][0][1].shape), np.nan)
                if np.iscomplexobj(sub_dict[0][0][0][1]):
                    recon = recon.astype(np.complex64)
                save_recon_sub = True
            for i, i_dict in sub_dict.items():
                for j, j_dict in i_dict.items():
                    for k, k_dict in j_dict.items():
                        if not save_recon_sub:
                            emb[i, j, k] = k_dict 
                        else:
                            emb[i, j, k], recon[i, j, k] = k_dict
            res_store[key] = emb if not save_recon_sub else (emb, recon)
        return res_store
    
    def __getstate__(self):
        return (self.save_recon, dict(self))
    
    def __setstate__(self, state):
        self.save_recon = state[0]
        self.update(state[1])
    
    def __reduce__(self):
        return (ResHandler, (self.save_recon,), self.__getstate__())

def Res2H5(results, output_path, save_recon=False):
    if save_recon:
        with h5py.File(f"{output_path}/recon.h5", 'w') as f:
            for key, val in results.items():
                if isinstance(val, tuple):
                    patientID, fieldID, instanceID, dsName = key.split("/")
                    group_path = f"{patientID}/{fieldID}/{instanceID}"
                    g = f[group_path] if group_path in f else f.create_group(group_path)
                    _, val = val
                    g.create_dataset(f'{dsName}', data=val)    
    with h5py.File(f"{output_path}/emb.h5", 'w') as f:
        for key, val in results.items():
            patientID, fieldID, instanceID, dsName = key.split("/")
            group_path = f"{patientID}/{fieldID}/{instanceID}"
            g = f[group_path] if group_path in f else f.create_group(group_path)
            if isinstance(val, tuple):
                val, _ = val
            g.create_dataset(f'{dsName}', data=val)

class AdditionalResHandler(defaultdict):
    #This class acts as defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) 
    # #for storing results as: H5 key -> i (Channels) -> j (Time) -> k (Slice)

    def __init__(self, ds_flags=None):
        super().__init__(lambda: AdditionalResHandler())
        self.ds_flags = ds_flags
    
    def __missing__(self, key):
        value = self[key] = self.default_factory()
        return value
    
    def store_res(self, batch, data, tag):
        for key, idx_0, idx_1, idx_2, d in zip(batch['key'], batch['indices'][0], batch['indices'][1], batch['indices'][2], data):
            self[tag][key][idx_0.item()][idx_1.item()][idx_2.item()] = d.tolist()
    
    def to_array(self, tag):
        res_store = {}
        for key, sub_dict in self[tag].items():
            max_i = max(sub_dict.keys())
            max_j = max(max(sub_dict[i].keys()) for i in sub_dict.keys())
            max_k = max(max(sub_dict[i][j].keys()) for i in sub_dict.keys() for j in sub_dict[i].keys())
            data = np.full((max_i+1, max_j+1, max_k+1, len(sub_dict[0][0][0])), np.nan)
            for i, i_dict in sub_dict.items():
                for j, j_dict in i_dict.items():
                    for k, k_dict in j_dict.items():
                        data[i, j, k] = k_dict 
            res_store[key] = data 
        return res_store
    
    def __getstate__(self):
        return (self.ds_flags, dict(self))
    
    def __setstate__(self, state):
        self.ds_flags = state[0]
        self.update(state[1])
    
    def __reduce__(self):
        return (ResHandler, (self.ds_flags,), self.__getstate__())

def Res2H5(results, output_path, save_recon=False, additional={}):
    if save_recon:
        with h5py.File(f"{output_path}/recon.h5", 'w') as f:
            for key, val in results.items():
                if isinstance(val, tuple):
                    patientID, fieldID, instanceID, dsName = key.split("/")
                    group_path = f"{patientID}/{fieldID}/{instanceID}"
                    g = f[group_path] if group_path in f else f.create_group(group_path)
                    _, val = val
                    g.create_dataset(f'{dsName}', data=val)    
    with h5py.File(f"{output_path}/emb.h5", 'w') as f:
        for key, val in results.items():
            patientID, fieldID, instanceID, dsName = key.split("/")
            group_path = f"{patientID}/{fieldID}/{instanceID}"
            g = f[group_path] if group_path in f else f.create_group(group_path)
            if isinstance(val, tuple):
                val, _ = val
            g.create_dataset(f'{dsName}', data=val)
    if additional:
        for tag, res in additional.items():
            with h5py.File(f"{output_path}/{tag}.h5", 'w') as f:
                for key, val in res.items():
                    patientID, fieldID, instanceID, dsName = key.split("/")
                    group_path = f"{patientID}/{fieldID}/{instanceID}"
                    g = f[group_path] if group_path in f else f.create_group(group_path)
                    g.create_dataset(f'{dsName}', data=val)

def MetricsSave(metrics_list, out_path):
    df = pd.DataFrame([res for batch_res in metrics_list for res in batch_res])
    df[['SubID', 'FieldID', 'InstanceID', 'Acq']] = df['key'].str.split('/', expand=True)
    df = df.drop(columns=['key'])
    df.to_pickle(f"{out_path}/metrics.pkl")
    df.to_csv(f"{out_path}/metrics.csv", index=False)

class ResSaver():
    def __init__(self, out_path, save_inp=False, save_gt=False, do_norm=False):
        self.out_path = out_path
        self.save_inp = save_inp
        self.save_gt = save_gt
        self.do_norm = do_norm

    def CalcNSave(self, datumHandler: DataHandler, outfolder, datacon_operator = None):
        outpath = os.path.join(self.out_path, outfolder)
        os.makedirs(outpath, exist_ok=True)

        inp = datumHandler.getImInp()
        if torch.is_complex(inp):
            inp = abs(inp)
        inp = inp.float().numpy()

        out = datumHandler.getImOut()
        if torch.is_complex(out):
            out = abs(out)
        out = out.float().numpy()

        SaveNIFTI(out, os.path.join(outpath, "out.nii.gz"))

        if self.save_inp:
            SaveNIFTI(inp, os.path.join(outpath, "inp.nii.gz"))

        gt = datumHandler.getImGT()
        if gt is not None:
            if torch.is_complex(gt):
                gt = abs(gt)
            gt = gt.float().numpy()

            if self.save_gt:
                SaveNIFTI(gt, os.path.join(outpath, "gt.nii.gz"))

            if self.do_norm:
                out = minmax(out)
                inp = minmax(inp)
                gt = minmax(gt)

            out_metrics, out_ssimMAP, out_diff = calc_metircs(
                gt, out, tag="Out", norm4diff=not self.do_norm)
            SaveNIFTI(out_ssimMAP, os.path.join(outpath, "ssimMAPOut.nii.gz"))
            SaveNIFTI(out_diff, os.path.join(outpath, "diffOut.nii.gz"))

            inp_metrics, inp_ssimMAP, inp_diff = calc_metircs(
                gt, inp, tag="Inp", norm4diff=not self.do_norm)
            SaveNIFTI(inp_ssimMAP, os.path.join(outpath, "ssimMAPInp.nii.gz"))
            SaveNIFTI(inp_diff, os.path.join(outpath, "diffInp.nii.gz"))

            metrics = {**out_metrics, **inp_metrics}
        else:
            metrics = None

        if datacon_operator is not None:
            datumHandler.setOutCorrectedK(datacon_operator.apply(out_ksp=datumHandler.getKOut(
                imnorm=False), full_ksp=datumHandler.getKGT(imnorm=False), under_ksp=datumHandler.inpK, metadict=datumHandler.metadict))  # TODO: param for imnorm
            # outCorrected = abs(datumHandler.getImOutCorrected()).float().numpy()

            #TODO: fix the things below, only one should be there
            outCorrected = abs(datumHandler.getImOutCorrected(
            )).float().numpy()  # TODO: param real v abs
            SaveNIFTI(outCorrected, os.path.join(
                outpath, "outCorrected.nii.gz"))
            if gt is not None:
                # if self.do_norm:
                #     outCorrected = minmax(outCorrected)
                outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(
                    gt, outCorrected, tag="OutCorrected", norm4diff=not self.do_norm)
                SaveNIFTI(outCorrected_ssimMAP, os.path.join(
                    outpath, "ssimMAPOutCorrected.nii.gz"))
                SaveNIFTI(outCorrected_diff, os.path.join(
                    outpath, "diffOutCorrected.nii.gz"))
                metrics = {**metrics, **outCorrected_metrics}

            #TODO: fix the things below, only one should be there
            outCorrected = datumHandler.getImOutCorrected(
            ).real.float().numpy()  # TODO: param real v abs
            SaveNIFTI(outCorrected, os.path.join(
                outpath, "outCorrectedReal.nii.gz"))
            if gt is not None:
                # if self.do_norm:
                #     outCorrected = minmax(outCorrected)
                outCorrected_metrics, outCorrected_ssimMAP, outCorrected_diff = calc_metircs(
                    gt, outCorrected, tag="OutCorrectedReal", norm4diff=not self.do_norm)
                SaveNIFTI(outCorrected_ssimMAP, os.path.join(
                    outpath, "ssimMAPOutCorrectedReal.nii.gz"))
                SaveNIFTI(outCorrected_diff, os.path.join(
                    outpath, "diffOutCorrectedReal.nii.gz"))
                metrics = {**metrics, **outCorrected_metrics}

        return metrics


def CustomInitialiseWeights(m):
    """Initialises Weights for our networks.
    Currently it's only for Convolution and Batch Normalisation"""

    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        if type(m.weight) is torch.nn.ParameterList or m.weight.dtype is torch.cfloat:
            torchcomplex.nn.init.trabelsi_standard_(m.weight, kind="glorot")
        else:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def process_testbatch(out_aggregators, datum, prediction):
    for i in range(len(datum['filename'])):
        out_aggregators[datum['filename'][i]][datum['sliceID']
                                              [i].item()] = prediction[i].detach().cpu()


def process_slicedict(dict_sliceout, axis=-1):
    sliceIDs = sorted(list(dict_sliceout.keys()))
    out = []
    for s in sliceIDs:
        out.append(dict_sliceout[s].squeeze())
    if torch.is_tensor(out[0]):
        return torch.stack(out, axis=axis)
    else:
        return np.stack(out, axis=axis)


def fetch_vol_subds(subjectds, filename, slcaxis=-1):
    df = subjectds.df
    ids = np.array(df.index[df['filename'] == filename].tolist())
    sliceIDs = df[df['filename'] == filename].sliceID.tolist()
    ids = ids[np.argsort(sliceIDs)]
    inp = []
    gt = []
    for i in ids:
        inp.append(subjectds[i]['inp']['data'].squeeze())
        gt.append(subjectds[i]['gt']['data'].squeeze())
    sub = {
        "inp": {
            "data": np.stack(inp, axis=slcaxis)
        },
        "gt": {
            "data": np.stack(gt, axis=slcaxis)
        },
        "filename": filename
    }
    return sub
    # else:
    #     return torch.stack(inp, axis=slcaxis), torch.stack(gt, axis=slcaxis)


def fetch_vol_subds_fastMRI(subjectds, filename, slcaxis=-1):
    df = pd.DataFrame(subjectds.examples, columns=[
                      "fname", "dataslice", "metadata"])
    df["fname"] = df["fname"].apply(lambda x: os.path.basename(x))
    ids = np.array(df.index[df['fname'] == filename].tolist())
    sliceIDs = df[df['fname'] == filename].dataslice.tolist()
    ids = ids[np.argsort(sliceIDs)]
    inp = []
    inpK = []
    gt = []
    gtK = []
    mask = []
    fastMRIAttrs = []
    for i in ids:
        ds = subjectds[i]
        inp.append(ds['inp']['data'].squeeze())
        inpK.append(ds['inp']['ksp'].squeeze())
        gt.append(ds['gt']['data'].squeeze())
        gtK.append(ds['gt']['ksp'].squeeze())
        if "mask" in ds['metadict']:
            mask.append(ds['metadict']['mask'].squeeze(0))
        if "fastMRIAttrs" in ds['metadict']:
            fastMRIAttrs.append(ds['metadict']['fastMRIAttrs'])
    sub = {
        "inp": {
            "data": np.stack(inp, axis=slcaxis),
            # "ksp": np.stack(inpK, axis=slcaxis)
        },
        "gt": {
            "data": np.stack(gt, axis=slcaxis),
            # "ksp": np.stack(gtK, axis=slcaxis)
        },
        "filename": filename
    }
    if len(mask) > 0 and len(fastMRIAttrs) > 0:
        sub["metadict"] = {
            "mask": np.stack(mask, axis=slcaxis),
            "fastMRIAttrs": fastMRIAttrs
        }
    else:
        if len(mask) > 0:
            sub["metadict"] = {"mask": np.stack(mask, axis=slcaxis)}
        elif len(fastMRIAttrs) > 0:
            sub["metadict"] = {"fastMRIAttrs": fastMRIAttrs}
    return sub
    # else:
    #     return torch.stack(inp, axis=slcaxis), torch.stack(gt, axis=slcaxis)

# class MetaLogger():
#     def __init__(self, active=True) -> None:
#         self.activate = active

#     def __call__(self, tag, batch_idx, metas):
#         if self.active:


#     metas = {key:val for (key,val) in batch['inp'].items() if "Meta" in key}


