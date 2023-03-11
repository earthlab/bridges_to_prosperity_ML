import os
import random
import shutil
import time
import warnings
from enum import Enum
from argparse import Namespace
from datetime import date
import geopandas as gpd
import numpy as np
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing as mp 
import json 
import copy 
import tqdm 
import PIL
from itertools import repeat

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from shapely.geometry import polygon
from skimage import io, transform

from src.utilities.coords import *

TFORM = transforms.Compose(
        [
            # transforms.ToTensor(), # take a 0, 255 image and scales it tp [0,1]
            transforms.RandomVerticalFlip(), # flip image 1/2 the time
            transforms.RandomHorizontalFlip(), # flip image 1/2 the time
            # transforms.adjust_brightness(0.1) # copied from fastai leaving out now
            # normalize,
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
        ]
    )

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

class B2PDataset(torch.utils.data.Dataset):
    ## Make it so np doesn't yell about using str
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(
            csv_file,
            index_col=0,
            dtype={
                'tile':str, 
                'bbox':object, 
                'is_bridge':bool, 
                'bridge_loc':object
            }
        )
        self.classes = ['bridge', 'no bridge']
        self.class_to_idx = {'bridge': 1, 'no bridge': 0}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_file = self.df.iloc[idx]['tile']
        is_bridge = self.df.iloc[idx]['is_bridge']
        image = torch.load(tile_file)
        bbox = self.df.iloc[idx]['bbox']
        w, c, h = image.shape
        if w == h and w != c:
            image = torch.moveaxis(image, 1, 0)
        if self.transform:
            image = self.transform(image)

        return (image, int(is_bridge), tile_file, bbox) # image, label, file name, bbox
    
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if np.isnan(val): return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = torch.max(output, 1)
        correct = pred.eq(target).float().sum(0, keepdim=True)
        total_acc = correct.mul_(100.0 / batch_size)
        total_acc = total_acc[0]

        bridge_acc = torch.nan
        bridge_ix = target.eq(1)
        if bridge_ix.any():
            bridge_correct = pred[bridge_ix].eq(target[bridge_ix]).float().sum(0, keepdim=True)
            bridge_acc = bridge_correct.mul_(100.0 / torch.sum(bridge_ix))
            bridge_acc = bridge_acc[0]

        no_bridge_acc = torch.nan
        no_bridge_ix = target.eq(0)
        if no_bridge_ix.any():
            no_bridge_correct = pred[no_bridge_ix].eq(target[no_bridge_ix]).float().sum(0, keepdim=True)
            no_bridge_acc = no_bridge_correct.mul_(100.0 / torch.sum(no_bridge_ix))
            no_bridge_acc = no_bridge_acc[0]

        return total_acc, bridge_acc, no_bridge_acc

