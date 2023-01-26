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
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from shapely.geometry import polygon

from src.utilities.coords import *

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) 

DEFAULT_ARGS = Namespace(
    datadir = os.path.join(BASE_DIR, "data", "torch"),
    arch = 'resnet18',
    workers = 4,
    epochs = 90,
    start_epoch = 0,
    batch_size = 256,
    lr = 0.1,
    momentum = 0.9,
    weight_decay=1e-4,
    print_freq=10,
    resume = '',
    evaluate = False,
    pretrained = False,
    world_size= -1,
    rank =  -1,
    dist_url='tcp://224.66.41.62:23456',
    dist_backend='nccl',
    seed = None,
    gpu = None,
    multiprocessing_distributed = False,
    dummy = False,
    best_acc1 = 0,
    distributed = False
)

best_acc1 = 0
best_model = None

def split_list(list, n):
    m = len(list)
    step = int(round(m/n))
    for i in range(n):
        yield list[i*step:(i+1)*step] 

def finder(args):
    bridge_tiles = []
    no_bridge_tiles = []
    for tif, bbox in tqdm.tqdm(args.list, desc=f'hella {args.n}', position=args.n):
        p = polygon.Polygon(bbox)
        found = False
        for loc in args.bridge_locations: 
            # check if the current tiff tile contains the current verified bridge location
            if p.contains(loc):
                bridge_tiles.append(tif)
                found = True
                break
        if not found:
            no_bridge_tiles.append(tif)
    return (bridge_tiles, no_bridge_tiles)

def bitchen(FF) : 
    hella_list = []
    for f in FF: 
        with open(f, 'r') as geo: 
            g = json.load(geo)
            for tif, bbox in g.items():
                hella_list.append((tif, bbox))
    return hella_list

def mv(args):
    for tif in tqdm.tqdm(args.args.file_list, position=args.n):
        d, fn = os.path.split(tif)
        _, abbrv =os.path.split(d)
        shutil.copyfile(tif, os.path.join(args.dir, 'yes', f'{abbrv}_{fn}'))

def mover(ix, file_list, dir):
    n = mp.cpu_count() - 1
    with mp.Pool(n) as p:
        args = []
        cnt = 0
        for ii in split_list(ix, n):
            args.append(
                Namespace(
                    file_list=[file_list[i] for i in ii], 
                    dir=dir,
                    n=cnt
                )
            )
            cnt+=1
        for _ in p.map(mv, args):
            pass
    
def _torch_prepare_inputs(ground_truth_dir:str = None, out_dir:str = None, tile_dir:str = None):
    # mp.set_start_method('spawn')
    if ground_truth_dir is None: 
        ground_truth_dir = os.path.join(BASE_DIR, 'data', 'ground_truth')
    if tile_dir is None: 
        tile_dir = os.path.join(BASE_DIR, 'data', f'tiles')  
    if out_dir is None: 
        today = date.today()
        # datestr = today.strftime('%Y-%m-%d')
        out_dir = os.path.join(BASE_DIR, 'data', f'torch')   
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    n = mp.cpu_count() - 1 # num cores 

    with Timer():
        truth_files = glob(os.path.join(ground_truth_dir,"*.csv"))
        geo_files = glob(os.path.join(tile_dir,'**', "geom_lookup.json"), recursive=True)
        print('Made it through the globs')

    tiles_list = []
    with Timer():
        with mp.Pool(n) as p: 
            res = p.map(bitchen, split_list(geo_files, n))
            for r in res:
                tiles_list += r
            
        print(f'tiles_list: {len(tiles_list)}')

    bridge_locations = []
    with Timer():
        for tfile in truth_files:
            tDf = pd.read_csv(tfile)
            bridge_locations += gpd.points_from_xy(tDf['Latitude'], tDf['Longitude'])
        print(f'bridge_locations : {len(bridge_locations)}')
    # figure out which tiles have bridges in them 
    bridge_tiles = []
    no_bridge_tiles = []   
    with mp.Pool(n) as p:
        args = []
        with Timer():
            cnt = 0
            for  t in split_list(tiles_list, n):
                args.append(
                    Namespace(
                        n=n,
                        list=t, 
                        bridge_locations=copy.deepcopy(bridge_locations)
                    )
                )
                cnt += 1
            print('Prepared args')
        res = None
        with Timer():
            res = p.map(finder, args)
            print('found bridge locations')
        
        with Timer():
            for r in res:
                bridge_tiles += r[0]
                no_bridge_tiles += r[1]
            print('Combining results')
    print(f'bridge_tiles : {len(bridge_tiles)}')
    print(f'no_bridge_tiles : {len(no_bridge_tiles)}')

    with open(os.path.join(BASE_DIR, 'data', 'bridge_locations.pkl'), 'wb') as outp:
        pickle.dump(bridge_tiles, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(no_bridge_tiles, outp, pickle.HIGHEST_PROTOCOL)

    b_train_ix = []
    nb_train_ix = []
    b_val_ix = []
    nb_val_ix = []
    with Timer():
        # stratified sampling to set aside x% of data for traingin and 1-x% for validation
        b_train_ix = np.random.choice(len(bridge_tiles), size = 0.7*len(bridge_tiles), replace = False)
        nb_train_ix = np.random.choice(len(no_bridge_tiles), size = 0.7*len(no_bridge_tiles), replace = False)
        b_val_ix = np.setdiff1d(np.arange(len(bridge_tiles)), b_train_ix)
        nb_val_ix = np.setdiff1d(np.arange(len(no_bridge_tiles)), nb_train_ix)
        print(f'b_train_ix: {len(b_train_ix)}')
        print(f'nb_train_ix: {len(nb_train_ix)}')
        print(f'b_val_ix: {len(b_val_ix)}')
        print(f'nb_val_ix: {len(nb_val_ix)}')

    # create directories for the training and validation
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    if not os.path.isdir(train_dir): 
        os.makedirs(train_dir)
        os.makedirs(os.path.join(train_dir, 'yes'))
        os.makedirs(os.path.join(train_dir, 'no'))
    if not os.path.isdir(val_dir): 
        os.makedirs(val_dir)
        os.makedirs(os.path.join(val_dir, 'yes'))
        os.makedirs(os.path.join(val_dir, 'no'))

    with Timer():
        mover(b_train_ix, bridge_tiles, os.makedirs(os.path.join(train_dir, 'yes')))
        mover(nb_train_ix, bridge_tiles, os.makedirs(os.path.join(train_dir, 'no')))
        mover(b_val_ix, bridge_tiles, os.makedirs(os.path.join(val_dir, 'yes')))
        mover(nb_val_ix, bridge_tiles, os.makedirs(os.path.join(val_dir, 'no')))
        
    return None

def _torch_train_optical(datadir:str=None, saveFile:str=None):
    args = DEFAULT_ARGS
    if datadir is not None:
        args.datadir = datadir
    if saveFile is None: 
        saveDir = os.path.join(BASE_DIR, 'data', 'models')
        today = date.today()
        datestr = today.strftime('%Y-%m-%d')
        saveFile = os.path.join(saveDir, f'{args.model}_{datestr}.pkl')
    else:
        saveDir = os.path.dirname(saveFile)
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), saveFile=saveFile)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, saveFile)

def main_worker(gpu, ngpus_per_node, args, saveFile):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)
        
    torch.save({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
    }, saveFile)

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

