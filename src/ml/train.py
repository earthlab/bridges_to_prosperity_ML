import os
import shutil
import time
import warnings
from argparse import Namespace
from typing import Union

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from src.ml.util import AverageMeter, ProgressMeter, Summary, accuracy, B2PTruthDataset, TFORM

MODEL_NAME = sorted(name for name in torchvision.models.__dict__ if name.islower() and not name.startswith("__")
                    and callable(torchvision.models.__dict__[name]))

DEFAULT_ARGS = Namespace(
    workers=4,
    epochs=40,
    start_epoch=0,
    batch_size=1000,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=10,
    resume='',
    evaluate=False,
    pretrained=False,
    world_size=-1,
    rank=-1,
    dist_url='tcp://224.66.41.62:23456',
    dist_backend='nccl',
    seed=None,
    gpu=None,
    multiprocessing_distributed=False,
    dummy=False,
    best_acc1=0,
    distributed=False
)

BEST_ACC1 = 0


def train_torch(results_dir: str, train_csv_path: str, test_csv_path: str, architecture: str,
                bridge_no_bridge_ratio: Union[None, float], seed: Union[None, int] = None):
    # Configure the namespace for this run

    args = DEFAULT_ARGS
    args.results_dir = results_dir
    args.train_csv_path = train_csv_path
    args.test_csv_path = test_csv_path
    args.architecture = architecture
    args.bridge_no_bridge_ratio = bridge_no_bridge_ratio

    os.makedirs(results_dir, exist_ok=True)

    if seed is not None:
        torch.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when '
                      'restarting from checkpoints.')

    # TODO: Maybe make these configurable from the CLI / signature
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global BEST_ACC1
    BEST_ACC1 = 0
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
        print("=> using pre-trained model '{}'".format(args.architecture))
        model = torchvision.models.__dict__[args.architecture](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.architecture))
        model = torchvision.models.__dict__[args.architecture]()
        # num_channels = 3
        # model.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # torch.nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:

        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device
        # scope, otherwise, DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)

                # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()

                # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are
                # not set
                model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)

    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.architecture.startswith('alexnet') or args.architecture.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        print('Using CUDA')
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
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
            else:
                assert False, "Shouldn't happen"
            args.start_epoch = checkpoint['epoch']
            BEST_ACC1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                BEST_ACC1 = BEST_ACC1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_csv = args.train_csv_path
    val_csv = args.test_csv_path
    assert os.path.isfile(train_csv), f'file dne: {train_csv}'
    assert os.path.isfile(val_csv), f'file dne: {val_csv}'
    train_dataset = B2PTruthDataset(
        train_csv,
        TFORM,
        args.batch_size,
        ratio=args.bridge_no_bridge_ratio
    )

    val_dataset = B2PTruthDataset(
        val_csv,
        TFORM,
        args.batch_size,
        ratio=args.bridge_no_bridge_ratio
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    if args.evaluate:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1, bridge_acc, no_bridge_acc = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > BEST_ACC1
        BEST_ACC1 = max(acc1, BEST_ACC1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.architecture,
                    'state_dict': model.state_dict(),  # this was full of NaN with all the training data
                    'best_acc1': BEST_ACC1,
                    'total_acc': acc1,
                    'bridge_acc': bridge_acc,
                    'no_bridge_acc': no_bridge_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                is_best,
                os.path.join(args.results_dir, f'{args.architecture}.chkpt{epoch + 1}.tar')
            )
        train_dataset.update()
        val_dataset.update()


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    total_mt = AverageMeter('Total Acc', ':6.2f', Summary.AVERAGE)
    bridge_mt = AverageMeter('Bridge Acc', ':6.2f', Summary.AVERAGE)
    no_bridge_mt = AverageMeter('No Bridge Acc', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, total_mt, bridge_mt, no_bridge_mt],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        images = images.float()[:,[4,6,7],:,:]

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        total_acc, bridge_acc, no_bridge_acc = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        total_mt.update(total_acc, images.size(0))
        bridge_mt.update(bridge_acc, images.size(0))
        no_bridge_mt.update(no_bridge_acc, images.size(0))

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
            for i, (images, target, _, _) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                images = images.float()[:, [4,6,7], :, :]
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                total_acc, bridge_acc, no_bridge_acc = accuracy(output, target)
                losses.update(loss.item(), images.size(0))
                total_mt.update(total_acc, images.size(0))
                bridge_mt.update(bridge_acc, images.size(0))
                no_bridge_mt.update(no_bridge_acc, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    total_mt = AverageMeter('Total Acc', ':6.2f', Summary.AVERAGE)
    bridge_mt = AverageMeter('Bridge Acc', ':6.2f', Summary.AVERAGE)
    no_bridge_mt = AverageMeter('No Bridge Acc', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, total_mt, bridge_mt, no_bridge_mt],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        total_mt.all_reduce()
        bridge_mt.all_reduce()
        no_bridge_mt.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return total_mt.avg, bridge_mt.avg, no_bridge_mt.avg


def save_checkpoint(state, is_best, filename):
    root, _ = os.path.split(filename)
    if not os.path.isdir(root):
        os.makedirs(root)
    torch.save(state, filename)
    if is_best:
        d = os.path.dirname(filename)
        f = os.path.basename(filename)
        prts = f.split('.chkpt')
        shutil.copyfile(
            filename,
            os.path.join(d, '.'.join([prts[0], 'best']) + ".tar")
        )
