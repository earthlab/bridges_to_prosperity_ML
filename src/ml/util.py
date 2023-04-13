import warnings
from enum import Enum
from math import isnan

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

TFORM = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),  # flip image 1/2 the time
        transforms.RandomHorizontalFlip(),  # flip image 1/2 the time
    ]
)


class BaseB2PDataset(torch.utils.data.Dataset):

    # Make it so np doesn't yell about using str
    warnings.simplefilter(action='ignore', category=FutureWarning)

    def __init__(self, transform=None, batch_size=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tiles = None
        self.bbox = None
        self.classes = ['bridge', 'no bridge']
        self.class_to_idx = {'bridge': 1, 'no bridge': 0}
        self.transform = transform
        self.batch_size = batch_size

        # Set info for iteration
        self.__curr = 0

    def _calc_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_file = self.tiles[idx]
        assert isinstance(tile_file, str), 'tile_file is a {}, = {}'.format(type(tile_file), tile_file)
        image = torch.load(tile_file)
        bbox = self.bbox[idx]
        w, c, h = image.shape
        if w == h and w != c:
            image = torch.moveaxis(image, 1, 0)
        if self.transform:
            image = self.transform(image)

        return image, tile_file, bbox

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self._calc_item(idx)

    def __iter__(self):
        return self

    def __next__(self):
        # Terminate if range over, otherwise return current, calculate next.
        try:
            if self.__curr > self.__term:
                self.__curr = 0
                raise StopIteration()
            (cur, self.__curr) = (self.__curr, self.__curr + 1)
            return self._calc_item(cur)
        except KeyError:
            self._curr = 0


class B2PNoTruthDataset(BaseB2PDataset):

    def __init__(self, csv_file, transform=None, batch_size=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(transform, batch_size)
        df = pd.read_csv(
            csv_file,
            index_col=0,
            dtype={
                'tile': str,
                'bbox': object
            }
        )
        self.df = df

        self.tiles = df['tile']
        self.bbox = df['bbox']

        # Set info for iteration
        self.__term = len(self.tiles) - 1

    def _calc_item(self, idx):
        image, tile_file, bbox = super()._calc_item(idx)

        # return none for target value
        return image, [], tile_file, bbox


class B2PTruthDataset(BaseB2PDataset):

    def __init__(self, csv_file, transform=None, batch_size=1, ratio=None, replacement=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(transform, batch_size)
        df = pd.read_csv(
            csv_file,
            index_col=0,
            dtype={
                'tile': str,
                'bbox': object,
                'is_bridge': bool,
                'bridge_loc': object
            }
        )
        self.df = df
        self.ratio = ratio
        self.replacement = replacement
        self.ix_already_sampled = []
        if ratio is not None:
            if not isinstance(ratio, (int, float)):
                raise TypeError('Ratio must be a number')
            self.update()
        else:
            self.tiles = df['tile']
            self.bbox = df['bbox']
            self.is_bridge = df['is_bridge']

        # Set info for iteration
        self.__term = len(self.tiles) - 1

    def update(self):
        if self.ratio is None:
            return
        ix_bridge = np.where(self.df['is_bridge'].to_numpy())[0]
        ix_no_bridge = np.where(np.logical_not(self.df['is_bridge'].to_numpy()))[0]
        if not self.replacement: ix_no_bridge = np.setdiff1d(ix_no_bridge, self.ix_already_sampled)
        num_bridge = len(ix_bridge)
        num_no_bridge = int(round(self.ratio * num_bridge))  # ratio = num_no_bridge / num_bridge
        ix_no_bridge = np.random.choice(ix_no_bridge, num_no_bridge)
        self.ix_already_sampled = np.concatenate((ix_no_bridge, self.ix_already_sampled), axis=0)
        ix_dset = np.concatenate((ix_no_bridge, ix_bridge), axis=0)
        np.random.shuffle(ix_dset)
        self.tiles = self.df['tile'][ix_dset]
        self.bbox = self.df['bbox'][ix_dset]
        self.is_bridge = self.df['is_bridge'][ix_dset]

        # Update the index attribute of each Series
        self.tiles.index = [i for i in range(len(self.tiles))]
        self.bbox.index = [i for i in range(len(self.bbox))]
        self.is_bridge.index = [i for i in range(len(self.is_bridge))]

    def _calc_item(self, idx):
        image, tile_file, bbox = super()._calc_item(idx)
        is_bridge = self.is_bridge[idx]

        return image, int(is_bridge), tile_file, bbox


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
        if isnan(val): return
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
