import os
import time
from typing import Union

import pandas as pd
import torch
import torchvision

from src.ml.util import B2PNoTruthDataset, B2PTruthDataset, AverageMeter, ProgressMeter, DEFAULT_ARGS
from file_types import TrainedModel, SingleRegionTileMatch, MultiRegionTileMatch, InferenceResultsCSV


def inference_torch(model_file: TrainedModel, tile_match_file: Union[SingleRegionTileMatch, MultiRegionTileMatch], results_file: InferenceResultsCSV, truth_data: bool, 
                    batch_size: int = None, gpu=None, num_workers: int = None, print_frequency: int = 100, args=DEFAULT_ARGS):

    print("Using pre-trained model '{}'".format(model_file.architecture))
    model = torchvision.models.__dict__[model_file.architecture](pretrained=True)
    args.layers = model_file.layers

    if batch_size is not None: 
        args.batch_size = batch_size
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if model_file.architecture.startswith('alexnet') or model_file.architecture.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if gpu is None:
        checkpoint = torch.load(model_file.archive_path)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_file.archive_path, map_location=loc)
    else:
        assert False, "Shouldn't happen"

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    dset = B2PTruthDataset(tile_match_file.archive_path, layers=args.layers) if truth_data else B2PNoTruthDataset(tile_match_file.archive_path, layers=args.layers)
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    n = dset.__len__()
    columns = ['tile', 'bbox', 'target', 'pred', 'conf'] if truth_data else ['tile', 'bbox', 'pred', 'conf']
    res_df = pd.DataFrame(
        columns=columns,
        index=range(n)
    )
    # switch to evaluate mode
    model.eval()
    # again no gradients needed
    with torch.no_grad():
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        progress = ProgressMeter(
            len(dloader),
            [batch_time, data_time],
            prefix="Inference: "
        )

        end = time.time()
        for i, (data, target, tile, bbox) in enumerate(dloader):
            data_time.update(time.time() - end)
            # move data to the same device as model
            data = data.double()
            model = model.double()
            output = model(data)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

            # store res to file
            ix = range(
                i * batch_size,
                min(
                    (i + 1) * batch_size,
                    n
                )
            )
            res_df.loc[ix, 'tile'] = tile
            res_df.loc[ix, 'bbox'] = bbox
            if truth_data:
                res_df.loc[ix, 'target'] = target
            res_df.loc[ix, 'pred'] = pred.cpu().numpy()
            res_df.loc[ix, 'conf'] = conf.cpu().numpy()
            # update time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_frequency == 0:
                progress.display(i + 1)

    res_df.to_csv(results_file.archive_path)
    return res_df
