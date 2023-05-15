import os
import time

import pandas as pd
import torch
import torchvision

from src.ml.util import B2PNoTruthDataset, B2PTruthDataset, AverageMeter, ProgressMeter


def inference_torch(model_file_path: str, tile_csv_path: str, results_csv_path: str, truth_data: bool,
                    batch_size: int = 1000, gpu=None, num_workers: int = 12, print_frequency: int = 100):
    os.makedirs(os.path.basename(results_csv_path), exist_ok=True)
    architecture = os.path.basename(model_file_path).split('.')[0]
    print("Using pre-trained model '{}'".format(architecture))
    model = torchvision.models.__dict__[architecture](pretrained=True)

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
        if architecture.startswith('alexnet') or architecture.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if gpu is None:
        checkpoint = torch.load(model_file_path)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_file_path, map_location=loc)
    else:
        assert False, "Shouldn't happen"

    num_channels = 8
    model.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    torch.nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    dset = B2PTruthDataset(tile_csv_path) if truth_data else B2PNoTruthDataset(tile_csv_path)
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
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
            if target:
                res_df.loc[ix, 'target'] = target
            res_df.loc[ix, 'pred'] = pred.cpu().numpy()
            res_df.loc[ix, 'conf'] = conf.cpu().numpy()
            # update time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_frequency == 0:
                progress.display(i + 1)

    res_df.to_csv(results_csv_path)
    return res_df
