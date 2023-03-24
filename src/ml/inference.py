from src.ml.util import *

ARGS = Namespace(
    gpu=None,
    batch_size=100,
    num_workers=12,
    tile_csv=os.path.join(BASE_DIR, "data", "final_tiles", "cote_divore.csv"),
    model_file=os.path.join(BASE_DIR, "data", "torch", "resnet18.best.tar"),
    res_csv=os.path.join(BASE_DIR, "data", "cote_divore_inference.csv"),
    print_freq=100
)


def inference_torch(model_file: str = None, tile_csv: str = None, res_csv: str = None):
    args = ARGS
    if model_file is not None:
        args.model_file = model_file
    if tile_csv is not None:
        args.tile_csv = tile_csv
    if res_csv is not None:
        args.res_csv = res_csv
    root, _ = os.path.split(res_csv)
    os.makedirs(root, exist_ok=True)
    arch = os.path.basename(args.model_file).split('.')[0]
    print("Using pre-trained model '{}'".format(arch))
    model = models.__dict__[arch](pretrained=True)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
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
    checkpoint = None
    if args.gpu is None:
        checkpoint = torch.load(args.model_file)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.model_file, map_location=loc)
    else:
        assert False, "Shouldn't happen"

    model.load_state_dict(checkpoint['state_dict'])
    dset = B2PDataset(args.tile_csv)
    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    n = dset.__len__()
    res_df = pd.DataFrame(
        columns=['tile', 'bbox', 'pred_bridge'],
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
            output = model(data)
            _, pred = torch.max(output, 1)

            # store res to file
            ix = range(
                i * args.batch_size,
                min(
                    (i + 1) * args.batch_size,
                    n
                )
            )
            res_df.loc[ix, 'tile'] = tile
            res_df.loc[ix, 'bbox'] = bbox
            res_df.loc[ix, 'pred'] = pred.numpy()
            # update time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i + 1)
    res_df.to_csv(args.res_csv)
    return res_df
