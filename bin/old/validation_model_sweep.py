import sys, os
from glob import glob
from tqdm.auto import tqdm
sys.path.append("/b2p")
from bin.run_inference import *

root = 'data/trained_models'
for model_file_path in glob(os.path.join(root, '**','*.best*.tar'), recursive=True):
    ratio_dir, model_name = os.path.split(model_file_path)
    resnet_dir, ratio_dir = os.path.split(ratio_dir)
    date_dir, _ = os.path.split(resnet_dir)
    metrics_dir = os.path.join(date_dir, 'metrics')
    results_csv = os.path.join(metrics_dir, 'validation.'+'.'.join(model_name.split('.')[:-1])+'.'+ratio_dir+'.csv' )
    if os.path.isfile(results_csv):
        os.remove(results_csv)
    print("==============================================================")
    print(results_csv)
    print("==============================================================")
    run_inference(
        model_file_path,
        'data/tiles/val_df.csv',
        results_csv,
        True, # truth data
        batch_size = CONFIG.TORCH.INFERENCE.BATCH_SIZE,
        num_workers = CONFIG.TORCH.INFERENCE.NUM_WORKERS)
    print()
    print()
