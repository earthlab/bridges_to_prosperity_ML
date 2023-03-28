from src.ml.inference import inference_torch
from src.ml.train import train_torch
from bin.composites_to_tiles import create_tiles, mp
import torch, os
from glob import glob

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

def main():
    tile_dir = 'data/tiles'
    data_csv = os.path.join('data', 'tiles', 'all_df.csv')
    
    with open('overnight.log', 'w') as f:
        print('model,total_acc,bridge_acc,no_bridge_acc',file=f)
        for model in ['resnet18']:
            if len(glob(f'data/torch/{model}/*')) < 20:
                train_torch(
                    _arch=model, 
                    tile_dir=tile_dir,
                    datadir='data/torch'
                )
            res_csv = os.path.join(PROJECT_DIR, 'data', 'resnet18_1.5r.csv')
            model_file = f'data/torch/{model}/{model}_1.5.best.tar'
            state_dict = torch.load(model_file)
            total_acc = state_dict['total_acc']
            bridge_acc = state_dict['bridge_acc']
            no_bridge_acc = state_dict['no_bridge_acc']
            print(f'{model},{total_acc},{bridge_acc},{no_bridge_acc}', file=f)
            if os.path.isfile(res_csv): continue
            inference_torch(
                model_file=model_file, 
                res_csv=res_csv,
                tile_csv=data_csv
            )


if __name__ == '__main__':
    main()