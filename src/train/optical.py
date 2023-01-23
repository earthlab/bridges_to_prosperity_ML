from scr.train.optical_fastai import _fastai_format_inputs, _fastai_train_optical, _fastai_metris
from scr.train.optical_torch import _torch_format_inputs, _torch_train_optical, _torch_metris

"""
    given paths to tiff fi
"""
def format_inputs(
    csv_files: dict, 
    tiff_dirs: dict,
    api: str ="fastai"
    ):
    if api == "fastai":
        return _fastai_format_inputs(
            csv_files, 
            tiff_dirs
        )
    elif api == "torch":
        return _torch_format_inputs(
            csv_files,
            tiff_dirs
        )
    raise Exception(f"API {api} not implemented")

def train_optical(
    train_df, 
    resnt, 
    api="fastai",
    batch_sz=16,
    model_dir=None
):
    if api == "fastai":
        return _fastai_train_optical(
            train_df, 
            batch_sz,
            resnt,
            model_dir
        )
    raise Exception(f"{api} not implemented")

def metrics(api, val_df, interps, resnt):
    if api == "fastai":
        return _fastai_metris(val_df, interps, resnt)
    
    elif api == "torch":
        return _torch_metrics(
            csv_files,
            tiff_dirs
        )
    raise Exception(f"{api} not implemented")