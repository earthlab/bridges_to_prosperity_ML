"""
Parser class that encapsulates configuration settings into different namespaces. This is done so strings are used as
little as possible when accessing configuration values.
"""
import os

import yaml

from definitions import B2P_DIR

with open(os.path.join(B2P_DIR, 'config.yaml'), 'r') as f:
    configurations = yaml.safe_load(f)


class CONFIG:
    configuration_dict = configurations

    class AWS:
        BUCKET = f"{configurations['aws']['bucket']}"

    class TORCH:
        class INFERENCE:
            _inference_dict = configurations['torch']['inference']
            BATCH_SIZE = _inference_dict['batch_size']
            NUM_WORKERS = _inference_dict['num_workers']
