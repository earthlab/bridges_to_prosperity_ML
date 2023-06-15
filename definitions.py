import os

# Local
B2P_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(B2P_DIR, 'data')
REGION_FILE_PATH = os.path.join(DATA_DIR, 'region_info.yaml')
COMPOSITE_DIR = os.path.join(DATA_DIR, 'composites')
TILE_DIR = os.path.join(DATA_DIR, 'tiles')
MULTI_REGION_TILE_MATCH = os.path.join(DATA_DIR, 'multi_region_tile_match')
TRAIN_VALIDATE_SPLIT_DIR = os.path.join(TILE_DIR, 'train_validate_splits')
TRUTH_DIR = os.path.join(DATA_DIR, 'ground_truth')
SENTINEL_2_DIR = os.path.join(DATA_DIR, 'sentinel2')
ELEVATION_DIR = os.path.join(DATA_DIR, 'elevation')
SLOPE_DIR = os.path.join(DATA_DIR, 'slope')
TORCH_DIR = os.path.join(DATA_DIR, 'torch')
MODEL_DIR = os.path.join(TORCH_DIR, 'trained_models')
INFERENCE_RESULTS_DIR = os.path.join(DATA_DIR, 'inference_results')
OSM_DIR = os.path.join(DATA_DIR, 'osm')
SECRETS_FILE_PATH = os.path.join(B2P_DIR, 'secrets.yaml')
MGRS_INDEX_FILE = os.path.join(DATA_DIR, 'mgrs_index.json')

# S3
S3_COMPOSITE_DIR = 'composites'

# Order of data cube layers
LAYER_TO_IX = [
    'red',
    'blue',
    'green',
    'nir',
    'osm-water',
    'osm-boundary',
    'elevation',
    'slope'
]