import os

# Local
B2P_DIR = os.path.abspath(os.path.dirname(__file__))
REGION_FILE_PATH = os.path.join(B2P_DIR, 'data', 'region_info.yaml')
COMPOSITE_DIR = os.path.join(B2P_DIR, 'data', 'composites')
TILE_DIR = os.path.join(B2P_DIR, 'data', 'tiles')
TRUTH_DIR = os.path.join(B2P_DIR, 'data', 'ground_truth')
SENTINEL_2_DIR = os.path.join(B2P_DIR, 'data', 'sentinel2')
ELEVATION_DIR = os.path.join(B2P_DIR, 'data', 'elevation')
SLOPE_DIR = os.path.join(B2P_DIR, 'data', 'slope')
MULTIVARIATE_DIR = os.path.join(B2P_DIR, 'data', 'multivariate')
TORCH_DIR = os.path.join(B2P_DIR, 'data', 'torch')
MODEL_DIR = os.path.join(TORCH_DIR, 'trained_models')
INFERENCE_RESULTS_DIR = os.path.join(B2P_DIR, 'data', 'inference_results')
OSM_DIR = os.path.join(B2P_DIR, 'data', 'osm')
SECRETS_FILE_PATH = os.path.join(B2P_DIR, 'secrets.yaml')
MGRS_INDEX_FILE = os.path.join(B2P_DIR, 'data', 'mgrs_index.json')

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