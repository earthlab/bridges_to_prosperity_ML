import os

# Local
B2P_DIR = os.path.abspath(os.path.dirname(__file__))
REGION_FILE_PATH = os.path.join(B2P_DIR, 'data', 'region_info.yaml')
COMPOSITE_DIR = os.path.join(B2P_DIR, 'data', 'composites')
TILE_DIR = os.path.join(B2P_DIR, 'data', 'tiles')
TRUTH_DIR = os.path.join(B2P_DIR, 'data', 'ground_truth')
SENTINEL_2_DIR = os.path.join(B2P_DIR, 'data', 'sentinel2')
TORCH_DIR = os.path.join(B2P_DIR, 'data', 'torch')

# S3
S3_COMPOSITE_DIR = 'composites'
