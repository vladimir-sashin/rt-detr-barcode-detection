from typing import NamedTuple

from source.constants import DATA_DIR

BARCODE_LABEL_NAME = 'barcode'
GT_FIELD_LOADING = 'ground_truth'
GT_FIELD = f'{GT_FIELD_LOADING}_detections'
PREDICT_FIELD = 'prediction_detections'

RAW_DATASETS = DATA_DIR / 'raw_datasets'
PREPARED_DATASETS = DATA_DIR / 'prepared_datasets'

TMP_DATASETS = DATA_DIR / 'temporary'
TMP_RAW_DATASETS = TMP_DATASETS / 'raw_datasets'
TMP_PREPARED_DATASETS = TMP_DATASETS / 'prepared_datasets'

RAW_DATASET_PREFIX = 'raw_'
PREPARED_DATASET_PREFIX = ''

BARCODES_DATA_DIR = DATA_DIR / 'barcodes'

RAW_ZIP_PATH = BARCODES_DATA_DIR / 'raw_data.zip'
RAW_DATASET_PATH = BARCODES_DATA_DIR / 'raw_data'

DEFAULT_COCO_PATH = BARCODES_DATA_DIR / 'splits_coco'

DEFAULT_RAW_URL = 'https://drive.google.com/u/0/uc?id=1g4Z-j9k3fuKGYFd-leOBpYN-qkmPrVRW'


class SplitNames(NamedTuple):
    train: str
    valid: str
    test: str


SPLITS = SplitNames('train', 'val', 'test')
DEFAULT_SPLIT_RATIO_VALS = (0.7, 0.15, 0.15)
