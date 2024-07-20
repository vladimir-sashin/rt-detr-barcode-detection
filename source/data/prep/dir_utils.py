from pathlib import Path

from source.data.dir_utils import get_raw_data_dir
from source.data.prep.logger import LOGGER


class LocalDatasetNotFound(ValueError):
    """Dataset not found locally."""


def get_local_raw_dataset(raw_dataset_name: str) -> Path:
    LOGGER.info('Looking for a dataset locally...')
    dir_path = get_raw_data_dir(raw_dataset_name)
    if dir_path.is_dir():
        LOGGER.info('Found dataset at %s', dir_path)
        return dir_path
    raise LocalDatasetNotFound(f"Couldn't find dataset at {dir_path}.")
