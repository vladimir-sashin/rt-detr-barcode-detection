import zipfile
from pathlib import Path

import gdown

from source.constants import PROJECT_ROOT
from source.data.fetch.logger import LOGGER


def unzip_raw_dataset(zip_path: Path, output_dir: Path) -> Path:
    output_dir_relative = output_dir.relative_to(PROJECT_ROOT)
    LOGGER.info('Unpacking raw dataset %s to %s...', zip_path.relative_to(PROJECT_ROOT), output_dir_relative)
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    LOGGER.info('Dataset is successfully unpacked to %s.', output_dir_relative)

    zip_path.unlink()
    return output_dir


def download_raw_zip(url: str, zip_path: Path) -> bool:
    zip_path.parent.mkdir(exist_ok=True, parents=True)
    if not zip_path.is_file():
        LOGGER.info('Downloading raw dataset to %s...', zip_path.relative_to(PROJECT_ROOT))
        gdown.download(url, str(zip_path))
        return True
    LOGGER.info(
        'Raw barcodes dataset is already found in %s, skipping downloading...',
        zip_path.relative_to(PROJECT_ROOT),
    )
    return False


def download_and_unzip(url: str, dataset_zip_path: Path, output_dir: Path) -> None:
    if download_raw_zip(url, dataset_zip_path):
        unzip_raw_dataset(dataset_zip_path, output_dir)
