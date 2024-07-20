from source.data.clearml_utils import create_dataset_task
from source.data.config import BarcodesDataConfig, get_data_config
from source.data.constants import TMP_RAW_DATASETS
from source.data.dir_utils import (
    get_raw_data_dir,
    get_raw_ds_name,
    handle_tmp_dir,
)
from source.data.fetch.clearml_versioning import RawDataManager
from source.data.fetch.core import download_and_unzip
from source.train_eval.config.data_source_cfg import StorageEnum


def fetch_raw_dataset(cfg: BarcodesDataConfig) -> None:
    dataset_zip_path = TMP_RAW_DATASETS / f'{cfg.dataset_name}.zip'

    if cfg.storage == StorageEnum.clearml:
        cml_dataset_name = get_raw_ds_name(cfg.dataset_name)
        cfg = create_dataset_task(cfg, cml_dataset_name)
        unzipped_data_dir = get_raw_data_dir(cfg.dataset_name, tmp=True)

        download_and_unzip(cfg.url, dataset_zip_path, unzipped_data_dir)
        data_manager = RawDataManager(cfg.clearml_cfg.project_name, cml_dataset_name)
        data_manager.upload_raw_dataset(unzipped_data_dir)
        handle_tmp_dir(unzipped_data_dir, cfg.clearml_cfg.keep_local_copy)

    elif cfg.storage == StorageEnum.local:
        unzipped_data_dir = get_raw_data_dir(cfg.dataset_name, tmp=False)
        download_and_unzip(cfg.url, dataset_zip_path, unzipped_data_dir)


if __name__ == '__main__':
    cfg = get_data_config()
    fetch_raw_dataset(cfg)
