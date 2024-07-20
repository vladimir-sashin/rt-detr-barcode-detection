from pathlib import Path

from source.data.config import BarcodesDataConfig
from source.data.dir_utils import handle_tmp_dir
from source.data.prep.clearml_versioning import (
    create_data_manager,
    get_ds_local_copy,
)
from source.data.prep.core.export_coco import splits_to_coco
from source.data.prep.core.split import split_raw_dataset


def prepare_local_data(cfg: BarcodesDataConfig, raw_dataset_path: Path, prepared_dataset_dir: Path) -> None:
    # Split to train/val/test
    splits = split_raw_dataset(raw_dataset_path, cfg.split_cfg.split_ratios, cfg.split_cfg.seed)
    # Save in COCO format
    splits_to_coco(splits, prepared_dataset_dir)


def prepare_data_from_cml(
    cfg: BarcodesDataConfig,
    dataset_name: str,
    raw_dataset_name: str,
    prepared_dataset_dir: Path,
) -> None:
    if data_manager := create_data_manager(cfg.clearml_cfg.project_name, dataset_name, raw_dataset_name):
        raw_dataset_path = get_ds_local_copy(data_manager.raw_dataset)
        prepare_local_data(cfg, raw_dataset_path, prepared_dataset_dir)
        data_manager.upload_processed_ds(prepared_dataset_dir)
        handle_tmp_dir(prepared_dataset_dir, cfg.clearml_cfg.keep_local_copy)
