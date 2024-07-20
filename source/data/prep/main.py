from source.data.clearml_utils import create_dataset_task
from source.data.config import BarcodesDataConfig, get_data_config
from source.data.dir_utils import (
    get_prepared_data_dir,
    get_prepared_ds_name,
    get_raw_ds_name,
)
from source.data.prep.core.pipe import (  # noqa: I001 flake8 FP
    prepare_data_from_cml,
    prepare_local_data,
)
from source.data.prep.dir_utils import (  # noqa: I005 flake8 FP
    get_local_raw_dataset,
)
from source.train_eval.config.data_source_cfg import StorageEnum


def preprocess_data(cfg: BarcodesDataConfig) -> None:
    if cfg.storage == StorageEnum.clearml:
        # Find out names and paths, create dataset task in ClearML
        cml_dataset_name = get_prepared_ds_name(cfg.dataset_name)
        cfg = create_dataset_task(cfg, cml_dataset_name)

        # Find out parent dataset name (raw version) and output path
        cml_raw_ds_name = get_raw_ds_name(cfg.dataset_name)
        prepared_dataset_dir = get_prepared_data_dir(cfg.dataset_name, tmp=True)

        # Run preprocessing and upload to ClearML as a new dataset version
        prepare_data_from_cml(
            cfg=cfg,
            dataset_name=cml_dataset_name,
            raw_dataset_name=cml_raw_ds_name,
            prepared_dataset_dir=prepared_dataset_dir,
        )

    elif cfg.storage == StorageEnum.local:
        # Find out output path
        prepared_dataset_dir = get_prepared_data_dir(cfg.dataset_name, tmp=False)
        # Run preprocessing and save output dataset locally
        prepare_local_data(
            cfg=cfg,
            raw_dataset_path=get_local_raw_dataset(cfg.dataset_name),
            prepared_dataset_dir=prepared_dataset_dir,
        )


if __name__ == '__main__':
    cfg = get_data_config()
    preprocess_data(cfg)
