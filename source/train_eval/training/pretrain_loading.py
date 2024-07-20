from functools import partial
from pathlib import Path
from typing import Optional

import requests

from source.constants import PRETRAINED_DIR
from source.train_eval.config.det_model_cfg import DetModelConfig
from source.train_eval.constants import (
    PLUS365_SUFFIX,
    PRETRAINED_MODELS,
    PRETRAINED_MODELS_PLUS,
)
from source.train_eval.logger import LOGGER


def _download_checkpoint(model_name: str, models_map: dict[str, str], checkpoints_dir: Path) -> Path:
    checkpoint_url = models_map[model_name]
    checkpoint_filename = checkpoints_dir / Path(checkpoint_url).name

    if not checkpoint_filename.is_file():
        LOGGER.info('Downloading %s model checkpoint from %s ...', model_name, checkpoint_url)
        response = requests.get(checkpoint_url, timeout=10)
        with open(checkpoint_filename, 'wb') as checkpoint_file:
            checkpoint_file.write(response.content)
        LOGGER.info('Successfully downloaded %s model checkpoint to %s', model_name, str(checkpoints_dir))

    return checkpoint_filename


def load_pretrained(model_cfg: DetModelConfig) -> Optional[Path]:
    if model_cfg.pretrain.load_pretrain is False:
        return None

    PRETRAINED_DIR.mkdir(exist_ok=True, parents=True)
    download_checkpoint = partial(_download_checkpoint, checkpoints_dir=PRETRAINED_DIR)
    model_name = model_cfg.architecture

    if model_cfg.pretrain.use_plus_obj365 is True:
        model_name += PLUS365_SUFFIX
        if model_name not in PRETRAINED_MODELS_PLUS:
            raise ValueError(f'There is no "COCO+Objects365" version for {model_cfg.architecture}')
        return download_checkpoint(model_name, PRETRAINED_MODELS_PLUS)

    return download_checkpoint(model_name, PRETRAINED_MODELS)
