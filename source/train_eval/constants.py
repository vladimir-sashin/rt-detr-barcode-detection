from enum import Enum

from source.constants import PROJECT_ROOT

# RTDETR configuration
CFG_TEMPLATES = PROJECT_ROOT / 'source' / 'cfg_templates'
RTDETR_PATH = PROJECT_ROOT / 'third_party' / 'rtdetr_pytorch'
RTDETR_SRC_PATH = RTDETR_PATH / 'src'

# Metrics evaluation
DEFAULT_IMG_SIZE = (640, 640)
VISUALIZATIONS_TO_TRACK = 30
IOU_RANGE = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

# Search for confidence threshold
CONF_SEARCH_LOWER_BOUND = 0.1
CONF_SEARCH_UPPER_BOUND = 0.9
CONF_SEARCH_XTOL = 0.01
CONF_SEARCH_IOU = 0.7

# RTDETR models
MODEL_ARCH_SUFFIX = '_6x_coco.yml'
PLUS365_SUFFIX = '+obj365'
# TODO: upload to own storage and fix WPS407
PRETRAINED_MODELS = {  # noqa: WPS407
    'rtdetr_r18vd':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',
    'rtdetr_r34vd':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth',
    'rtdetr_r50vd_m':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth',
    'rtdetr_r50vd':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth',
    'rtdetr_r101vd':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth',
}
PRETRAINED_MODELS_PLUS = {  # noqa: WPS407
    f'rtdetr_r18vd{PLUS365_SUFFIX}':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth',
    f'rtdetr_r50vd{PLUS365_SUFFIX}':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth',
    f'rtdetr_r101vd{PLUS365_SUFFIX}':
        'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth',
}


# Training params
class RtdetrOptims(str, Enum):  # noqa: WPS600 Required by pydantic
    adam_w = 'AdamW'
    sgd = 'SGD'
    adam = 'Adam'


class RtdetrSchedulers(str, Enum):  # noqa: WPS600 Required by pydantic
    multi_step = 'MultiStepLR'
    cosine_annealing = 'CosineAnnealingLR'
    one_cycle = 'OneCycleLR'
    lambda_lr = 'LambdaLR'


class RtdetrScalers(str, Enum):  # noqa: WPS600 Required by pydantic
    grad_scaler = 'GradScaler'
