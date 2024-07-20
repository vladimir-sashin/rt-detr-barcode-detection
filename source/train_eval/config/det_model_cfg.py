from pydantic import Field

from source.base_config import BaseValidatedConfig


class PretrainConfig(BaseValidatedConfig):
    load_pretrain: bool = True
    use_plus_obj365: bool = True


class DetModelConfig(BaseValidatedConfig):
    architecture: str = 'rtdetr_r18vd'
    pretrain: PretrainConfig = Field(PretrainConfig())
