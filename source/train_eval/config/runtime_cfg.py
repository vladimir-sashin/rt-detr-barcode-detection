from pydantic import Field

from source.base_config import BaseValidatedConfig
from source.train_eval.constants import RtdetrScalers


class ScalerConfig(BaseValidatedConfig):
    enabled: bool = False
    type: RtdetrScalers = RtdetrScalers.grad_scaler


class RuntimeConfig(BaseValidatedConfig):
    sync_bn: bool = False
    find_unused_parameters: bool = False
    use_amp: bool = False
    scaler: ScalerConfig = Field(default=ScalerConfig())
