from pathlib import Path
from typing import Type, TypeVar, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

BVC = TypeVar('BVC', bound='BaseValidatedConfig')


class BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True, use_enum_values=True)


class ConfigYamlMixin(BaseModel):
    @classmethod
    def from_yaml(cls: Type[BVC], path: Union[str, Path]) -> BVC:
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)
