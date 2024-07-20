from source.base_config import BaseValidatedConfig


class ClearMLConfig(BaseValidatedConfig):
    project_name: str = 'Barcode_Detection'
    experiment_name: str = 'Training'
    track_in_clearml: bool = True
