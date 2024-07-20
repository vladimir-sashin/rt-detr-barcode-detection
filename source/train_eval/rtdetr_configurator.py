from pathlib import Path
from typing import Any

from jinja2 import Template

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.constants import CFG_TEMPLATES, MODEL_ARCH_SUFFIX
from source.train_eval.pathfinding.inputs import (  # noqa: I001, flake8 FP
    SplitsPathsConfig,
    find_dataset,
)
from source.train_eval.pathfinding.outputs import (  # noqa: I005, flake8 FP
    get_output_dir,
)


def render_template(  # type: ignore # Allow explicit `Any
    template_path: Path,
    cfg_dump: dict[str, Any],
    output_path: Path,
) -> None:
    with open(template_path, 'r') as template_file:
        yaml_template = template_file.read()

    template = Template(yaml_template)
    rendered_yaml = template.render(**cfg_dump)

    with open(output_path, 'w') as rendered_file:
        rendered_file.write(rendered_yaml)


class RenderEvalConfig(TrainEvalConfig):
    data_paths: SplitsPathsConfig
    output_dir: Path

    @property
    def rtdetr_cfg_dir(self) -> Path:
        return self.output_dir / 'rtdetr_rendered_configs'


def _create_test_eval_dir(output_dir: Path) -> Path:
    output_dir = output_dir / 'test_eval_rtdetr'
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def prepare_render_cfg(cfg: TrainEvalConfig, eval_on_test: bool = False) -> RenderEvalConfig:
    coco_splits = find_dataset(cfg)

    output_dir = get_output_dir(cfg)

    if eval_on_test:
        output_dir = _create_test_eval_dir(output_dir)
        coco_splits.valid = coco_splits.test

    return RenderEvalConfig(
        data_paths=coco_splits,
        output_dir=output_dir,
        **cfg.model_dump(),
    )


def render_templates(render_cfg: RenderEvalConfig, templates_dir: Path) -> None:
    cfg_dump = render_cfg.model_dump()

    for template_path in templates_dir.rglob('*.yml'):
        relative_path = template_path.relative_to(templates_dir)
        render_path = render_cfg.rtdetr_cfg_dir / relative_path
        render_path.parent.mkdir(exist_ok=True, parents=True)
        render_template(template_path, cfg_dump, render_path)


def generate_rtdetr_configs(cfg: TrainEvalConfig, eval_on_test: bool = False) -> Path:
    render_cfg = prepare_render_cfg(cfg, eval_on_test)
    render_templates(render_cfg, CFG_TEMPLATES)
    cfg.to_yaml(render_cfg.output_dir / 'train.yaml')

    return render_cfg.rtdetr_cfg_dir


def get_arch_cfg_path(cfg: TrainEvalConfig, rendered_path: Path) -> Path:
    return rendered_path / 'rtdetr' / f'{cfg.det_model_cfg.architecture}{MODEL_ARCH_SUFFIX}'
