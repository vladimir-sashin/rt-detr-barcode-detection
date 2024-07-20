from typing import NamedTuple

import clearml
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, computed_field


class Size(NamedTuple):
    width: int
    height: int


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int

    # https://github.com/pydantic/pydantic/issues/1303#issuecomment-599712964
    def __hash__(self) -> int:
        dict_values = self.__dict__.values()
        return hash((type(self),) + tuple(dict_values))

    def track_in_clearml(self, cml_logger: clearml.Logger, split: str, iou: float) -> None:
        for name, metric_value in self:
            cml_logger.report_scalar(
                title=f'{split.capitalize()} {name} per IoU',
                series=f'IoU:{iou}',
                value=metric_value,
                iteration=0,
            )


class BaseModelTypesAllowed(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SingleEvalResult(BaseModelTypesAllowed):
    iou: float
    metrics: Metrics
    confusion_matrix: plt.Figure


class IousMetricsColumns(NamedTuple):
    ious: tuple[float, ...]
    metrics_names: tuple[str, ...]
    metrics_values: tuple[float, ...]


class ResultsPerIou(BaseModelTypesAllowed):
    iou_range: list[float]
    metrics_range: list[Metrics]
    matrices_range: list[plt.Figure]

    @property
    def iou_metrics_cols(self) -> IousMetricsColumns:
        ious: list[float] = []
        metrics_names: list[str] = []
        metrics_values: list[float] = []

        for curr_iou, curr_metrics in zip(self.iou_range, self.metrics_range):
            curr_metrics_dict: dict[str, float] = curr_metrics.model_dump(exclude='support')

            ious.extend([curr_iou for _ in curr_metrics_dict])
            metrics_names.extend(curr_metrics_dict.keys())
            metrics_values.extend(curr_metrics_dict.values())

        return IousMetricsColumns(tuple(ious), tuple(metrics_names), tuple(metrics_values))

    def track_in_clearml(self, cml_logger: clearml.Logger, split: str) -> None:
        self._track_metrics(cml_logger, split)
        self._track_matrices(cml_logger, split)

    @computed_field
    def iou_metrics_mapping(self) -> dict[float, Metrics]:
        return dict(zip(self.iou_range, self.metrics_range))

    @classmethod
    def from_single_results(cls, single_results: list[SingleEvalResult]) -> 'ResultsPerIou':
        iou_range, metrics_range, matrices_range = [], [], []

        for single_result in single_results:
            iou_range.append(single_result.iou)
            metrics_range.append(single_result.metrics)
            matrices_range.append(single_result.confusion_matrix)

        return cls(
            iou_range=iou_range,
            metrics_range=metrics_range,
            matrices_range=matrices_range,
        )

    def _track_metrics(self, cml_logger: clearml.Logger, split: str) -> None:
        for iou, metrics_at_iou in zip(self.iou_range, self.metrics_range):
            metrics_at_iou.track_in_clearml(cml_logger, split, iou)

    def _track_matrices(self, cml_logger: clearml.Logger, split: str) -> None:
        for iou, matrix_at_iou in zip(self.iou_range, self.matrices_range):
            cml_logger.report_matplotlib_figure(
                title=f'{split.capitalize()} confusion matrices per IoU',
                series=f'IoU:{iou}',
                figure=matrix_at_iou,
            )
