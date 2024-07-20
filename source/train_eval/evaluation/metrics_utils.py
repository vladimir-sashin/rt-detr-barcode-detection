import fiftyone
from fiftyone import ViewField

from source.data.constants import PREDICT_FIELD


def filter_preds_by_conf(
    preds_dataset: fiftyone.Dataset,
    conf: float,
    pred_field: str = PREDICT_FIELD,
) -> fiftyone.DatasetView:
    return preds_dataset.filter_labels(field=pred_field, filter=ViewField('confidence') >= conf, only_matches=False)
