import sys

from source.train_eval.clearml_tracking import setup_clearml, track_artifacts
from source.train_eval.config.train_eval_cfg import (
    TrainEvalConfig,
    get_train_cfg,
)
from source.train_eval.constants import RTDETR_SRC_PATH
from source.train_eval.evaluation.evaluate import (
    evaluate_f1_conf_on_val,
    evaluate_f1_on_test,
    evaluate_map_on_test,
)
from source.train_eval.model_export import export_onnx
from source.train_eval.training.train import run_training

sys.path.append(str(RTDETR_SRC_PATH))


def train(cfg: TrainEvalConfig) -> None:
    cfg, task = setup_clearml(cfg)

    training_output = run_training(cfg)
    onnx_model_path = export_onnx(training_output, cfg)

    val_eval_results = evaluate_f1_conf_on_val(onnx_model_path, cfg)
    evaluate_map_on_test(cfg, training_output.best_checkpoint)
    test_eval_results = evaluate_f1_on_test(onnx_model_path, cfg, val_eval_results.conf_thres)

    track_artifacts(task, training_output.best_checkpoint, onnx_model_path, val_eval_results, test_eval_results)


if __name__ == '__main__':
    config = get_train_cfg()
    train(config)
