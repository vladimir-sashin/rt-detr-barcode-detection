# RT-DETR Barcode Detection Model Training using ClearML Experiment Tracking

<a href="https://clear.ml/docs/latest/"><img alt="ClearML" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

This repo is a ready-to-use framework for training and evaluation of pretrained pyTorch RT-DETR ([paper](https://arxiv.org/abs/2304.08069), [official repo](https://github.com/lyuwenyu/RT-DETR)) models on custom data in COCO format.
Using this repo, barcodes detection task can be solved out-of-the box in just 2 CLI commands, but this project is capable of training a model on any custom object detection dataset that follows COCO format.

**The project features:**

1. The official RT DETR implementation with a few additional changes in the original code that enable more detailed logging.
1. Full-scale experiment tracking in [CLearML](https://clear.ml/) including:
   1. Config and hyperparameter tracking
   1. Model artifacts versioning (`.pt`, `.onnx`)
   1. Tracking of numerous losses, metrics (see below) and visualizations of batches and model predicts.
1. End-to-end RT DETR model training and evaluation pipeline that:
   1. Has a clean and easy configuration in a single `yaml` config file.
   1. Optionally pulls the dataset from ClearML.
   1. Downloads RT DETR pretrained model and trains it on custom dataset in COCO format.
   1. Finds the best checkpoint (epoch) based on mAP on validation set.
   1. Finds of the best confidence threshold value that maximizes the f-1 score on validation set.
   1. Evaluates a bunch of metrics on both validation and test sets for a range of IoU thresholds: mAP COCO evaluation, confusion matrix, precision/recall/f-1.
   1. Exports the best checkpoint to [ONNX](https://onnx.ai/).
1. Barcode detection dataset preprocessing pipeline, featuring data versioning in ClearML.

Check out [the example of barcode detection model training experiment in ClearML](https://app.clear.ml/projects/d64acf44e28d43fb924d0bce24a55d48/experiments/80e0289d086a4dbebce878cb457ce3e2/output/execution).

______________________________________________________________________

## Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org) to install Poetry. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace.
   - Unix:
   ```bash
   make setup_ws
   ```
   - Windows:
   ```bash
   make setup_ws PYTHON_EXEC=<path_to_your_python_3.10_executable>
   ```
1. Activate poetry virtual environment
   ```bash
   poetry shell
   ```

______________________________________________________________________

# Model Training

## I. Barcode Detection

### 1. Configure (or skip to use defaults, which is perfectly fine)

1. [Data config](configs/data.yaml) (`configs/data.yaml`) to set how to split the data and whether to version it in ClearML.
1. [Train and evaluation config](configs/train_eval.yaml) (`configs/train_eval.yaml`) to set everything else: use local dataset or from ClearML, which pretrain to use, hyperparameters, ClearML tracking settings, etc.

### 2.  Run data pipeline

To download and preprocess barcodes detection dataset

```bash
make run_data_pipe
```

OR the same thing in 2 steps:

```bash
make fetch_data
make prep_data
```

### 3. Run training and evaluation pipeline

```bash
make run_train_eval
```

That's it, RT DETR goes brrr, and you'll be able to see all the results and your model in ClearML, already trained and exported to ONNX.

### 4. \[Alternatively\] Run end-to-end pipeline

To run everything at once in a single line: fetch data + preprocess data + train and evaluate pretrained RT DETR model.

```bash
make run_e2e_pipeline
```

## II. Train detection model on custom data

1. Convert data to COCO object detection format that follows this structure:

```
dataset_name:
   train:
      data:
         img0.jpg, img1.jpg, ...
      labels.json
   val:
      data:
         img0.jpg, img1.jpg, ...
      labels.json
   test:
      data:
         img0.jpg, img1.jpg, ...
      labels.json
```

2. Upload dataset to ClearML using CLI or python SDK OR put it in `datasets` dir.
1. Set `data_source_cfg` in [Train and evaluation config](configs/train_eval.yaml) (`configs/train_eval.yaml`):
   - If you want to use dataset from CLearML, set `storage: clearml` and fill other fields.
   - If you want to use dataset from `datasets` dir, set `storage: local` and `dataset_name: {dataset_dir_name}`

______________________________________________________________________

# Acknowledgments

1. Firstly, thanks a lot to the authors of the amazing RT DETR architecture, that is so fast and well-performing that I managed to train a decent model on CPU in just a few hours using a small dataset.
1. Secondly, shoutout to the team of [DeepShchool's](https://deepschool.ru/) [CV Rocket course](https://deepschool.ru/cvrocket), where I (hopefully) learned best practices of ML development and ClearML experiment tracking. BTW this is one of the course's graduation projects :)

______________________________________________________________________

# TODO List

1. Add CI.
1. Add sanity and some unit tests.
1. Add conversion of the model to [OpenVino](https://docs.openvino.ai/2024/index.html) to the pipeline.
