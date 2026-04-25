# CICIDS 1D Experiment Guide

This document records the complete 1D CROSR/OpenMax workflow for the CICIDS experiment in this repository.

It is intended for the CICIDS-style tabular traffic pipeline using:

- `prepare_cicids_1d.py`
- `train_net_1d.py`
- `get_model_features_1d.py`
- `MAV_Compute.py`
- `compute_distances.py`
- `compute_openmax.py`
- `evaluate_openmax.py`

The recommended setup uses two conda environments:

- `crosr-py3` for preprocessing, training, and feature extraction
- `crosr-py27` for `libMR` and OpenMax evaluation

## 1. Environments

Create the environments once:

```bash
conda env create -f environment-py3.yml
conda env create -f environment-py27.yml
```

Build `libMR` in the Python 2.7 environment:

```bash
conda activate crosr-py27
cd libMR
python setup.py build_ext -i
cd ..
```

## 2. Dataset Layout

Place the raw CICIDS CSV files under:

```text
./cicids/
```

The preprocessing script will scan every CSV file in that directory and generate three `.npz` files:

- `train_known.npz`
- `val_known.npz`
- `open_set.npz`

## 3. Preprocess CICIDS

Run:

```bash
conda activate crosr-py3
python prepare_cicids_1d.py \
  --data_dir ./cicids \
  --output_dir ./processed_cicids \
  --overwrite
```

This step:

- reads the raw CICIDS CSV files
- cleans invalid numeric rows
- standardizes the features
- builds the known/open split
- saves processed arrays for 1D training

Expected output files:

```text
./processed_cicids/train_known.npz
./processed_cicids/val_known.npz
./processed_cicids/open_set.npz
./processed_cicids/scaler.npz
./processed_cicids/metadata.json
```

## 4. Train the 1D DHR Model

Recommended baseline training command:

```bash
conda activate crosr-py3
python train_net_1d.py \
  --train_path ./processed_cicids/train_known.npz \
  --val_path ./processed_cicids/val_known.npz \
  --save_path ./save_models/cicids_1d \
  --epochs 40
```

Useful notes:

- If `--epochs` is omitted, the default is `100`.
- Checkpoints are saved under `./save_models/cicids_1d`.
- The latest checkpoint is:

```text
./save_models/cicids_1d/latest.pth
```

Optional knobs worth trying:

```bash
--optimizer adamw
--scheduler cosine
--base_channels 128
--hidden_dim 512
```

## 5. Extract Features

After training, extract features for:

- training known set
- validation known set
- open set

Run:

```bash
conda activate crosr-py3
python get_model_features_1d.py \
  --train_path ./processed_cicids/train_known.npz \
  --val_path ./processed_cicids/val_known.npz \
  --open_path ./processed_cicids/open_set.npz \
  --load_path ./save_models/cicids_1d/latest.pth \
  --save_path ./saved_features/cicids_1d
```

Expected layout:

```text
./saved_features/cicids_1d/train/
./saved_features/cicids_1d/val/
./saved_features/cicids_1d/open_set/
```

## 6. Compute MAVs

Run:

```bash
conda activate crosr-py3
python MAV_Compute.py \
  --feature_dir ./saved_features/cicids_1d/train \
  --save_path ./saved_MAVs/cicids_1d
```

This computes one MAV per known class from correctly classified training features.

## 7. Compute Distance Scores

Run:

```bash
conda activate crosr-py3
python compute_distances.py \
  --feature_dir ./saved_features/cicids_1d/train \
  --MAV_path ./saved_MAVs/cicids_1d \
  --save_path ./saved_distance_scores/cicids_1d
```

## 8. Run OpenMax

Switch to the Python 2.7 environment:

```bash
conda activate crosr-py27
```

Baseline OpenMax command:

```bash
PYTHONPATH=./libMR python compute_openmax.py \
  --MAV_path ./saved_MAVs/cicids_1d \
  --distance_scores_path ./saved_distance_scores/cicids_1d \
  --feature_dir ./saved_features/cicids_1d \
  --alpha_rank 3 \
  --weibull_tail_size 20 \
  --distance_type euclidean
```

This prints the AUROC based on the OpenMax unknown probability.

## 9. Evaluate Full Metrics

Run:

```bash
PYTHONPATH=./libMR python evaluate_openmax.py \
  --MAV_path ./saved_MAVs/cicids_1d \
  --distance_scores_path ./saved_distance_scores/cicids_1d \
  --feature_dir ./saved_features/cicids_1d \
  --alpha_rank 3 \
  --weibull_tail_size 20 \
  --distance_type euclidean
```

This reports:

- `AUROC`
- `AUPR_OUT`
- `AUPR_IN`
- `FPR@95TPR`
- best balanced-accuracy threshold metrics

## 10. Recommended OpenMax Settings

For the CICIDS experiment, Euclidean distance worked much better than the default `eucos` in our tuning runs.

Strong settings to try:

### Balanced setting

```text
tail = 15
alpha = 3
distance = euclidean
```

### AUROC-oriented setting

```text
tail = 28
alpha = 3
distance = euclidean
```

Example:

```bash
PYTHONPATH=./libMR python evaluate_openmax.py \
  --MAV_path ./saved_MAVs/cicids_1d \
  --distance_scores_path ./saved_distance_scores/cicids_1d \
  --feature_dir ./saved_features/cicids_1d \
  --alpha_rank 3 \
  --weibull_tail_size 15 \
  --distance_type euclidean
```

## 11. Optional Parameter Scan

You can scan multiple OpenMax settings with:

```bash
conda activate crosr-py27
PYTHONPATH=./libMR python scan_openmax_params.py \
  --MAV_path ./saved_MAVs/cicids_1d \
  --distance_scores_path ./saved_distance_scores/cicids_1d \
  --feature_dir ./saved_features/cicids_1d
```

Typical search dimensions:

- `tail_size`
- `alpha_rank`
- `distance_type`

## 12. Recommended Run Order Summary

```bash
conda activate crosr-py3
python prepare_cicids_1d.py --data_dir ./cicids --output_dir ./processed_cicids --overwrite
python train_net_1d.py --train_path ./processed_cicids/train_known.npz --val_path ./processed_cicids/val_known.npz --save_path ./save_models/cicids_1d --epochs 40
python get_model_features_1d.py --train_path ./processed_cicids/train_known.npz --val_path ./processed_cicids/val_known.npz --open_path ./processed_cicids/open_set.npz --load_path ./save_models/cicids_1d/latest.pth --save_path ./saved_features/cicids_1d
python MAV_Compute.py --feature_dir ./saved_features/cicids_1d/train --save_path ./saved_MAVs/cicids_1d
python compute_distances.py --feature_dir ./saved_features/cicids_1d/train --MAV_path ./saved_MAVs/cicids_1d --save_path ./saved_distance_scores/cicids_1d

conda activate crosr-py27
PYTHONPATH=./libMR python compute_openmax.py --MAV_path ./saved_MAVs/cicids_1d --distance_scores_path ./saved_distance_scores/cicids_1d --feature_dir ./saved_features/cicids_1d --alpha_rank 3 --weibull_tail_size 20 --distance_type euclidean
PYTHONPATH=./libMR python evaluate_openmax.py --MAV_path ./saved_MAVs/cicids_1d --distance_scores_path ./saved_distance_scores/cicids_1d --feature_dir ./saved_features/cicids_1d --alpha_rank 3 --weibull_tail_size 15 --distance_type euclidean
```

## 13. Common Failure Modes

- `No correctly classified features found for class X`
  The checkpoint is weak, the label mapping is inconsistent, or the wrong feature split is being used.

- `compute_openmax.py` reports overflow or `NaN`
  Use the current repository version, which includes a numerically stable OpenMax probability computation.

- `KeyError` in `evt_fitting.py` or `compute_openmax.py`
  A class is missing from the processed data, feature directory, or MAV directory.

- `unrecognized arguments: --MAV_save_path` or `--distance_path`
  The older scripts use `--save_path`, not dataset-specific save argument names.

## 14. Notes

- Keep the CICIDS experiment outputs in their own directories to avoid mixing them with CICIDS2018, UNSW-NB15, or NSL-KDD runs.
- If you want to compare multiple variants, use suffixes such as `_v2`, `_improved`, or `_scan1`.
