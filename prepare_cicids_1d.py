import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_KNOWN_CLASSES = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
]

DEFAULT_UNKNOWN_CLASSES = [
    "Bot",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Heartbleed",
    "Infiltration",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare CICIDS CSV files for 1D DHR training")
    parser.add_argument("--data_dir", default="./cicids", type=str, help="Directory containing CICIDS CSV files")
    parser.add_argument("--output_dir", default="./processed_cicids", type=str, help="Output directory for processed arrays")
    parser.add_argument("--known_classes", nargs="+", default=DEFAULT_KNOWN_CLASSES, type=str, help="Known classes used for training")
    parser.add_argument("--unknown_classes", nargs="+", default=DEFAULT_UNKNOWN_CLASSES, type=str, help="Unknown classes used only for open-set evaluation")
    parser.add_argument("--val_fraction", default=0.2, type=float, help="Validation fraction for known classes")
    parser.add_argument("--max_train_per_class", default=20000, type=int, help="Maximum known-class training samples per class")
    parser.add_argument("--max_val_per_class", default=5000, type=int, help="Maximum known-class validation samples per class")
    parser.add_argument("--max_unknown_per_class", default=5000, type=int, help="Maximum unknown-class evaluation samples per class")
    parser.add_argument("--chunksize", default=50000, type=int, help="CSV chunk size")
    parser.add_argument("--seed", default=222, type=int, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    return parser.parse_args()


def canonicalize_label(label):
    label = str(label).strip()
    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � Sql Injection": "Web Attack - Sql Injection",
        "Web Attack � XSS": "Web Attack - XSS",
    }
    return replacements.get(label, label)


def ensure_output_dir(output_dir, overwrite):
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError("{} already exists. Pass --overwrite to rebuild it.".format(output_dir))
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file_name in files:
                os.remove(os.path.join(root, file_name))
            for dir_name in dirs:
                os.rmdir(os.path.join(root, dir_name))
    os.makedirs(output_dir, exist_ok=True)


def append_sample(bucket_x, bucket_y, row_values, label_index):
    bucket_x.append(row_values.astype(np.float32, copy=False))
    bucket_y.append(label_index)


def maybe_add_known_samples(frame, feature_columns, label_name, label_index, args, rng,
                            train_x, train_y, val_x, val_y, train_counts, val_counts):
    if label_name not in train_counts:
        train_counts[label_name] = 0
        val_counts[label_name] = 0

    label_rows = frame[frame["Label"] == label_name]
    if label_rows.empty:
        return

    label_rows = label_rows.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1))
    values = label_rows[feature_columns].to_numpy(dtype=np.float32, copy=False)

    for row_values in values:
        choose_val = (rng.rand() < args.val_fraction) and (val_counts[label_name] < args.max_val_per_class)
        if choose_val:
            append_sample(val_x, val_y, row_values, label_index)
            val_counts[label_name] += 1
        elif train_counts[label_name] < args.max_train_per_class:
            append_sample(train_x, train_y, row_values, label_index)
            train_counts[label_name] += 1
        elif val_counts[label_name] < args.max_val_per_class:
            append_sample(val_x, val_y, row_values, label_index)
            val_counts[label_name] += 1

        if train_counts[label_name] >= args.max_train_per_class and val_counts[label_name] >= args.max_val_per_class:
            break


def maybe_add_unknown_samples(frame, feature_columns, label_name, label_index, args, rng,
                              open_x, open_y, open_counts):
    if label_name not in open_counts:
        open_counts[label_name] = 0

    label_rows = frame[frame["Label"] == label_name]
    if label_rows.empty:
        return

    label_rows = label_rows.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1))
    values = label_rows[feature_columns].to_numpy(dtype=np.float32, copy=False)

    for row_values in values:
        if open_counts[label_name] >= args.max_unknown_per_class:
            break
        append_sample(open_x, open_y, row_values, label_index)
        open_counts[label_name] += 1


def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    known_classes = [canonicalize_label(label) for label in args.known_classes]
    unknown_classes = [canonicalize_label(label) for label in args.unknown_classes]
    overlap = set(known_classes).intersection(set(unknown_classes))
    if overlap:
        raise ValueError("known_classes and unknown_classes overlap: {}".format(sorted(overlap)))

    ensure_output_dir(args.output_dir, args.overwrite)

    csv_files = sorted(
        os.path.join(args.data_dir, file_name)
        for file_name in os.listdir(args.data_dir)
        if file_name.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError("No CSV files found under {}".format(args.data_dir))

    feature_columns = None
    train_x, train_y = [], []
    val_x, val_y = [], []
    open_x, open_y = [], []

    known_label_to_index = {label: idx for idx, label in enumerate(known_classes)}
    unknown_label_to_index = {label: idx for idx, label in enumerate(unknown_classes)}
    train_counts = {}
    val_counts = {}
    open_counts = {}

    for csv_path in csv_files:
        print("Processing {}".format(os.path.basename(csv_path)))
        with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
            for chunk in pd.read_csv(handle, chunksize=args.chunksize, low_memory=False):
                chunk.columns = [str(col).strip() for col in chunk.columns]
                if "Label" not in chunk.columns:
                    raise KeyError("Expected a 'Label' column in {}".format(csv_path))

                if feature_columns is None:
                    feature_columns = [col for col in chunk.columns if col != "Label"]

                chunk["Label"] = chunk["Label"].map(canonicalize_label)
                filtered = chunk[chunk["Label"].isin(set(known_classes) | set(unknown_classes))].copy()
                if filtered.empty:
                    continue

                filtered[feature_columns] = filtered[feature_columns].apply(pd.to_numeric, errors="coerce")
                filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
                filtered.dropna(subset=feature_columns, inplace=True)
                if filtered.empty:
                    continue

                for label_name, label_index in known_label_to_index.items():
                    maybe_add_known_samples(
                        filtered, feature_columns, label_name, label_index, args, rng,
                        train_x, train_y, val_x, val_y, train_counts, val_counts
                    )

                for label_name, label_index in unknown_label_to_index.items():
                    maybe_add_unknown_samples(
                        filtered, feature_columns, label_name, label_index, args, rng,
                        open_x, open_y, open_counts
                    )

    if not train_x or not val_x or not open_x:
        raise RuntimeError("Insufficient samples collected. Check class names and caps.")

    x_train = np.stack(train_x).astype(np.float32)
    y_train = np.array(train_y, dtype=np.int64)
    x_val = np.stack(val_x).astype(np.float32)
    y_val = np.array(val_y, dtype=np.int64)
    x_open = np.stack(open_x).astype(np.float32)
    y_open = np.array(open_y, dtype=np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_open = scaler.transform(x_open).astype(np.float32)

    np.savez(
        os.path.join(args.output_dir, "train_known.npz"),
        x=x_train,
        y=y_train,
        label_names=np.array(known_classes),
        feature_names=np.array(feature_columns),
    )
    np.savez(
        os.path.join(args.output_dir, "val_known.npz"),
        x=x_val,
        y=y_val,
        label_names=np.array(known_classes),
        feature_names=np.array(feature_columns),
    )
    np.savez(
        os.path.join(args.output_dir, "open_set.npz"),
        x=x_open,
        y=y_open,
        label_names=np.array(unknown_classes),
        feature_names=np.array(feature_columns),
    )
    np.savez(
        os.path.join(args.output_dir, "scaler.npz"),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        feature_names=np.array(feature_columns),
    )

    metadata = {
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "open_counts": open_counts,
        "num_features": len(feature_columns),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved processed CICIDS data to {}".format(args.output_dir))
    print("x_train:", x_train.shape, "x_val:", x_val.shape, "x_open:", x_open.shape)
    print("Known classes:", known_classes)
    print("Unknown classes:", unknown_classes)


if __name__ == "__main__":
    main()
