import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_KNOWN_CLASSES = [
    "benign",
    "ddos attack-hoic",
    "ddos attack-loic-udp",
    "ddos attacks-loic-http",
    "dos attacks-goldeneye",
    "dos attacks-hulk",
    "dos attacks-slowhttptest",
    "dos attacks-slowloris",
]

DEFAULT_UNKNOWN_CLASSES = [
    "bot",
    "infilteration",
    "brute force -web",
    "brute force -xss",
    "ftp-bruteforce",
    "ssh-bruteforce",
    "sql injection",
]

DROP_COLUMNS = ["Label", "Timestamp", "Flow ID", "Src IP", "Dst IP"]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare CICIDS2018 CSV files for 1D DHR training")
    parser.add_argument("--data_dir", default="./cicids2018", type=str, help="Directory containing CICIDS2018 CSV files")
    parser.add_argument("--output_dir", default="./processed_cicids2018", type=str, help="Output directory for processed arrays")
    parser.add_argument("--known_classes", nargs="+", default=DEFAULT_KNOWN_CLASSES, type=str, help="Known classes used for training")
    parser.add_argument("--unknown_classes", nargs="+", default=DEFAULT_UNKNOWN_CLASSES, type=str, help="Unknown classes used only for open-set evaluation")
    parser.add_argument("--val_fraction", default=0.2, type=float, help="Validation fraction for known classes")
    parser.add_argument("--max_train_per_class", default=20000, type=int, help="Maximum known-class training samples per class")
    parser.add_argument("--max_val_per_class", default=5000, type=int, help="Maximum known-class validation samples per class")
    parser.add_argument("--max_unknown_per_class", default=5000, type=int, help="Maximum unknown-class evaluation samples per class")
    parser.add_argument("--chunksize", default=50000, type=int, help="CSV chunk size")
    parser.add_argument("--seed", default=222, type=int, help="Random seed")
    parser.add_argument("--split_mode", default="chronological", choices=["chronological", "random"], help="How to split known samples into train and validation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    return parser.parse_args()


def canonicalize_label(label):
    label = str(label).strip().lower()
    label = label.lstrip("\ufeff")
    label = label.replace("_", " ").replace("\t", " ")
    label = re.sub(r"\s*-\s*", "-", label)
    label = re.sub(r"\s+", " ", label)

    replacements = {
        "benign": "benign",
        "label": "__header__",
        "ddos attack-hoic": "ddos attack-hoic",
        "ddos attacks-hoic": "ddos attack-hoic",
        "ddos attack hoic": "ddos attack-hoic",
        "ddos attack loic-udp": "ddos attack-loic-udp",
        "ddos attack-loic-udp": "ddos attack-loic-udp",
        "ddos attack loic udp": "ddos attack-loic-udp",
        "ddos attacks-loic-http": "ddos attacks-loic-http",
        "ddos attacks loic-http": "ddos attacks-loic-http",
        "ddos attacks loic http": "ddos attacks-loic-http",
        "dos attacks-hulk": "dos attacks-hulk",
        "dos attacks hulk": "dos attacks-hulk",
        "dos attacks-goldeneye": "dos attacks-goldeneye",
        "dos attacks goldeneye": "dos attacks-goldeneye",
        "dos attacks-slowhttptest": "dos attacks-slowhttptest",
        "dos attacks slowhttptest": "dos attacks-slowhttptest",
        "dos attacks-slowloris": "dos attacks-slowloris",
        "dos attacks slowloris": "dos attacks-slowloris",
        "ftp-bruteforce": "ftp-bruteforce",
        "ssh-bruteforce": "ssh-bruteforce",
        "infilteration": "infilteration",
        "infiltration": "infilteration",
        "bot": "bot",
        "sql injection": "sql injection",
        "brute force -web": "brute force -web",
        "brute force -xss": "brute force -xss",
        "brute force - web": "brute force -web",
        "brute force - xss": "brute force -xss",
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


def clean_chunk(chunk, feature_columns=None):
    chunk.columns = [str(col).strip() for col in chunk.columns]
    if "Label" not in chunk.columns:
        raise KeyError("Expected a 'Label' column in input CSV")

    if feature_columns is None:
        feature_columns = [col for col in chunk.columns if col not in DROP_COLUMNS]

    chunk["Label"] = chunk["Label"].map(canonicalize_label)
    chunk = chunk[chunk["Label"] != "__header__"].copy()
    chunk[feature_columns] = chunk[feature_columns].apply(pd.to_numeric, errors="coerce")
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(subset=feature_columns, inplace=True)
    return chunk, feature_columns


def find_missing_classes(target_classes, counts):
    return [label for label in target_classes if counts.get(label, 0) <= 0]


def compute_file_known_counts(csv_path, known_classes, chunksize):
    counts = {label: 0 for label in known_classes}
    feature_columns = None
    with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize, low_memory=False):
            chunk, feature_columns = clean_chunk(chunk, feature_columns)
            if chunk.empty:
                continue
            filtered = chunk[chunk["Label"].isin(known_classes)]
            if filtered.empty:
                continue
            label_counts = filtered["Label"].value_counts()
            for label_name, count in label_counts.items():
                counts[label_name] += int(count)
    return counts


def compute_known_file_targets(file_counts, args, known_classes, train_counts, val_counts):
    targets = {}
    for label_name in known_classes:
        total_available = file_counts.get(label_name, 0)
        if total_available <= 0:
            targets[label_name] = {"train": 0, "val": 0}
            continue

        train_cap_left = max(0, args.max_train_per_class - train_counts.get(label_name, 0))
        val_cap_left = max(0, args.max_val_per_class - val_counts.get(label_name, 0))

        file_val_target = int(round(total_available * args.val_fraction))
        file_val_target = min(file_val_target, total_available)
        file_train_target = total_available - file_val_target

        train_target = min(file_train_target, train_cap_left)
        remaining = total_available - train_target
        val_target = min(file_val_target, val_cap_left, remaining)

        if train_target < train_cap_left and remaining - val_target > 0:
            extra_train = min(train_cap_left - train_target, remaining - val_target)
            train_target += extra_train

        targets[label_name] = {"train": train_target, "val": val_target}
    return targets


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
        file_targets = None
        file_progress = None
        if args.split_mode == "chronological":
            file_known_counts = compute_file_known_counts(csv_path, set(known_classes), args.chunksize)
            file_targets = compute_known_file_targets(file_known_counts, args, known_classes, train_counts, val_counts)
            file_progress = {label: 0 for label in known_classes}

        with open(csv_path, "r", encoding="utf-8", errors="replace") as handle:
            for chunk in pd.read_csv(handle, chunksize=args.chunksize, low_memory=False):
                chunk, feature_columns = clean_chunk(chunk, feature_columns)
                filtered = chunk[chunk["Label"].isin(set(known_classes) | set(unknown_classes))].copy()
                if filtered.empty:
                    continue

                if args.split_mode == "random":
                    for label_name, label_index in known_label_to_index.items():
                        label_rows = filtered[filtered["Label"] == label_name]
                        if label_rows.empty:
                            continue
                        if label_name not in train_counts:
                            train_counts[label_name] = 0
                            val_counts[label_name] = 0

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
                else:
                    for label_name, label_index in known_label_to_index.items():
                        label_rows = filtered[filtered["Label"] == label_name]
                        if label_rows.empty:
                            continue
                        if label_name not in train_counts:
                            train_counts[label_name] = 0
                            val_counts[label_name] = 0
                        targets = file_targets[label_name]
                        train_target = targets["train"]
                        val_target = targets["val"]
                        if train_target <= 0 and val_target <= 0:
                            continue

                        values = label_rows[feature_columns].to_numpy(dtype=np.float32, copy=False)
                        for row_values in values:
                            seen = file_progress[label_name]
                            if seen < train_target:
                                append_sample(train_x, train_y, row_values, label_index)
                                train_counts[label_name] += 1
                            elif seen < train_target + val_target:
                                append_sample(val_x, val_y, row_values, label_index)
                                val_counts[label_name] += 1
                            else:
                                break
                            file_progress[label_name] += 1

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

    missing_known_train = find_missing_classes(known_classes, train_counts)
    missing_known_val = find_missing_classes(known_classes, val_counts)
    missing_unknown_open = find_missing_classes(unknown_classes, open_counts)
    if missing_known_train or missing_known_val or missing_unknown_open:
        raise RuntimeError(
            "Missing samples after preprocessing. "
            "missing_known_train={}, missing_known_val={}, missing_unknown_open={}".format(
                missing_known_train, missing_known_val, missing_unknown_open
            )
        )

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
        "split_mode": args.split_mode,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "open_counts": open_counts,
        "num_features": len(feature_columns),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved processed CICIDS2018 data to {}".format(args.output_dir))
    print("x_train:", x_train.shape, "x_val:", x_val.shape, "x_open:", x_open.shape)
    print("Known classes:", known_classes)
    print("Unknown classes:", unknown_classes)


if __name__ == "__main__":
    main()
