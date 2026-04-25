import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


DEFAULT_KNOWN_CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
]

DEFAULT_UNKNOWN_CLASSES = [
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "label", "attack_cat"]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare UNSW-NB15 CSV files for 1D DHR training")
    parser.add_argument("--data_dir", default="./unsw_nb15", type=str, help="Directory containing UNSW-NB15 CSV files")
    parser.add_argument("--output_dir", default="./processed_unsw_nb15", type=str, help="Output directory for processed arrays")
    parser.add_argument("--known_classes", nargs="+", default=DEFAULT_KNOWN_CLASSES, type=str, help="Known classes used for training and validation")
    parser.add_argument("--unknown_classes", nargs="+", default=DEFAULT_UNKNOWN_CLASSES, type=str, help="Unknown classes used only for open-set evaluation")
    parser.add_argument("--max_train_per_class", default=None, type=int, help="Optional cap for training samples per known class")
    parser.add_argument("--max_val_per_class", default=None, type=int, help="Optional cap for validation samples per known class")
    parser.add_argument("--max_unknown_per_class", default=None, type=int, help="Optional cap for open-set samples per unknown class")
    parser.add_argument("--seed", default=222, type=int, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    return parser.parse_args()


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


def load_split(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("Expected CSV not found: {}".format(csv_path))
    frame = pd.read_csv(csv_path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "attack_cat" not in frame.columns:
        raise KeyError("Expected an 'attack_cat' column in {}".format(csv_path))
    frame["attack_cat"] = frame["attack_cat"].fillna("Normal").astype(str).str.strip()
    return frame


def sample_per_class(frame, class_names, limit_per_class, rng):
    if limit_per_class is None:
        return frame.reset_index(drop=True)

    sampled_frames = []
    for class_name in class_names:
        class_rows = frame[frame["attack_cat"] == class_name]
        if class_rows.empty:
            continue
        sample_size = min(limit_per_class, len(class_rows))
        sampled_frames.append(class_rows.sample(n=sample_size, random_state=rng.randint(0, 2**31 - 1)))
    if not sampled_frames:
        return frame.iloc[0:0].copy()
    return pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)


def build_feature_matrices(train_frame, eval_frames):
    feature_columns = [column for column in train_frame.columns if column not in DROP_COLUMNS]
    categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    train_numeric = train_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(numeric_imputer.fit_transform(train_numeric)).astype(np.float32)

    encoder = None
    train_categorical = None
    if categorical_columns:
        train_categorical_frame = train_frame[categorical_columns].fillna("-").astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        train_categorical = encoder.fit_transform(train_categorical_frame).astype(np.float32)

    transformed = {}
    for split_name, frame in [("train", train_frame)] + list(eval_frames.items()):
        numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_values = scaler.transform(numeric_imputer.transform(numeric_frame)).astype(np.float32)

        if categorical_columns:
            categorical_frame = frame[categorical_columns].fillna("-").astype(str)
            categorical_values = encoder.transform(categorical_frame).astype(np.float32)
            x = np.concatenate([numeric_values, categorical_values], axis=1)
        else:
            categorical_values = None
            x = numeric_values

        transformed[split_name] = x.astype(np.float32)

    feature_names = list(numeric_columns)
    if categorical_columns:
        feature_names.extend(encoder.get_feature_names_out(categorical_columns).tolist())

    preprocessing = {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names": feature_names,
        "numeric_imputer": numeric_imputer,
        "scaler": scaler,
        "encoder": encoder,
    }
    return transformed, preprocessing


def main():
    args = get_args()
    rng = np.random.RandomState(args.seed)

    known_classes = [str(label).strip() for label in args.known_classes]
    unknown_classes = [str(label).strip() for label in args.unknown_classes]
    overlap = sorted(set(known_classes).intersection(set(unknown_classes)))
    if overlap:
        raise ValueError("known_classes and unknown_classes overlap: {}".format(overlap))

    ensure_output_dir(args.output_dir, args.overwrite)

    train_csv = os.path.join(args.data_dir, "UNSW_NB15_training-set.csv")
    test_csv = os.path.join(args.data_dir, "UNSW_NB15_testing-set.csv")

    print("Loading {}".format(os.path.basename(train_csv)))
    train_frame = load_split(train_csv)
    print("Loading {}".format(os.path.basename(test_csv)))
    test_frame = load_split(test_csv)

    train_known = train_frame[train_frame["attack_cat"].isin(known_classes)].copy()
    val_known = test_frame[test_frame["attack_cat"].isin(known_classes)].copy()
    open_unknown = test_frame[test_frame["attack_cat"].isin(unknown_classes)].copy()

    train_known = sample_per_class(train_known, known_classes, args.max_train_per_class, rng)
    val_known = sample_per_class(val_known, known_classes, args.max_val_per_class, rng)
    open_unknown = sample_per_class(open_unknown, unknown_classes, args.max_unknown_per_class, rng)

    if train_known.empty or val_known.empty or open_unknown.empty:
        raise RuntimeError("Insufficient samples collected. Check class names, dataset files, and caps.")

    known_label_to_index = {label: idx for idx, label in enumerate(known_classes)}
    unknown_label_to_index = {label: idx for idx, label in enumerate(unknown_classes)}

    transformed, preprocessing = build_feature_matrices(
        train_known,
        {
            "val": val_known,
            "open": open_unknown,
        },
    )

    y_train = train_known["attack_cat"].map(known_label_to_index).to_numpy(dtype=np.int64)
    y_val = val_known["attack_cat"].map(known_label_to_index).to_numpy(dtype=np.int64)
    y_open = open_unknown["attack_cat"].map(unknown_label_to_index).to_numpy(dtype=np.int64)

    np.savez(
        os.path.join(args.output_dir, "train_known.npz"),
        x=transformed["train"],
        y=y_train,
        label_names=np.array(known_classes),
        feature_names=np.array(preprocessing["feature_names"]),
    )
    np.savez(
        os.path.join(args.output_dir, "val_known.npz"),
        x=transformed["val"],
        y=y_val,
        label_names=np.array(known_classes),
        feature_names=np.array(preprocessing["feature_names"]),
    )
    np.savez(
        os.path.join(args.output_dir, "open_set.npz"),
        x=transformed["open"],
        y=y_open,
        label_names=np.array(unknown_classes),
        feature_names=np.array(preprocessing["feature_names"]),
    )

    scaler = preprocessing["scaler"]
    encoder = preprocessing["encoder"]
    np.savez(
        os.path.join(args.output_dir, "preprocessing.npz"),
        feature_names=np.array(preprocessing["feature_names"]),
        feature_columns=np.array(preprocessing["feature_columns"]),
        numeric_columns=np.array(preprocessing["numeric_columns"]),
        categorical_columns=np.array(preprocessing["categorical_columns"]),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        encoder_categories=np.array(encoder.categories_ if encoder is not None else [], dtype=object),
    )

    metadata = {
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "train_counts": train_known["attack_cat"].value_counts().sort_index().to_dict(),
        "val_counts": val_known["attack_cat"].value_counts().sort_index().to_dict(),
        "open_counts": open_unknown["attack_cat"].value_counts().sort_index().to_dict(),
        "num_features": int(transformed["train"].shape[1]),
        "categorical_columns": preprocessing["categorical_columns"],
        "numeric_columns": preprocessing["numeric_columns"],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved processed UNSW-NB15 data to {}".format(args.output_dir))
    print("x_train:", transformed["train"].shape, "x_val:", transformed["val"].shape, "x_open:", transformed["open"].shape)
    print("Known classes:", known_classes)
    print("Unknown classes:", unknown_classes)


if __name__ == "__main__":
    main()
