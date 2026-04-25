import argparse
import os
import shutil
import tarfile

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare CIFAR-10 into ImageFolder directories for CROSR"
    )
    parser.add_argument(
        "--download_dir",
        default="./data/raw_cifar10",
        type=str,
        help="Directory used by torchvision to download CIFAR-10",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/cifar10",
        type=str,
        help="Output directory in ImageFolder format",
    )
    parser.add_argument(
        "--known_classes",
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        type=int,
        help="Known classes used for train/val splits",
    )
    parser.add_argument(
        "--unknown_classes",
        nargs="+",
        default=[6, 7, 8, 9],
        type=int,
        help="Unknown classes used for open-set evaluation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before exporting the dataset",
    )
    parser.add_argument(
        "--local_tar",
        default=None,
        type=str,
        help="Path to a local cifar-10-python.tar.gz archive for offline use",
    )
    parser.add_argument(
        "--extracted_dir",
        default=None,
        type=str,
        help="Path to an existing extracted cifar-10-batches-py directory for offline use",
    )
    return parser.parse_args()


def ensure_clean_output(output_dir, overwrite):
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                "{} already exists. Pass --overwrite to rebuild it.".format(output_dir)
            )
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


def save_image(image, split_dir, class_index, image_index):
    class_dir = os.path.join(split_dir, str(class_index))
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, "{:05d}.png".format(image_index))
    image.save(image_path)


def export_split(dataset, allowed_classes, split_dir):
    counters = {class_index: 0 for class_index in allowed_classes}
    for image, label in dataset:
        if label not in allowed_classes:
            continue
        save_image(image, split_dir, label, counters[label])
        counters[label] += 1
    return counters


def load_pickle_file(file_path):
    import pickle

    with open(file_path, "rb") as handle:
        if hasattr(pickle, "DEFAULT_PROTOCOL"):
            return pickle.load(handle, encoding="latin1")
        return pickle.load(handle)


def build_offline_dataset(extracted_dir, train):
    if train:
        batch_files = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]
    else:
        batch_files = ["test_batch"]

    images = []
    labels = []
    for batch_name in batch_files:
        batch = load_pickle_file(os.path.join(extracted_dir, batch_name))
        images.append(batch["data"])
        labels.extend(batch["labels"])

    images = np.vstack(images).reshape(-1, 3, 32, 32)
    images = images.transpose((0, 2, 3, 1))

    dataset = []
    for image_array, label in zip(images, labels):
        dataset.append((Image.fromarray(image_array), label))
    return dataset


def prepare_offline_source(download_dir, local_tar, extracted_dir):
    if extracted_dir:
        if not os.path.isdir(extracted_dir):
            raise FileNotFoundError("Extracted CIFAR-10 directory not found: {}".format(extracted_dir))
        return extracted_dir

    if local_tar:
        if not os.path.isfile(local_tar):
            raise FileNotFoundError("Local CIFAR-10 archive not found: {}".format(local_tar))
        os.makedirs(download_dir, exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as archive:
            archive.extractall(path=download_dir)
        extracted_path = os.path.join(download_dir, "cifar-10-batches-py")
        if not os.path.isdir(extracted_path):
            raise FileNotFoundError("Archive extracted, but cifar-10-batches-py was not found under {}".format(download_dir))
        return extracted_path

    return None


def main():
    args = get_args()

    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)
    overlap = set(known_classes).intersection(set(unknown_classes))
    if overlap:
        raise ValueError("known_classes and unknown_classes overlap: {}".format(sorted(overlap)))

    ensure_clean_output(args.output_dir, args.overwrite)
    extracted_path = prepare_offline_source(args.download_dir, args.local_tar, args.extracted_dir)

    if extracted_path:
        train_dataset = build_offline_dataset(extracted_path, train=True)
        test_dataset = build_offline_dataset(extracted_path, train=False)
    else:
        try:
            train_dataset = CIFAR10(root=args.download_dir, train=True, download=True)
            test_dataset = CIFAR10(root=args.download_dir, train=False, download=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download CIFAR-10 automatically. "
                "If the server is offline, first upload cifar-10-python.tar.gz or the extracted "
                "cifar-10-batches-py directory, then rerun with --local_tar or --extracted_dir. "
                "Original error: {}".format(exc)
            )

    train_counts = export_split(
        train_dataset,
        set(known_classes),
        os.path.join(args.output_dir, "train"),
    )
    val_counts = export_split(
        test_dataset,
        set(known_classes),
        os.path.join(args.output_dir, "val"),
    )
    open_set_counts = export_split(
        test_dataset,
        set(unknown_classes),
        os.path.join(args.output_dir, "open_set"),
    )

    print("Prepared CIFAR-10 dataset at {}".format(args.output_dir))
    print("Known classes: {}".format(known_classes))
    print("Unknown classes: {}".format(unknown_classes))
    print("Train counts: {}".format(train_counts))
    print("Val counts: {}".format(val_counts))
    print("Open-set counts: {}".format(open_set_counts))


if __name__ == "__main__":
    main()
