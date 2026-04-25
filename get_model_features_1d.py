import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from DHR_Net_1D import DHRNet1D


def get_args():
    parser = argparse.ArgumentParser(description='Get 1D activation vectors')
    parser.add_argument('--train_path', default="./processed_cicids/train_known.npz", type=str, help="Path to known-class training data")
    parser.add_argument('--val_path', default="./processed_cicids/val_known.npz", type=str, help="Path to known-class validation data")
    parser.add_argument('--open_path', default="./processed_cicids/open_set.npz", type=str, help="Path to open-set evaluation data")
    parser.add_argument('--save_path', default="./saved_features/cicids_1d", type=str, help="Directory for saved features")
    parser.add_argument('--load_path', default="./save_models/cicids_1d/latest.pth", type=str, help="Path to the trained 1D checkpoint")
    parser.add_argument('--input_channels', default=None, type=int, help="Number of 1D input channels. If omitted, try to read from checkpoint.")
    parser.add_argument('--base_channels', default=None, type=int, help="Base number of convolution channels. If omitted, try to read from checkpoint.")
    parser.add_argument('--hidden_dim', default=None, type=int, help="Hidden dimension in the classifier head. If omitted, try to read from checkpoint.")
    parser.add_argument('--batch_size', default=512, type=int, help="Batch size for feature extraction")
    return parser.parse_args()


def to_sequence_tensor(array, input_channels):
    x = torch.from_numpy(array).float()
    if input_channels == 1:
        return x.unsqueeze(1)
    if x.size(1) % input_channels != 0:
        raise ValueError("Feature dimension {} is not divisible by input_channels {}".format(x.size(1), input_channels))
    sequence_length = x.size(1) // input_channels
    return x.view(x.size(0), input_channels, sequence_length)


def load_npz_dataset(npz_path, input_channels):
    data = np.load(npz_path, allow_pickle=True)
    x = to_sequence_tensor(data["x"], input_channels)
    y = torch.from_numpy(data["y"]).long()
    label_names = [str(label) for label in data["label_names"].tolist()]
    return TensorDataset(x, y), label_names


def move_to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda(non_blocking=True)
    return tensor


def pool_latent_features(latent_layers):
    pooled = []
    pool = nn.AdaptiveAvgPool1d(1)
    for layer in latent_layers:
        pooled.append(pool(layer).flatten(start_dim=1))
    return pooled


def export_split(net, loader, save_root):
    os.makedirs(save_root, exist_ok=True)
    counters = {}

    net.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = move_to_device(inputs)
            logits, _, latent = net(inputs)

            pooled_latent = pool_latent_features(latent)
            feature = torch.cat([logits] + pooled_latent, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            for index in range(feature.shape[0]):
                class_name = str(int(labels[index]))
                class_dir = os.path.join(save_root, class_name)
                os.makedirs(class_dir, exist_ok=True)
                counters.setdefault(class_name, 0)
                file_name = "{:06d}.npy".format(counters[class_name])
                np.save(os.path.join(class_dir, file_name), feature[index:index + 1], allow_pickle=False)
                counters[class_name] += 1


def main():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    checkpoint = torch.load(args.load_path, map_location="cpu")
    input_channels = checkpoint.get("input_channels", args.input_channels if args.input_channels is not None else 1)
    base_channels = checkpoint.get("base_channels", args.base_channels if args.base_channels is not None else 64)
    hidden_dim = checkpoint.get("hidden_dim", args.hidden_dim if args.hidden_dim is not None else 512)

    train_dataset, train_label_names = load_npz_dataset(args.train_path, input_channels)
    val_dataset, val_label_names = load_npz_dataset(args.val_path, input_channels)
    open_dataset, open_label_names = load_npz_dataset(args.open_path, input_channels)

    checkpoint_label_names = [str(label) for label in checkpoint.get("label_names", np.array(train_label_names)).tolist()]
    if checkpoint_label_names != train_label_names:
        raise ValueError("Checkpoint label names do not match training dataset label names")
    if val_label_names != train_label_names:
        raise ValueError("Validation label names do not match training label names")

    num_classes = len(train_label_names)
    net = DHRNet1D(
        num_classes=num_classes,
        input_channels=input_channels,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
    )
    net.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    if torch.cuda.is_available():
        net.cuda()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)
    open_loader = DataLoader(open_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    export_split(net, train_loader, os.path.join(args.save_path, "train"))
    export_split(net, val_loader, os.path.join(args.save_path, "val"))
    export_split(net, open_loader, os.path.join(args.save_path, "open_set"))

    metadata = {
        "known_label_names": train_label_names,
        "open_label_names": open_label_names,
    }
    np.save(os.path.join(args.save_path, "metadata.npy"), metadata, allow_pickle=True)
    print("Saved 1D features to {}".format(args.save_path))


if __name__ == "__main__":
    main()
