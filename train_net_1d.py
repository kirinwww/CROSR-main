import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from DHR_Net_1D import DHRNet1D

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_args():
    parser = argparse.ArgumentParser(description='Train 1D DHR Net')
    parser.add_argument('--train_path', default="./processed_cicids/train_known.npz", type=str, help="Path to known-class training data")
    parser.add_argument('--val_path', default="./processed_cicids/val_known.npz", type=str, help="Path to known-class validation data")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size")
    parser.add_argument('--momentum', default=0.9, type=float, help="Momentum")
    parser.add_argument('--weight_decay', default=0.0001, type=float, help="Weight decay")
    parser.add_argument('--save_path', default="./save_models/cicids_1d", type=str, help="Directory for checkpoints")
    parser.add_argument('--save_every', default=10, type=int, help="Save a checkpoint every N epochs")
    parser.add_argument('--input_channels', default=1, type=int, help="Number of 1D input channels")
    parser.add_argument('--base_channels', default=128, type=int, help="Base number of convolution channels")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimension in the classifier head")
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'sgd'], help="Optimizer type")
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step', 'plateau', 'none'], help="Learning-rate scheduler")
    parser.add_argument('--min_lr', default=1e-5, type=float, help="Minimum learning rate for cosine annealing")
    return parser.parse_args()


def to_sequence_tensor(array, input_channels):
    x = torch.from_numpy(array).float()
    if input_channels == 1:
        return x.unsqueeze(1)
    if x.size(1) % input_channels != 0:
        raise ValueError("Feature dimension {} is not divisible by input_channels {}".format(x.size(1), input_channels))
    sequence_length = x.size(1) // input_channels
    return x.view(x.size(0), input_channels, sequence_length)


def load_dataset(npz_path, input_channels):
    data = np.load(npz_path, allow_pickle=True)
    x = to_sequence_tensor(data["x"], input_channels)
    y = torch.from_numpy(data["y"]).long()
    label_names = data["label_names"]
    return TensorDataset(x, y), label_names


def run_epoch(epoch_no, net, loader, optimizer=None):
    is_train = optimizer is not None
    net.train(is_train)

    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iterations = 0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    iterator = loader
    if tqdm is not None:
        desc = "Train Epoch {}".format(epoch_no + 1) if is_train else "Val Epoch {}".format(epoch_no + 1)
        iterator = tqdm(loader, desc=desc, leave=False)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels in iterator:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits, reconstruct, _ = net(inputs)
            cls_loss = cls_criterion(logits, labels)
            reconst_loss = reconst_criterion(reconstruct, inputs)
            loss = cls_loss + reconst_loss

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reconst_loss += reconst_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iterations += 1

            if tqdm is not None:
                iterator.set_postfix({
                    "acc": "{:.2f}".format(100.0 * correct / total),
                    "cls": "{:.4f}".format(total_cls_loss / iterations),
                    "rec": "{:.4f}".format(total_reconst_loss / iterations),
                })

    return [
        100.0 * correct / total,
        total_cls_loss / iterations,
        total_reconst_loss / iterations,
        total_loss / iterations,
    ]


def main():
    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    train_dataset, label_names = load_dataset(args.train_path, args.input_channels)
    val_dataset, val_label_names = load_dataset(args.val_path, args.input_channels)
    if len(label_names) != len(val_label_names):
        raise ValueError("Training and validation label sets do not match")

    num_classes = len(label_names)
    print("Num classes {}".format(num_classes))
    print("Input shape {}".format(tuple(train_dataset.tensors[0].shape[1:])))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    net = DHRNet1D(
        num_classes=num_classes,
        input_channels=args.input_channels,
        base_channels=args.base_channels,
        hidden_dim=args.hidden_dim,
    )
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net.cuda())

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        scheduler = None

    best_val_acc = -1.0

    for epoch in range(args.epochs):
        train_metrics = run_epoch(epoch, net, train_loader, optimizer=optimizer)
        val_metrics = run_epoch(epoch, net, val_loader, optimizer=None)

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics[0])
            else:
                scheduler.step()

        print("Train accuracy and cls, reconstruct and total loss for epoch {} is {}".format(epoch, train_metrics))
        print("Test accuracy and cls, reconstruct and total loss for epoch {} is {}".format(epoch, val_metrics))

        model_state = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'train_acc': train_metrics[0],
            'train_loss': train_metrics[3],
            'val_acc': val_metrics[0],
            'val_loss': val_metrics[3],
            'label_names': np.array(label_names),
            'input_channels': args.input_channels,
            'base_channels': args.base_channels,
            'hidden_dim': args.hidden_dim,
            'num_classes': num_classes,
        }

        if val_metrics[0] > best_val_acc:
            best_val_acc = val_metrics[0]
            torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))

        if ((epoch + 1) % args.save_every == 0) or (epoch == args.epochs - 1):
            torch.save(checkpoint, os.path.join(args.save_path, "{:03d}.pth".format(epoch + 1)))
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))


if __name__ == "__main__":
    main()
