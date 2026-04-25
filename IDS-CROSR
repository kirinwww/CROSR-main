import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DHR_Net as models
import numpy as np
import pickle
import os
from PIL import Image
import argparse


def move_to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda(non_blocking=True)
    return tensor


def pool_latent_features(latent_layers):
    pooled = []
    pool = nn.AdaptiveAvgPool2d((1,1))
    for layer in latent_layers:
        pooled_layer = pool(layer).flatten(start_dim=1)
        pooled.append(pooled_layer)
    return pooled

def epoch_anomalous(net,save_path,root,transform_test):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"open_set")):
            count=0
            
            for file_name in os.listdir(os.path.join(root,"open_set",str(folder))):

                if(count>=120):
                    break
                count = count + 1

                
                image = Image.open(os.path.join(root,"open_set",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image = move_to_device(image)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                squeezed_latent.extend(pool_latent_features(latent))
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"open_set",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                


def epoch_train(net,save_path,root,transform_test):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"train")):

            
            for file_name in os.listdir(os.path.join(root,"train",str(folder))):

                image = Image.open(os.path.join(root,"train",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image = move_to_device(image)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                squeezed_latent.extend(pool_latent_features(latent))
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"train",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                
                
           

def epoch_val(net,save_path,root,transform_test):

    net.eval()

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,"val")):

           
            for file_name in os.listdir(os.path.join(root,"val",str(folder))):

                image = Image.open(os.path.join(root,"val",str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image = move_to_device(image)
                logits, _, latent = net(image)

                squeezed_latent = []
                squeezed_latent.append(logits)
                squeezed_latent.extend(pool_latent_features(latent))
                
                feature = torch.cat(squeezed_latent,1)

                
                save_name = file_name.split(".")[0]
                print(save_name)
                np.save(os.path.join(save_path,"val",str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                 
def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--dataset_dir',default="./data/cifar10",type=str,help="Number of members in ensemble")
    parser.add_argument('--num_classes',default=6,type=int,help="Number of classes in dataset")
    parser.add_argument('--means',nargs='+',default=[0.4914, 0.4822, 0.4465],type=float,help="channelwise means for normalization")
    parser.add_argument('--stds',nargs='+',default=[0.2023, 0.1994, 0.2010],type=float,help="channelwise std for normalization")
    parser.add_argument('--save_path',default="./saved_features/cifar10",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--load_path',default="./save_models/cifar10/latest.pth",type=str,help="Path to the trained checkpoint")
    parser.set_defaults(argument=True)

    return parser.parse_args()

def main():

    args = get_args()

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    num_classes = args.num_classes
    print("Num classes "+str(num_classes))


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.means, args.stds),
    ])

    root = args.dataset_dir
    if not os.path.isdir(os.path.join(root, "train")):
        raise FileNotFoundError("Training images not found under {}".format(os.path.join(root, "train")))
    if not os.path.isdir(os.path.join(root, "val")):
        raise FileNotFoundError("Validation images not found under {}".format(os.path.join(root, "val")))
    os.makedirs(args.save_path, exist_ok=True)
    for split in ["train", "val", "open_set"]:
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            continue
        for class_name in os.listdir(split_root):
            os.makedirs(os.path.join(args.save_path, split, class_name), exist_ok=True)

    net = models.DHRNet(num_classes)
    checkpoint = torch.load(args.load_path,map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    if torch.cuda.is_available():
        net.cuda()
 
    epoch_train(net,args.save_path,root,transform_test)
    epoch_val(net,args.save_path,root,transform_test)
    if os.path.isdir(os.path.join(root,"open_set")):
        epoch_anomalous(net,args.save_path,root,transform_test)


if __name__=="__main__":
    main()
    
