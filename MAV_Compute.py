
import os, sys
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pickle
import os.path as path
import torch
import numpy as np
import argparse


def compute_mean_vector(category_index,save_path,featurefilepath,):
    
    featurefile_list = os.listdir(os.path.join(featurefilepath,category_index))
    
    correct_features = []
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurefilepath,category_index,featurefile)))

        predicted_category = torch.max(feature,dim=1)[1].item()
        
        if(predicted_category == int(category_index)):
            correct_features.append(feature)

    if not correct_features:
        raise RuntimeError("No correctly classified features found for class {}".format(category_index))

    correct_features = torch.cat(correct_features,0)

    mav = torch.mean(correct_features,dim=0)

    np.save(os.path.join(save_path,category_index+".npy"),mav.data.numpy(),allow_pickle=False)

def get_args():
    parser = argparse.ArgumentParser(description='Get activation vectors')
    parser.add_argument('--save_path',default="./saved_MAVs/cifar10/",type=str,help="Path to save the ensemble weights")
    parser.add_argument('--feature_dir',default="./saved_features/cifar10/train",type=str,help="Path to training-set feature vectors")
    parser.set_defaults(argument=True)

    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.isdir(args.feature_dir):
        raise FileNotFoundError("Feature directory not found: {}".format(args.feature_dir))
    os.makedirs(args.save_path, exist_ok=True)

    for class_no in os.listdir(args.feature_dir):
        print("Class index ",class_no)
        compute_mean_vector(class_no,args.save_path,args.feature_dir)

if __name__ == "__main__":
    main()
