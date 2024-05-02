import cl as cl
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
import h5py
from os import path

def adj_matrix(coords, step_size=256, down_sampling_factor=2):
    total = len(coords)
    adj_s = np.zeros((total, total))

    if down_sampling_factor > 0:
        step_size = step_size * down_sampling_factor * 2

    coords = np.array(coords)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    x_diff = np.abs(x_coords[:, None] - x_coords)
    y_diff = np.abs(y_coords[:, None] - y_coords)

    mask = (x_diff <= step_size) & (y_diff <= step_size)
    adj_s[mask] = 1

    adj_s = torch.from_numpy(adj_s)

    return adj_s


def save_coords(txt_file, coords):
    for coord in coords:
        x, y = coord
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

def load_coords(path_to_coords:str):
    with h5py.File(path_to_coords,'r') as hdf5_file:
        coord = hdf5_file['coords'][:]
    return coord

def compute_feats(joined_list, save_path = None, step_size = 256, down_sampling_factor = 2):
    num_bags = len(joined_list)
    for i, (file_name, path_to_features, path_to_coords) in tqdm(enumerate(joined_list), total=num_bags, desc='Computing adj. matrices'):
        # load coords
        coords = load_coords(path_to_coords)
        # make top dir
        os.makedirs(os.path.join(save_path, 'simclr_files', file_name), exist_ok=True)
        # save coords
        txt_file = open(os.path.join(save_path, 'simclr_files', file_name, 'c_idx.txt'), "w+")
        save_coords(txt_file, coords)
        # make symlink to features
        # ckeck if path_to_features is an absolute path, if not make it absolute
        if not path.isabs(path_to_features):
            path_to_features = os.path.abspath(path_to_features)
        os.symlink(path_to_features, os.path.join(save_path, 'simclr_files', file_name, 'features.pt'))
        # compute adjacent matrix
        adj_s = adj_matrix(coords, step_size=step_size, down_sampling_factor=down_sampling_factor)
        torch.save(adj_s, os.path.join(save_path, 'simclr_files', file_name, 'adj_s.pt'))
        print('\r Computed: {}/{}'.format(i+1, num_bags))
        

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    parser.add_argument('--path_to_features',help='Path where precomputed features are stored. The feature of each file are expected to be stored in one .pt file', type=str, default=None)
    parser.add_argument('--path_to_patches',help='Path to h5 files containing patches', type=str, default=None)
    parser.add_argument('--step_size', help='Step size used during patch extraction', type=int, default=256)
    parser.add_argument('--down_sampling_factor', help='Down sampling factor used during patch extraction', type=int, default=2)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.path_to_features + '/*.pt')
    patches_list = glob.glob(args.path_to_patches + '/*.h5')
    # make a list with all files for computing features and ajd matrix
    joined = []
    for i in bags_list:
        slide_id = i.split('/')[-1][:-3]
        coords_file = [f for f in patches_list if slide_id in f][0]
        joined.append([slide_id,i, coords_file])
       
    compute_feats(joined_list=joined, save_path=args.output, step_size=args.step_size, down_sampling_factor=args.down_sampling_factor)
if __name__ == '__main__':
    main()
