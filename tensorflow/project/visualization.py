import json
import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_config_yaml
from dataset import read_img, get_train_val_dataloader


def class_distribution(data):
    masks = data["masks"]
    patch_idx = data["patch_idx"]
    pixels = {"Water":0, "NON-Water":0}
    for i in range(len(masks)):
        mask = read_img(masks[i], label=True, patch_idx=patch_idx[i])
        pixels["Water"] += np.sum(mask)
        pixels["NON-Water"] += (mask.shape[0]*mask.shape[1]) - np.sum(mask)
    tot_pix = pixels["Water"] + pixels["NON-Water"]
    pixels["Water"] = pixels["Water"] / tot_pix
    pixels["NON-Water"] = pixels["NON-Water"] / tot_pix
    return pixels

def plot_mask(path, config):
    mask = np.array(Image.open(path))
    plt.imshow(mask)
    name = path.split("/")[-1] + ".png"
    plt.savefig(os.path.join(config['visualization_dir'], name), bbox_inches='tight')



if __name__=='__main__':

    config = get_config_yaml('config.yaml', {})
    
    pathlib.Path(config['visualization_dir']).mkdir(parents = True, exist_ok = True)
    path = ["D:/MsCourse/AI/project/flood_detection/pytorch/data/train/train-label-img/6715_lab.png",
            "D:/MsCourse/AI/project/flood_detection/pytorch/data/train/train-label-img/7110_lab.png",
            "D:/MsCourse/AI/project/flood_detection/pytorch/data/train/train-label-img/7256_lab.png"]
    for i in path:
        plot_mask(i, config)

    # with open(config['p_train_dir'], 'r') as j:
    #     train_dir = json.loads(j.read())
    # print("Train examples: ", len(train_dir["masks"]))
    # print(class_distribution(train_dir))

    # with open(config['p_test_dir'], 'r') as j:
    #     test_dir = json.loads(j.read())
    # print("Test examples: ", len(test_dir["masks"]))
    # print(class_distribution(test_dir))

    # with open(config['p_valid_dir'], 'r') as j:
    #     valid_dir = json.loads(j.read())
    # print("Valid examples: ", len(valid_dir["masks"]))
    # print(class_distribution(valid_dir))
