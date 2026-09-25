import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
import time 
class PathDataset(data.Dataset):
    def __init__(self, path_dict, transform, num_steps=None):
        
        self.files = []
        self.transform = transform
        labels = []

        for name, label in path_dict.items():
            label = int(label)

            labels.append(label)
            self.files.append([name, int(label)])

        if num_steps is not None:
            self.files = self.files * int(np.ceil(num_steps)/len(self.files) + 1)

    def __getitem__(self, index):
        files = self.files[index]
        name = files[0]
        label = files[1]
        img = Image.open(files[0]).convert('RGB')
        img = self.transform(img)

        return img, label, files[0], 0.0

    def __len__(self):
        return len(self.files)
