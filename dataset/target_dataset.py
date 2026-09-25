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
import random
import os.path as osp

class TargetClassAwareDataset(data.Dataset):
    def __init__(self, root, num_pclass, transform, tgt_class_set, tgt_plabel_dict, name2index, num_steps=None, uk_list=None, binary_label=None):
        self.num_pclass = num_pclass # number of samples per class in each domain
        self.files = []
        self.transform = transform
        labels = []
        self.num_steps=num_steps
        self.name2index = name2index

        self.ind2label = {}
        self.binary_label = binary_label
        self.tgt_files = {i:[] for i in tgt_class_set}
        self.label_set = tgt_class_set
        if uk_list is not None:
            self.uk_pool = list(uk_list.keys())

        for k, v in tgt_plabel_dict.items():
            if v in tgt_class_set:
                self.tgt_files[int(v)].append(k)
#        print({k:len(v) for k,v in self.tgt_files.items()})

    def __getitem__(self, index):
        global_index = None
        if self.num_steps is not None:
            index = index % (len(self.tgt_files))

        label = self.label_set[index]
        tgt_pool = self.tgt_files[label]

        tgt_index = np.random.choice(len(tgt_pool), self.num_pclass)
        tgt_path = [tgt_pool[i] for i in tgt_index]

        real_index = [self.name2index[name] for name in tgt_path]
        real_index = torch.Tensor(real_index).long()

        tgt_labels = [label for i in tgt_index]
        tgt_labels = torch.Tensor(tgt_labels).long()

        tgt_imgs = []
        tgt_bi_labels = []
        for p in tgt_path:
            cur_img = Image.open(p).convert('RGB')
            cur_img = self.transform(cur_img)
            tgt_imgs.append(cur_img)
            if self.binary_label is not None:
                bp_label = torch.Tensor([self.binary_label[p]])
            else:
                bp_label = torch.Tensor([0])
            tgt_bi_labels.append(bp_label)
        tgt_bi_labels = torch.stack(tgt_bi_labels).long()
        tgt_imgs = torch.stack(tgt_imgs, 0)
        return tgt_imgs, tgt_labels, tgt_path, tgt_bi_labels, real_index

    def __len__(self):
        if self.num_steps is None:
            return len(self.tgt_files)
        else:
            return self.num_steps


class TargetUkDataset(data.Dataset):
    def __init__(self, transform, plabel_dict, name2index, uk_class, num_steps=None):
        self.files = []
        self.transform = transform
        self.num_steps = num_steps
        self.name2index = name2index
        self.uk_class = uk_class

        for k, v in plabel_dict.items():
            if v is uk_class:
                self.files.append(k)

        if num_steps is not None:
            self.files = self.files * int(np.ceil(num_steps)/len(self.files) + 1)

        def __getitem__(self, index):
            name = self.files[index]
            img = Image.open(name).convert('RGB')
            img = self.transform(img)
            return img, self.uk_class

        def __len__(self):
            if self.num_steps is None:
                return len(self.files)
            else:
                return self.num_steps

