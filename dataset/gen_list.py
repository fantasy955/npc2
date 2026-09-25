# -*- coding: utf-8 -*-
import os
import argparse
import json
import yaml
from init_config import init_config

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,
                    default='/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb52957081/weijl/Domain-Consensus-Clustering-main/config/domainnet.yaml')
parser.add_argument('--dataset', type=str, default='domainnet')
parser.add_argument('--tensorboard', type=bool, default=False)
args = parser.parse_args()

config, _ = init_config(args.config_path, args)

dataset = args.dataset
root = config.root[dataset]

if dataset == 'office-31':
    domains = ['amazon', 'dslr', 'webcam']
elif dataset == 'office-caltech':
    domains = ['amazon', 'dslr', 'webcam', 'caltech']
elif dataset == 'officehome':
    domains = ['Art', 'Clipart', 'Product', 'Real_World']
elif dataset == 'domainnet':
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
else:
    print('No such dataset exists!')

def gen_train_and_val():
    for domain in domains:
        json_set = {
            'train_list': [],
            'val_list': []
        }
        directory = os.path.join(root, dataset, os.path.join(domain, 'images'))
        classes = [x[0] for x in os.walk(directory)]
        classes = classes[1:]
        classes.sort()
        for idx, f in enumerate(classes):
            files = os.listdir(f)

            file_num = len(files)
            train_idx_end = int(file_num * 0.7) - 1
            val_idx_start = train_idx_end

            for idx_1, file in enumerate(files):
                if idx_1 > train_idx_end:
                    break
                s = os.path.abspath(os.path.join(f, file)) + ' ' + str(idx) + '\n'
                json_set['train_list'].append(s)

            for idx_1 in range(file_num):
                if idx_1 + val_idx_start > file_num - 1:
                    break
                s = os.path.abspath(os.path.join(f, files[idx_1 + val_idx_start])) + ' ' + str(idx) + '\n'
                json_set['val_list'].append(s)

        with open(os.path.join(root, dataset, domain, 'list.json'), 'w') as f:
            json.dump(json_set, f, indent=4)

def gen_list():
    for domain in domains:
        directory = os.path.join(root, os.path.join(domain, 'images'))
        if dataset in ['officehome', 'domainnet']:
            directory = os.path.join(root, os.path.join(domain))
        classes = [x[0] for x in os.walk(directory)]
        classes = classes[1:]
        classes.sort()

        with open(os.path.join('/media/ubuntu/7d17c4ae-0255-4946-a82e-1ebcb52957081/weijl/Domain-Consensus-Clustering-main/dataset/list',
                               dataset, '{}.txt'.format(domain)), 'w') as txt:
            for idx, f in enumerate(classes):
                files = os.listdir(f)
                for idx_1, file in enumerate(files):
                    txt.write(os.path.abspath(os.path.join(f, file)) + ' ' + str(idx) + '\n')
            txt.close()


if __name__ == '__main__':
    gen_list()



