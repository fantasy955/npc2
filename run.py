import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from init_config import *
import argparse
import warnings
from trainer.trainer import Trainer
import os

warnings.filterwarnings("ignore")

domain_list = {}
domain_list['officehome'] = ['Art', 'Product', 'Clipart', 'Real_World']

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,
                    default='config/oh.yaml'
                    )
args = parser.parse_args()


def main():
    config = init_config(args.config_path, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    message = show_config(config)

    if config.tensorboard:
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

    if config.setting == 'uda':
        config.cls_share = 10
        config.cls_src = 5
        config.cls_total = 65
    elif config.setting == 'osda':
        config.cls_share = 25
        config.cls_src = 0
        config.cls_total = 65
    elif config.setting == 'pda':
        config.cls_share = 25
        config.cls_src = 40
        config.cls_total = 65

    config.num_classes = config.cls_share + config.cls_src
    config.uk_index = config.cls_share + config.cls_src
    a, b, c = config.cls_share, config.cls_src, config.cls_total
    c = c - a - b
    share_classes = [i for i in range(a)]
    source_classes = [a + i for i in range(b)]
    target_classes = [a + b + i for i in range(c)]
    config.share_classes = share_classes
    config.source_classes = share_classes + source_classes
    config.target_classes = share_classes + target_classes
    transfer_list = []
    domains = domain_list[config.task]
    for src in domains:
        for tgt in domains:
            if src != tgt:
                transfer_list.append((src, tgt))
    if not config.transfer_all:
        config.task_name = '{}2{}'.format(config.source[0], config.target[0])
        trainer = Trainer(config)
        trainer.train()
    else:
        transfer_list = [
            ('Art', 'Product'),
            ('Art', 'Clipart'),
            ('Art', 'Real_World'),
            ('Product', 'Art'),
            ('Product', 'Clipart'),
            ('Product', 'Real_World'),
            ('Clipart', 'Art'),
            ('Clipart', 'Product'),
            ('Clipart', 'Real_World'),
            ('Real_World', 'Art'),
            ('Real_World', 'Product'),
            ('Real_World', 'Clipart')
        ]
        print(transfer_list)
        for i, (src, tgt) in enumerate(transfer_list):
            print('{}-->{}'.format(src, tgt))
            config.task_name = '{}2{}'.format(src[0], tgt[0])
            config.source = src
            config.target = tgt
            trainer = Trainer(config)
            trainer.train()
            config.save_model = False
            config.load_pretrain = True


if __name__ == "__main__":
    main()