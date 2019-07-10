# -*- coding: utf-8 -*-
# @Time    : 2019/3/28 15:28
# @Author  : sliderSun
# @FileName: utils.py
import logging
import os
from collections import namedtuple


def set_logger(config):
    logger = logging.getLogger(config.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.exp_dir, 'log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def prepare_experiment(config, vocab_size, train_size):
    config_dict = config._asdict()
    for k, v in config_dict.items():
        if k.endswith('dir') and not os.path.exists(v):
            os.mkdir(v)

    model_dir = os.path.join(config.experiment_dir, config.model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    exp_dir = os.path.join(model_dir, config.experiment_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    sub_dir = ['checkpoints', 'results']
    for dir_ in sub_dir:
        if not os.path.exists(os.path.join(exp_dir, dir_)):
            os.mkdir(os.path.join(exp_dir, dir_))

    dir_dict = {'model_dir': model_dir, 'exp_dir': exp_dir, 'ckpt_dir': os.path.join(exp_dir, 'checkpoints'),
                'result_dir': os.path.join(exp_dir, 'results'), 'vocab_size': vocab_size, 'train_size': train_size}
    config_dict.update(dir_dict)
    config = namedtuple('Config', config_dict.keys())(**config_dict)
    return config
