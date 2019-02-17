# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 15:02
# @Author  : fanghaishuo
# @FileName: __init__.py.py

from models.dpcnn import DPCNN

__all__ = ['DPCNN']


def build_model(config, *args, **kwargs):
    if config.model_name in __all__:
        return globals()[config.model_name](config, *args, **kwargs)
    else:
        raise Exception("小哥哥,DO NOT have this model 鸭")
