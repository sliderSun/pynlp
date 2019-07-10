# -*- coding: utf-8 -*-
# @Time    : 2019/5/28 12:03
# @Author  : sliderSun
# @FileName: DataSet.py

import numpy as np
from collections import namedtuple, defaultdict
import math

BatchInputs = namedtuple('BatchInputs', ['texts', 'labels'])


class DataSet(object):
    def __init__(self, batch_size, x, y, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.x = x
        self.y = y

    def padding(self, batch):
        token_ids, label_ids = list(zip(*batch))

        def func(sent):
            _ = sent[:self.sequence_length] + [0] * (self.sequence_length - len(sent))
            return _

        token_ids = list(map(lambda x: func(x), token_ids))
        return BatchInputs(np.array(token_ids), np.array(label_ids))

    def next_batch(self):
        data = list(zip(self.x, self.y))
        num_batches = math.ceil(len(data) / self.batch_size)
        for i in range(num_batches):
            end_index = min((i + 1) * self.batch_size, len(data))
            batch = data[i * self.batch_size:end_index]
            yield self.padding(batch)
