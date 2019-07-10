# coding: utf-8

from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.keras as kr
from cnn_model import TCNNConfig, TextCNN
from data_loader import read_category, read_vocab, FullTokenizer
from run_cnn import base_dir, save_dir

try:
    bool(type(unicode))
except NameError:
    unicode = str

vocab_dir = os.path.join(base_dir, 'vocab.yml')

save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.word_to_id)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls, y_pred_prob = self.session.run([self.model.y_pred_cls, self.model.y_pred_prob], feed_dict=feed_dict)
        print(y_pred_prob)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    tokenizer = FullTokenizer(do_lower_case=True)
    # test_demo = []
    # with codecs.open('data/one_system/category/predict/predict', 'r', 'utf-8') as file:
    #     test_demo = [line.strip() for line in file]
    #
    # results = []
    # for i in test_demo:
    #     result = cnn_model.predict(i)
    #     results.append(result + '\t' + i)
    #
    # with codecs.open('data/one_system/category/predict/results', 'w', 'utf-8') as file:
    #     file.write('\n'.join(results))

    while True:
        text = input('input:')
        print()
        text = tokenizer.tokenize(text)
        result = cnn_model.predict(text)
        print(result)


