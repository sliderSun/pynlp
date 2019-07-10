"""
Created on @Time:2019/6/21 19:14
@Author:sliderSun 
@FileName: capsule_model.py
"""

# coding: utf-8

import tensorflow as tf
import keras.backend as K
from capsule.layer import capsules_init, capsule_conv_layer, capsule_flatten, capsule_fc_layer
from capsule.loss import spread_loss, margin_loss, cross_entropy
from capsule.utils import _conv2d_wrapper


class CapsuleConfig(object):
    """CNN配置参数"""
    margin = 0.2
    embedding_dim = 300  # 词向量维度
    seq_length = 30  # 序列长度
    num_classes = 8  # 类别数
    num_filters = 128  # 卷积核数目
    vocab_size = 5000  # 词汇表达小
    l2_reg_lambda = 0.0
    filter_sizes = [2, 3, 4, 5]
    kernel_size = 3
    hidden_dim = 128  # 全连接层神经元
    decay_steps = 1000
    decay_rate = 0.64
    dropout_keep_prob = 0.8  # dropout保留比例
    # learning_rate = 1e-3  # 学习率
    learning_rate = 0.001  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    use_leaky = False
    model_type = "capsule-B"
    loss_type = "cross_entropy"

class Capsule(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='rate')
        self.margin = tf.placeholder(tf.float32, name='margin')

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate

        self.l2_loss = tf.constant(0.0)
        self.capsule()

    def capsule(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0),
                                    name='embedding')
            embedding = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding = embedding[..., tf.newaxis]

        with tf.name_scope("output"):
            if self.config.model_type == 'capsule-A':
                poses, activations = self.capsule_model_A(embedding, self.config.num_classes)
            if self.config.model_type == 'capsule-B':
                poses, activations = self.capsule_model_B(embedding, self.config.num_classes)

        with tf.name_scope("optimize"):
            if self.config.loss_type == 'spread_loss':
                self.loss = spread_loss(self.input_y, activations, self.margin)
            if self.config.loss_type == 'margin_loss':
                self.loss = margin_loss(self.input_y, activations)
            if self.config.loss_type == 'cross_entropy':
                self.loss = cross_entropy(self.input_y, activations)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            self.y_pred_cls = tf.argmax(activations, axis=1, name="y_proba")
            correct = tf.equal(tf.argmax(self.input_y, axis=1), self.y_pred_cls, name="correct")

            self.acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    def capsule_model_A(self, X, num_classes):
        with tf.variable_scope('capsule_' + str(3)):
            nets = _conv2d_wrapper(
                X, shape=[3, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID',
                add_bias=True, activation_fn=tf.nn.relu, name='conv1'
            )
            tf.logging.info('output shape: {}'.format(nets.get_shape()))
            nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                                 padding='VALID', pose_shape=16, add_bias=True, name='primary')
            nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
            nets = capsule_flatten(nets)
            poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
        return poses, activations

    def capsule_model_B(self, X, num_classes):
        poses_list = []
        for _, ngram in enumerate([3, 4, 5]):
            with tf.variable_scope('capsule_' + str(ngram)):
                nets = _conv2d_wrapper(
                    X, shape=[ngram, 300, 1, 32], strides=[1, 2, 1, 1], padding='VALID',
                    add_bias=True, activation_fn=tf.nn.relu, name='conv1'
                )
                tf.logging.info('output shape: {}'.format(nets.get_shape()))
                nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                                     padding='VALID', pose_shape=16, add_bias=True, name='primary')
                nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
                nets = capsule_flatten(nets)
                poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
                poses_list.append(poses)

        poses = tf.reduce_mean(tf.convert_to_tensor(poses_list), axis=0)
        activations = K.sqrt(K.sum(K.square(poses), 2))
        return poses, activations
