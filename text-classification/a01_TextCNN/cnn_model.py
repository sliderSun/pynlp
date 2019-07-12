# coding: utf-8

import tensorflow as tf
import numpy as np


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
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


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='rate')

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate

        self.l2_loss = tf.constant(0.0)

        def gelu(input_tensor):
            """Gaussian Error Linear Unit.

            This is a smoother version of the RELU.
            Original paper: https://arxiv.org/abs/1606.08415

            Args:
              input_tensor: float Tensor to perform activation.

            Returns:
              `input_tensor` with the GELU activation applied.
            """
            cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
            return input_tensor * cdf

        def swish(x):
            return x * tf.nn.sigmoid(x)

        self.gelu = gelu
        self.swish = swish
        self.cnn()

    def cbam_module(self, inputs, reduction_ratio=0.5, name=""):
        with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
            batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]

            maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
            avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)

            maxpool_channel = tf.layers.Flatten()(maxpool_channel)
            avgpool_channel = tf.layers.Flatten()(avgpool_channel)

            mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                        reuse=None, activation=tf.nn.relu)
            mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
            mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])

            mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                        reuse=True, activation=tf.nn.relu)
            mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
            mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])

            channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
            channel_refined_feature = inputs * channel_attention

            maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
            avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)
            max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
            conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
                                          activation=None)
            spatial_attention = tf.nn.sigmoid(conv_layer)

            refined_feature = channel_refined_feature * spatial_attention

        return refined_feature

    @staticmethod
    def Global_Average_Pooling(x, stride=1):
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

    @staticmethod
    def Fully_connected(x, units=None, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=True, units=units)

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = self.Global_Average_Pooling(input_x)

            excitation = self.Fully_connected(squeeze, units=out_dim / ratio,
                                              layer_name=layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            scale = input_x * excitation

            return scale

    def attention_encoder(self, X, stddev=0.1):
        """
        attention encoder layer
        """
        M = X.get_shape().as_list()[1]
        N = X.get_shape().as_list()[2]
        reshaped_x = tf.reshape(X, [-1, N, M])
        attention = tf.layers.dense(reshaped_x, M, activation='softmax')
        attention = tf.reshape(attention, [-1, M, N])
        outputs = tf.multiply(X, attention)
        return outputs

    def conv2d_block(self, X, W):
        """
        gated dilation conv1d layer
        """
        glu = tf.sigmoid(tf.nn.conv2d(
            X,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv"))
        conv1 = tf.nn.conv2d(
            X,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        gated_conv = tf.multiply(conv1, glu)

        return gated_conv

    def cnn_block(self, num_filters, h, j, i):
        # W1 = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W1")
        W1 = tf.get_variable(
            "W1_" + str(i) + str(j),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1_" + str(i) + str(j))
        conv1 = tf.nn.conv2d(
            h,
            W1,
            strides=[1, 1, 1, 1],
            padding="SAME")
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
        # W2 = tf.Variable(tf.truncated_normal([3, 1, num_filters, num_filters], stddev=0.1), name="W2")
        W2 = tf.get_variable(
            "W2_" + str(i) + str(j),
            shape=[3, 1, num_filters, num_filters],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2_" + str(i) + str(j))
        conv2 = tf.nn.conv2d(
            h1,
            W2,
            strides=[1, 1, 1, 1],
            padding="SAME")
        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
        self.l2_loss += tf.nn.l2_loss(W2)
        self.l2_loss += tf.nn.l2_loss(b2)
        return h2

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0),
                                    name='embedding')
            self.embedding_inputs = tf.expand_dims(tf.nn.embedding_lookup(embedding, self.input_x), -1)

        with tf.name_scope("cnn"):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = self.conv2d_block(self.embedding_inputs, W)
                    conv = tf.layers.batch_normalization(conv, name='cnn_bn_%s' % filter_size)

                    # Apply nonlinearity
                    h = self.swish(tf.nn.bias_add(conv, b))
                    channel = int(np.shape(h)[-1])
                    h = self.Squeeze_excitation_layer(h, out_dim=channel, ratio=4,
                                                      layer_name='SE_B_%s' % filter_size)
                    # 残差结构
                    # for j in range(4):
                    #     h2 = self.cnn_block(self.config.num_filters, h, j, i)
                    #     h = h2 + h

                    # Maxpooling over the outputs
                    linear_max_pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    gated_max_pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="avg")
                    pooled = linear_max_pooled * tf.sigmoid(gated_max_pooled)
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="predictions")
            self.y_pred_prob = tf.reduce_max(tf.nn.softmax(self.logits), axis=1, name='prediction_prob')

        # Calculate mean cross-entropy loss
        with tf.name_scope("optimize"):
            # learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, self.decay_steps,
            #                                            self.decay_rate, staircase=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
