import tensorflow as tf
import logging


class DPCNN(object):
    def __init__(self, config, initializer=tf.contrib.layers.xavier_initializer()):
        self.config = config
        self.initializer = initializer
        self.logger = logging.getLogger(self.config.experiment_name)
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.inistiante_weight()
        self.logits = self.inference()
        self.loss = self.define_loss()
        self.train_op = self.train()
        self.pred = tf.cast(tf.argmax(self.logits, axis=1), tf.int32, name='predictions')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.pred), tf.float32), name='accuracy')

    def inistiante_weight(self):
        with tf.name_scope('weights'):
            self.Embedding = tf.get_variable('embedding', [self.config.vocab_size + 2, self.config.embedding_size],
                                             initializer=self.initializer)
            self.region_w = tf.get_variable("W_region", [self.config.kernel_size, self.config.embedding_size, 1,
                                                         self.config.num_filters], initializer=self.initializer,
                                            dtype=tf.float32)
            self.w_projection = tf.get_variable("W_projection", [self.config.num_filters, self.config.num_classes],
                                                initializer=self.initializer, dtype=tf.float32)
            self.b_projection = tf.get_variable('b_projection', [self.config.num_classes], initializer=self.initializer,
                                                dtype=tf.float32)

    def conv3(self, k, input_):
        conv3_w = tf.get_variable("W_conv%s" % k,
                                  [self.config.kernel_size, 1, self.config.num_filters, self.config.num_filters],
                                  initializer=self.initializer, dtype=tf.float32)
        conv = tf.nn.conv2d(input_, conv3_w, strides=[1, 1, 1, 1], padding='SAME')
        return conv

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x, name='look_up')
        self.embedded_words = tf.expand_dims(self.embedded_words, axis=-1)  # [None,seq,embedding,1]

        regoin_embedding = tf.nn.conv2d(self.embedded_words, self.region_w, strides=[1, 1, 1, 1],
                                        padding='VALID')  # [batch,seq-3+1,1,250]

        pre_activation = tf.nn.relu(regoin_embedding, name='preactivation')

        conv3 = self.conv3(0, pre_activation)  # [batch,seq-3+1,1,250]
        # batch norm
        conv3 = tf.layers.batch_normalization(conv3)

        conv3_pre_activation = tf.nn.relu(conv3, name='preactivation')
        conv3 = self.conv3(1, conv3_pre_activation)  # [batch,seq-3+1,1,250]
        # batch norm
        conv3 = tf.layers.batch_normalization(conv3)

        conv3 = conv3 + regoin_embedding  # [batch,seq-3+1,1,250]
        k = 1
        # print('conv3',conv3.get_shape().as_list())
        while conv3.get_shape().as_list()[1] >= 2:
            conv3, k = self._block(conv3, k)

        conv3 = tf.squeeze(conv3, [1, 2])  # [batch,250]
        print('conv3 ==>', conv3)
        conv3 = tf.nn.dropout(conv3, self.dropout_keep_prob)

        with tf.name_scope('output'):
            logits = tf.matmul(conv3, self.w_projection) + self.b_projection
            self.scores = tf.nn.softmax(logits, name='scores')
        return logits

    def define_loss(self):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        loss = tf.reduce_mean(losses)
        return loss

    def train(self):
        with tf.name_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # learning_rate = tf.train.exponential_decay(self.config.learning_rate,self.global_step,self.config.decay_step,self.config.decay_rate)
                train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                           learning_rate=self.config.learning_rate, optimizer='Adam')
                return train_op

    def _block(self, x, k):
        x = tf.pad(x, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])

        px = tf.nn.max_pool(x, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
        # conv
        k += 1
        x = tf.nn.relu(px)
        x = self.conv3(k, x)
        x = tf.layers.batch_normalization(x)

        # conv
        k += 1
        x = tf.nn.relu(x)
        x = self.conv3(k, x)
        x = tf.layers.batch_normalization(x)
        x = x + px
        return x, k
