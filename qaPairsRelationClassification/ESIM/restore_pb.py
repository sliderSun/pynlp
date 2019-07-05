"""
Created on @Time:2019/2/21 11:21
@Author:sliderSun 
@FileName: restore_pb.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datetime import datetime

from sklearn.metrics import f1_score

import Config
from utils import *

sess = tf.Session()
with tf.gfile.GFile("F:\python_work\siamese-lstm-network\ESIM\esim_model.pb", 'rb') as f:  # 加载模型
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图
# 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())
premise = sess.graph.get_tensor_by_name("premise:0")
hypothesis = sess.graph.get_operation_by_name("hypothesis").outputs[0]
dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
# Tensors we want to evaluate
logit = sess.graph.get_operation_by_name("composition/feed_forward/feed_foward_layer2/dense/Tanh").outputs[0]

print("load complete")
config = Config.ModelConfig()
arg = config.arg

vocab_dict = load_vocab(arg.vocab_path)
arg.vocab_dict_size = len(vocab_dict)
index2word = {index: word for word, index in vocab_dict.items()}
#
# if arg.embedding_path:
#     embeddings = load_embeddings(arg.embedding_path, vocab_dict)
# else:
#     embeddings = init_embeddings(vocab_dict, arg.embedding_size)
# arg.n_vocab, arg.embedding_size = embeddings.shape
#
# if arg.embedding_normalize:
#     embeddings = normalize_embeddings(embeddings)

arg.n_classes = len(CATEGORIE_ID)

dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
arg.log_path = 'config/log.{}'.format(dt)
log = open(arg.log_path, 'w')
print_log('CMD : python3 {0}'.format(' '.join(sys.argv)), file=log)
print_log('Testing with following options :', file=log)
print_args(arg, log)


def predict(logit):
    # load data
    print_log('Loading testing data ...', file=log)
    start_time = time.time()
    premise_test, premise_mask_test, hypothesis_test, hypothesis_mask_test, y_test = sentence2Index(
        arg.testset_path, vocab_dict)
    batches = next_batch(premise_test, premise_mask_test, hypothesis_test, hypothesis_mask_test, y_test, shuffle=False)
    time_diff = get_time_diff(start_time)
    print_log('Time usage : ', time_diff, file=log)

    # testing
    print_log('Start testing ...', file=log)
    start_time = time.time()
    y_pred = []
    for batch in batches:
        batch_premise_test, batch_premise_mask_test, batch_hypothesis_test, batch_hypothesis_mask_test, _ = batch
        feed_dict = {premise: batch_premise_test,
                     # model.premise_mask: batch_premise_mask_test,
                     hypothesis: batch_hypothesis_test,
                     # model.hypothesis_mask: batch_hypothesis_mask_test,
                     dropout_keep_prob: 1.0}
        logits = sess.run([logit], feed_dict=feed_dict)
        logits = np.array(logits)
        logits = logits.reshape([-1, logits.shape[-1]])
        y_pred.extend(logits)
    # evaluating
    y_pred = np.argmax(y_pred, 1)
    y_true = np.argmax(y_test, 1)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_true))
    acc = np.mean(y_true == y_pred)
    for id in range(len(y_true)):
        if y_true[id] != y_pred[id]:
            premise_text = ''.join(
                [index2word[idx] + ' ' for idx in premise_test[id] if index2word[idx] != '<PAD>'])
            hypothesis_text = ''.join(
                [index2word[idx] + ' ' for idx in hypothesis_test[id] if index2word[idx] != '<PAD>'])
            print('Left_text: {0}/ Right_text: {1}'.format(premise_text, hypothesis_text))
            print('The true label is {0}/ The pred label is {1}'.format(y_true[id], y_pred[id]))
    print('The test accuracy: {0:>6.2%}'.format(acc))
    print('The test F1: {0:>6.4}'.format(f1))
    time_diff = get_time_diff(start_time)
    print('Time usage: ', time_diff, '\n')


predict(logit)
log.close()
