#! /usr/bin/env python
import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score, f1_score
from tensorflow.python.platform import gfile

from utils.input_helpers import InputHelper

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "../atec_data/test.txt", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath",
                       "F:\python_work\siamese-lstm-network\deep-siamese-text-similarity\\atec_runs\\1552654885\checkpoints\\vocab_",
                       "Load training time vocabulary (Default: None)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
inpH = InputHelper()
x1_test, x2_test, y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)

session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)

sess = tf.Session()
with tf.gfile.GFile(
        'F:\python_work\siamese-lstm-network\deep-siamese-text-similarity\esim_model.pb',
        'rb') as f:  # 加载模型
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')  # 导入计算图
# 需要有一个初始化的过程
# sess.run(tf.global_variables_initializer())
input_x1 = sess.graph.get_tensor_by_name("input_x1:0")
input_x2 = sess.graph.get_operation_by_name("input_x2").outputs[0]
# input_y = sess.graph.get_operation_by_name("input_y").outputs[0]

dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
# Tensors we want to evaluate
predictions = sess.graph.get_operation_by_name("output/distance").outputs[0]

# accuracy = sess.graph.get_operation_by_name("accuracy/accuracy").outputs[0]

sim = sess.graph.get_operation_by_name("accuracy/temp_sim").outputs[0]
# Evaluation
# ==================================================
batches = inpH.batch_iter(list(zip(x1_test, x2_test, y_test)), 2 * FLAGS.batch_size, 1, shuffle=False)
# Collect the predictions here
all_predictions = []
all_d = []
for db in batches:
    x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
    sess.run(tf.local_variables_initializer())
    batch_predictions, batch_sim = sess.run([predictions, sim],
                                            {input_x1: x1_dev_b, input_x2: x2_dev_b,
                                             dropout_keep_prob: 1.0})
    all_predictions = np.concatenate([all_predictions, batch_predictions])
    all_d = np.concatenate([all_d, batch_sim])
    print(all_d)
    # print("DEV acc {}".format(batch_acc))
    # correct_predictions = float(np.mean(all_d == y_test))
    # # recall = recall_score(y_test, all_d, average='binary')
    # # f1score = f1_score(y_test, all_d, average='binary')
    # print("Accuracy: {:g}".format(correct_predictions))
