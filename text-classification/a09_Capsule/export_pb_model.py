# -*- coding: utf-8 -*-
# @Time    : 2019/3/27 18:09
# @Author  : Magic
# @Email   : hanjunm@haier.com
import os
from shutil import rmtree

import tensorflow as tf
# 保存为pb模型
import yaml

from cnn_model import TCNNConfig, TextCNN
from data_loader import read_category, read_vocab
from run_capsule import base_dir, save_dir

vocab_dir = os.path.join(base_dir, 'vocab.yml')

save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

def export_model_variable_pb(sess, model):
    # 只需要修改这一段，定义输入输出，其他保持默认即可
    inputs = {}
    input_bak = {}
    inputs['input_x'] = tf.saved_model.utils.build_tensor_info(model.input_x)
    input_bak['input_x'] = model.input_x.name
    inputs['keep_prob'] = tf.saved_model.utils.build_tensor_info(model.keep_prob)
    input_bak['keep_prob'] = model.keep_prob.name

    with tf.gfile.GFile(os.path.join(base_dir, 'data_bak', 'inputs.yml'), 'w') as output:
        output.write(yaml.dump(input_bak, allow_unicode=True, indent=4))

    outputs = {}
    output_bak = {}
    outputs['y_pred_cls'] = tf.saved_model.utils.build_tensor_info(model.y_pred_cls)
    output_bak['y_pred_cls'] = model.y_pred_cls.op.name
    outputs['y_pred_prob'] = tf.saved_model.utils.build_tensor_info(model.y_pred_prob)
    output_bak['y_pred_prob'] = model.y_pred_prob.op.name

    with tf.gfile.GFile(os.path.join(base_dir, 'data_bak', 'outputs.yml'), 'w') as output:
        output.write(yaml.dump(output_bak, allow_unicode=True, indent=4))


    model_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    export_path = 'pb/model'
    if os.path.exists(export_path):
        rmtree(export_path)
    tf.logging.info("Export the model to {}".format(export_path))

    # try:
    legacy_init_op = tf.group(
        tf.tables_initializer(), name='legacy_init_op')
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                model_signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()
    # except Exception as e:
    #     print("Fail to export saved model, exception: {}".format(e))


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


def export_model_to_variable_pb():
    cnn_model = CnnModel()

    export_model_variable_pb(cnn_model.session, cnn_model.model)



#加载pb模型
def load_variable_pb():
    session = tf.Session(graph=tf.Graph())
    model_file_path = "pb/model"
    meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_file_path)

    model_graph_signature = list(meta_graph.signature_def.items())[0][1]
    output_feed = []
    output_op_names = []
    output_tensor_dict = {}

    output_op_names.append('y_pred_cls')
    output_op_names.append('y_pred_prob')

    for output_item in model_graph_signature.outputs.items():
        output_op_name = output_item[0]
        output_tensor_name = output_item[1].name
        output_tensor_dict[output_op_name] = output_tensor_name

    for name in output_op_names:
        output_feed.append(output_tensor_dict[name])
        print(output_tensor_dict[name])
    print("load model finish!")

    config = TCNNConfig()
    categories, cat_to_id = read_category()
    word_to_id = read_vocab(vocab_dir)

    while True:

        string = input("请输入测试句子: ").strip()

        input_x = [[word_to_id.get(x, word_to_id['<PAD>']) for x in string]]

        input_x = tf.keras.preprocessing.sequence.pad_sequences(sequences=input_x, maxlen=config.seq_length)

        inputs = {}
        inputs['input_x'] = input_x
        inputs['keep_prob'] = 1.0

        feed_dict = {}
        for input_item in model_graph_signature.inputs.items():
            input_op_name = input_item[0]
            input_tensor_name = input_item[1].name
            feed_dict[input_tensor_name] = inputs[input_op_name]

        outputs = session.run(output_feed, feed_dict=feed_dict)

        print(categories[outputs[0][0]])

        print(outputs[1][0])



if __name__ == '__main__':
    export_model_to_variable_pb()
    load_variable_pb()