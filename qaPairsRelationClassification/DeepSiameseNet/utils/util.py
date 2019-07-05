import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph(output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "output/distance,accuracy/accuracy,accuracy/temp_sim"
    input_checkpoint = "F:\python_work\siamese-lstm-network\deep-siamese-text-similarity\\runs\\1546075513\checkpoints\model-485000.meta"
    model_path = 'F:\python_work\siamese-lstm-network\deep-siamese-text-similarity\\runs\\1546075513\checkpoints\model-485000' # 数据路径

    saver = tf.train.import_meta_graph(input_checkpoint, clear_devices=False)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    with tf.Session() as sess:
        saver.restore(sess, model_path)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


freeze_graph("./model-2019-01-03-V1.pb")
