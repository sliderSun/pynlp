import tensorflow as tf
import os


def predict_module(config, test_set):
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            os.path.join(config.experiment_dir, config.model_name, config.experiment_name, 'checkpoints',
                         'best_weights'))
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        pred = graph.get_tensor_by_name("predictions:0")
        input_x = graph.get_tensor_by_name("input_x:0")
        text_lens = graph.get_tensor_by_name("text_lens:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        result = []
        for batch in test_set.next_batch():
            predictions = sess.run(pred,
                                   feed_dict={input_x: batch.texts, text_lens: batch.text_lens, dropout_keep_prob: 1.0})
            result.extend(predictions.tolist())
    return result
