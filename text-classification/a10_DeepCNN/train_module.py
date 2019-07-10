from sklearn.metrics import precision_recall_fscore_support
from tqdm import trange
import tensorflow as tf
import os
import json


def train_epoch(model, sess, epoch, train_set, config):
    step, loss, acc, total_len = 0, 0.0, 0.0, 0.0
    t = trange(config.train_size // config.batch_size + 1, desc='epoch {}'.format(epoch), ascii=True)
    for batch in train_set.next_batch():
        _, cur_loss, cur_acc, global_step = sess.run([model.train_op, model.loss, model.accuracy, model.global_step],
                                                     feed_dict={model.dropout_keep_prob: config.keep_prob,
                                                                model.input_x: batch.texts, model.input_y: batch.labels})

        step += 1
        batch_len = len(batch.texts)
        loss += cur_loss * batch_len
        acc += cur_acc * batch_len
        total_len += batch_len

        t.update(1)
        # t.set_postfix_str('loss: {:.4f}, acc: {:.4f}'.format(loss/total_len, acc/total_len))
        if step % model.config.log_freq == 0:
            model.logger.info('Train loss {:.4f}, accuracy {:.4%}'.format(loss / total_len, acc / total_len))

        # if self.config.summary_freq > 0 and step % self.config.summary_freq == 0:  # writer.add_summary(summaries, global_step)


def eval_epoch(model, sess, eval_set):
    result, labels = [], []
    avg_loss, avg_acc, steps, total_len = 0, 0, 0, 0
    for batch in eval_set.next_batch():
        steps += 1
        predictions, batch_loss, batch_acc = sess.run([model.pred, model.loss, model.accuracy],
                                                      feed_dict={model.dropout_keep_prob: 1.0,
                                                                 model.input_x: batch.texts, model.input_y: batch.labels})
        batch_len = len(batch.texts)
        avg_loss += batch_loss * batch_len
        avg_acc += batch_acc * batch_len
        total_len += batch_len

        result.extend(predictions.tolist())
        labels.extend(batch.labels.tolist())

    avg_loss, avg_acc = avg_loss / total_len, avg_acc / total_len
    precision, recall, fscore, support = precision_recall_fscore_support(labels, result, average='weighted')
    metrics = {'loss': avg_loss, 'accuracy': avg_acc, 'precision': precision, 'recall': recall, 'fscore': fscore }

    return metrics, result


def train_module(model, config, train_set, dev_set):
    saver = tf.train.Saver()
    best_dev_acc = 0
    with tf.Session() as sess:
        print('Initializer Variables!')
        sess.run(tf.global_variables_initializer())
        model.logger.info("Start training!")
        for epoch in range(1, config.num_epochs + 1):
            model.logger.info('Epoch {}/{}'.format(epoch, config.num_epochs))
            train_epoch(model, sess, epoch, train_set, config)
            train_metrics, _ = eval_epoch(model, sess, train_set)
            model.logger.info('train_metrics:{}'.format(train_metrics))
            dev_metrics, _ = eval_epoch(model,sess, dev_set)
            model.logger.info('dev_metrics:{}'.format(dev_metrics))

            if dev_metrics['accuracy'] > best_dev_acc:
                best_dev_acc = dev_metrics['accuracy']

                save_metrics(dev_metrics, 'best_dev_metrics', config)
                model.logger.info('Found new model !!!')
                model_name = os.path.join(os.path.join(config.exp_dir, 'checkpoints'), 'best_weights', 'after_epoch')
                saver.save(sess, model_name, epoch)



def save_metrics(metrics, file_name, config):
    with open(os.path.join(config.exp_dir, file_name) + '.json', 'w', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False)
