import datetime
import os
import time
from random import random

import tensorflow as tf

from qaPairsRelationClassification.DeepSiameseNet.siamese_network import SiameseNet
from qaPairsRelationClassification.DeepSiameseNet.utils.input_helpers import InputHelper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 64)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "./atec_data/train.txt", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 50, "Number of hidden units (default:50)")
tf.flags.DEFINE_integer("max_document_length", 50, "max length of sentence (default:50)")
tf.flags.DEFINE_integer("percent_dev", 10, "percent_dev (default:10)")
tf.flags.DEFINE_integer("n_layers", 3, "rnn layers (default:3)")
tf.flags.DEFINE_string("initializer", "xavier", "initializer (default:xavier)")
tf.flags.DEFINE_string("cell", "gru", "cell type (default:lstm)")
tf.flags.DEFINE_integer("num_blocks", 6, " number of encoder/decoder blocks (default:6)")
tf.flags.DEFINE_integer("num_heads", 8, " num_heads (default:8)")
tf.flags.DEFINE_integer("num_units", 64, " alias = C (default:64)")
tf.flags.DEFINE_integer("attention_size", 128, "attention_size")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
if FLAGS.training_files is None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

inpH = InputHelper()
train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets(FLAGS.training_files,
                                                                          FLAGS.max_document_length, FLAGS.percent_dev,
                                                                          FLAGS.batch_size)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        siameseModel = SiameseNet(
            config=FLAGS,
            vocab_size=len(vocab_processor.vocabulary_)
        )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = AdamWeightDecayOptimizer(learning_rate=FLAGS.learning_rate, weight_decay_rate=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        print("initialized siameseModel object")

    grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
    capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var)
                  for grad, var in grads_and_vars]
    tr_op_set = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "atec_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab_"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random() > 0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist, sim, summaries = sess.run(
            [tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy,
             siameseModel.distance,
             siameseModel.temp_sim, train_summary_op],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)


    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random() > 0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim, summaries = sess.run(
            [global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim, dev_summary_op], feed_dict)
        time_str = datetime.datetime.now().isoformat()

        print(
            "DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        dev_summary_writer.add_summary(summaries, step)
        return accuracy


    # Generate batches
    batches = inpH.batch_iter(
        list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr = 0
    max_validation_acc = 0.0
    count = 0
    for nn in range(sum_no_of_batches * FLAGS.num_epochs):
        batch = batches.__next__()
        if len(batch) < 1:
            continue
        x1_batch, x2_batch, y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)

                sum_acc = sum_acc + acc

        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                count += 1
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                      checkpoint_prefix))
        if count > 100:
            break
