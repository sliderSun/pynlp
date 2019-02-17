#!/user/bin/env python
# -*- coding:utf-8 -*-

import argparse

from DataSet import DataSet
from data_process import *
from models import build_model
from predict import predict_module
from train_module import train_module
from utils import *

parser = argparse.ArgumentParser()
# actions to take
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--predict', action='store_true', help='predict the result')

# experiment configuration
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--model_name', type=str, default='TextCNN')
parser.add_argument('--experiment_name', type=str, default='test')

# model hyperparameters
# general
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embedding_size', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--keep_prob', type=float, default=0.5)

parser.add_argument('--decay_rate', type=int, default=0.99)

parser.add_argument('--decay_step', type=int, default=500)
# text rnn
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--rnn_mode', type=str, default='uni')
parser.add_argument('--cell_type', type=str, default='gru')
parser.add_argument('--num_layers', type=int, default=1)
# text cnn
parser.add_argument('--filter_sizes', nargs='+', type=int, default=[3, 4, 5])
# dpcnn
parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--num_filters', type=int, default=250)
parser.add_argument('--strides', type=int, default=2)

##rcnn

parser.add_argument('--cell_size', type=int, default=50)
parser.add_argument('--sequence_length', type=int, default=100)
# preparing settings
# parser.add_argument('--vocab_size', type=int, default=0,
#                     help='how many tokens you want to maintain in your vocab')
parser.add_argument('--file_names', nargs='+', type=str, default=['train.csv', 'dev.csv', 'test.csv'],
                    help='the name of the data files')
parser.add_argument('--data_dir', type=str, default='data', help='directory where the data stored in')
parser.add_argument('--file_prefix', type=str, default='',
                    help='prefix of the file name you want to generate this time')
parser.add_argument('--sampled', action='store_true', help='whether to sample the dataset')
parser.add_argument('--global_config', type=str, default='global_config.json',
                    help='some configurations derived from the preparation')
parser.add_argument('--dev_size', type=float, default=10000, help='if <= 1, this represents the proportion of the dataset to include\
                    in the dev set; if > 1, this represent the absolute number of dev samples')
parser.add_argument('--test_size', type=float, default=10000, help='if <= 1, this represents the proportion of the dataset to include\
                    in the test set; if > 1, this represent the absolute number of test samples')
parser.add_argument('--random_state', type=float, default=20180814,
                    help='the random state used to shuffle the data set')

parser.add_argument('--write_vocab', type=bool, default=False, help='the random state used to shuffle the data set')
# training settings
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train the model on training set')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')

# training settings
parser.add_argument('--restore_from', type=str, default='', help="the checkpoint file or the directory it's stored in")
parser.add_argument('--log_freq', type=int, default=10, help='the frequency of logging')
parser.add_argument('--summary_freq', type=int, default=0, help='the frequency of saving a summary')
parser.add_argument('--save_ckpts', action='store_true', help='whether to store checkpoints')


def train(config):
    # load train data
    print("start load data")
    train_data_df = load_data_from_csv(os.path.join(config.data_dir, config.file_names[0]))
    validate_data_df = load_data_from_csv(os.path.join(config.data_dir, config.file_names[1]))
    # explore data
    print("explore train data!")
    explore_data_analysis(train_data_df)
    print("explore dev data!")
    explore_data_analysis(validate_data_df)

    content_train = train_data_df.iloc[:, 0]

    content_val = validate_data_df.iloc[:, 0]

    if config.write_vocab:
        write_vocab(content_train, os.path.join(config.data_dir, config.file_prefix + 'vocab.data'), min_count=5)

    print("start convert str2id!")
    word2id = load_vocab(os.path.join(config.data_dir, config.file_prefix + 'vocab.data'))
    train_data = list(map(lambda x: string2id(x, word2id), content_train))
    print("train_data的长度", len(train_data))
    val_data = list(map(lambda x: string2id(x, word2id), content_val))

    print("create experiment dir")

    config = prepare_experiment(config, len(word2id), len(train_data_df))

    set_logger(config)
    train_label = train_data_df.iloc[:, 1]
    val_label = validate_data_df.iloc[:, 1]
    train_set = DataSet(config.batch_size, train_data, train_label, config.sequence_length)
    dev_set = DataSet(config.batch_size, val_data, val_label, config.sequence_length)
    print("-----start train  model------")
    model = build_model(config)
    train_module(model, config, train_set, dev_set)
    print("finish train %s model")


def predict(config):
    print("start load data")
    test_data_df = load_data_from_csv(os.path.join(config.data_dir, config.file_names[2]))
    # explore data
    print("explore train data!")
    explore_data_analysis(test_data_df)
    content_test = test_data_df.iloc[:, 0]
    print("start convert str2id!")
    word2id = load_vocab(os.path.join(config.data_dir, config.file_prefix + 'vocab.data'))
    test_data = list(map(lambda x: string2id(x, word2id), content_test))
    print("test_data的长度", len(test_data))
    test_data = list(map(lambda x: string2id(x, word2id), content_test))
    test_set = DataSet(config.batch_size, test_data, test_data, config.sequence_length)
    result = predict_module(config, test_set)

    test_data_df['result'] = result

    write_path = './result.csv'
    test_data_df.to_csv(write_path, index=False, encoding='utf8')


def run():
    args = parser.parse_args()
    config_dict = vars(args)
    config = namedtuple('Config', config_dict.keys())(**config_dict)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if config.train:
        train(config)
    if config.predict:
        predict(config)


if __name__ == "__main__":
    run()
