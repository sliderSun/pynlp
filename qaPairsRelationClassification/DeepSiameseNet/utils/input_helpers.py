import gc
import warnings
from random import random

import numpy as np

from utils.preprocess import MyVocabularyProcessor


class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None

    def getVocab(self, vocab_path, max_document_length, filter_h_pad):
        if self.vocab_processor is None:
            print('locading vocab_')
            vocab_processor = MyVocabularyProcessor(max_document_length - filter_h_pad, min_frequency=0)
            self.vocab_processor = vocab_processor.restore(vocab_path)
        return self.vocab_processor

    def deletePreEmb(self):
        self.pre_emb = dict()
        gc.collect()

    def getTsvData(self, filepath):
        print("Loading training data from " + filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in open(filepath):
            l = line.strip().split("\t")
            if len(l) < 2:
                continue
            if random() > 0.5:
                x1.append(l[0].lower())
                x2.append(l[1].lower())
            else:
                x1.append(l[1].lower())
                x2.append(l[0].lower())
            y.append(int(l[2]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsvDataCharBased(self, filepath):
        print("Loading training data from " + filepath)
        x1 = []
        x2 = []
        y = []
        x1_negative = []
        x2_negative = []
        y_negative = []
        # positive samples from file
        for line in open(filepath, encoding="utf-8"):
            l = line.strip().split("\t")
            if len(l) != 3:
                print(l)
                continue
            if l[2] == "0":
                x1_negative.append(l[0].lower())
                x2_negative.append(l[1].lower())
                y_negative.append(l[2])
            if random() > 0.5:
                x1.append(l[0].lower())
                x2.append(l[1].lower())
                y.append(l[2])  # np.array([0,1]))
            else:
                x1.append(l[1].lower())
                x2.append(l[0].lower())
                y.append(l[2])  # np.array([0,1]))
        # generate random negative samples
        # combined = np.asarray(x1 + x2)
        # shuffle_indices = np.random.permutation(np.arange(len(combined)))
        # combined_shuff = combined[shuffle_indices]
        # for i in range(len(combined)):
        #     x1.append(combined[i])
        #     x2.append(combined_shuff[i])
        #     y.append(0)  # np.array([1,0]))
        # for i in range(len(x1_negative)):
        #     x1.append(x1_negative[i])
        #     x2.append(x2_negative[i])
        #     y.append(y_negative[i])
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from " + filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in open(filepath, encoding="utf-8"):  # , encoding="utf-8"
            l = line.strip().split("\t")
            if len(l) < 3:
                continue
            x1.append(l[1].lower())
            x2.append(l[2].lower())
            y.append(int(l[0]))  # np.array([0,1]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def dumpValidation(self, x1_text, x2_text, y, shuffled_index, dev_idx, i):
        print("dumping dev " + str(i))
        x1_shuffled = x1_text[shuffled_index]
        x2_shuffled = x2_text[shuffled_index]
        y_shuffled = y[shuffled_index]
        x1_dev = x1_shuffled[dev_idx:]
        x2_dev = x2_shuffled[dev_idx:]
        y_dev = y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('F:\python_work\siamese-lstm-network\deep-siamese-text-similarity\\atec_data\dev.txt', 'w', encoding="utf-8") as f:
            for text1, text2, label in zip(x1_dev, x2_dev, y_dev):
                f.write(str(label) + "\t" + text1 + "\t" + text2 + "\n")
            f.close()
        del x1_dev
        del y_dev

    # Data Preparatopn
    # ==================================================

    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size):
        x1_text, x2_text, y = self.getTsvDataCharBased(training_paths)
        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        f = open("./voc", "w",encoding="utf-8")
        for i in range(len(vocab_processor.vocabulary_)):
            f.write(vocab_processor.vocabulary_.reverse(i)+"\n")
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches

    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp, x2_temp, y = self.getTsvTestData(data_path)
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        # f = open("./vocab_new", "w", encoding="utf-8")
        # for i in range(len(vocab_processor.vocabulary_)):
        #     f.write(vocab_processor.vocabulary_.reverse(i) + "\n")

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2, y
