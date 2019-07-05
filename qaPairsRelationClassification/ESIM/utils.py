'''
Created on July 20, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import pickle
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr

from pyhanlp import HanLP

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
CATEGORIE_ID = {'0': 0, '1': 1}


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    print('{0} : {1}'.format(varname, var.get_shape()))


# init embeddings randomly
def init_embeddings(vocab, embedding_dims):
    """
    :param vocab: word nums of the vocabulary
    :param embedding_dims: dimension of embedding vector
    :return: randomly init embeddings with shape (vocab, embedding_dims)
    """
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(size=(len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)


# load pre-trained embeddings
def load_embeddings(path, vocab):
    """
    :param path: path of the pre-trained embeddings file
    :param vocab: word nums of the vocabulary
    :return: pre-trained embeddings with shape (vocab, embedding_dims)
    """
    with open(path, 'rb') as fin:
        _embeddings, _vocab = pickle.load(fin)
    embedding_dims = _embeddings.shape[1]
    embeddings = init_embeddings(vocab, embedding_dims)
    for word, id in vocab.items():
        if word in _vocab:
            embeddings[id] = _embeddings[_vocab[word]]
    return embeddings.astype(np.float32)


# normalize the word embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


# time cost
def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds=int(round(diff)))


# build vocabulary according the training data
def build_vocab(dataPath, vocabPath, threshold=0, lowercase=True):
    """
    :param dataPath: path of training data file
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :param lowercase: boolean, lower words or not
    """
    cnt = Counter()
    with open(dataPath, mode='r', encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('\t')
                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                words1 = list(l1)
                for word in list(words1):
                    cnt[word] += 1
                words2 = list(l2)
                for word in list(words2):
                    cnt[word] += 1
            except:
                pass
    cntDict = [item for item in cnt.items() if item[1] >= threshold]
    cntDict = sorted(cntDict, key=lambda d: d[1], reverse=True)
    wordFreq = ['||'.join([word, str(freq)]) for word, freq in cntDict]
    with open(vocabPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(wordFreq) + '\n')
    print('Vacabulary is stored in : {}'.format(vocabPath))


# load vocabulary
def load_vocab(vocabPath, threshold=0):
    """
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :return: vocab: vocabulary dict {word : index}
    """
    vocab = {}
    index = 2
    vocab[PADDING] = 0
    vocab[UNKNOWN] = 1
    with open(vocabPath, encoding='utf-8') as f:
        for line in f:
            items = [v.strip() for v in line.split('||')]
            if len(items) != 2:
                print('Wrong format: ', line)
                continue
            word, freq = items[0], int(items[1])
            if freq >= threshold:
                vocab[word] = index
                index += 1
    return vocab


# data preproceing, convert words into indexes according the vocabulary
def sentence2Index(dataPath, vocabDict, maxLen=30, lowercase=True):
    """
    :param dataPath: path of data file
    :param vocabDict: vocabulary dict {word : index}
    :param maxLen: max length of sentence, if a sentence longer than maxLen, cut off it
    :param lowercase: boolean, lower words or not
    :return: s1Pad: padded sentence1
             s2Pad: padded sentence2
             s1Mask: actual length of sentence1
             s2Mask: actual length of sentence2
    """
    s1List, s2List, labelList = [], [], []
    s1Mask, s2Mask = [], []
    with open(dataPath, mode='r', encoding='utf-8') as f:
        for line in f:
            try:
                l, s1, s2 = [v.strip() for v in line.strip().split('\t')]
                if s1 == s2 and l == "0":
                    l = "1"
                if lowercase:
                    s1, s2 = s1.lower(), s2.lower()
                s1 = [v.strip() for v in list(s1)]
                s2 = [v.strip() for v in list(s2)]
                if len(s1) > maxLen:
                    s1 = s1[:maxLen]
                if len(s2) > maxLen:
                    s2 = s2[:maxLen]
                if l in CATEGORIE_ID:
                    labelList.append([CATEGORIE_ID[l]])
                    s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
                    s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
                    s1Mask.append(len(s1))
                    s2Mask.append(len(s2))
            except:
                ValueError('Input Data Value Error!')
    s1Pad, s2Pad = tf.keras.preprocessing.sequence.pad_sequences(s1List, maxLen,
                                                                 padding='post'), tf.keras.preprocessing.sequence.pad_sequences(
        s2List, maxLen, padding='post')
    s1MaskList, s2MaskList = (s1Pad > 0).astype(np.int32), (s2Pad > 0).astype(np.int32)
    # enc = OneHotEncoder(sparse=False)
    # labelList = enc.transform(labelList)
    labelList = kr.utils.to_categorical(labelList, num_classes=2)
    print(labelList)
    # s1Mask = np.asarray(s1Mask, np.int32)
    # s2Mask = np.asarray(s2Mask, np.int32)
    # labelList = np.asarray(labelList, np.int32)
    print(labelList.dtype)
    return s1Pad, s1MaskList, s2Pad, s2MaskList, labelList


# generator : generate a batch of data
def next_batch(premise, premise_mask, hypothesis, hypothesis_mask, y, batchSize=64, shuffle=True):
    """
    :param premise_mask: actual length of premise
    :param hypothesis_mask: actual length of hypothesis
    :param shuffle: boolean, shuffle dataset or not
    :return: generate a batch of data (premise, premise_mask, hypothesis, hypothesis_mask, label)
    """
    sampleNums = len(premise)
    batchNums = int((sampleNums - 1) / batchSize) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(sampleNums))
        premise = premise[indices]
        premise_mask = premise_mask[indices]
        hypothesis = hypothesis[indices]
        hypothesis_mask = hypothesis_mask[indices]
        y = y[indices]

    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, sampleNums)
        yield (premise[startIndex: endIndex], premise_mask[startIndex: endIndex],
               hypothesis[startIndex: endIndex], hypothesis_mask[startIndex: endIndex],
               y[startIndex: endIndex])


# convert SNLI dataset from json to txt (format : gold_label || sentence1 || sentence2)
def convert_data(jsonPath, txtPath):
    """
    :param jsonPath: path of SNLI dataset file
    :param txtPath: path of output
    """
    fout = open(txtPath, 'w')
    with open(jsonPath) as fin:
        i = 0
        cnt = {key: 0 for key in CATEGORIE_ID.keys()}
        cnt['-'] = 0
        for line in fin:
            text = json.loads(line)
            cnt[text['gold_label']] += 1
            print('||'.join([text['gold_label'], text['sentence1'], text['sentence2']]), file=fout)

            i += 1
            if i % 10000 == 0:
                print(i)

    for key, value in cnt.items():
        print('#{0} : {1}'.format(key, value))
    print('Source data has been converted from "{0}" to "{1}".'.format(jsonPath, txtPath))


# convert embeddings from txt to format : (embeddings, vocab_dict)
def convert_embeddings(srcPath, dstPath):
    """
    :param srcPath: path of source embeddings
    :param dstPath: path of output
    """
    vocab = {}
    id = 0
    wrongCnt = 0
    with open(srcPath, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        wordNums = len(lines)
        line = lines[0].strip().split()
        vectorDims = len(line) - 1
        embeddings = np.zeros((wordNums, vectorDims), dtype=np.float32)
        for line in lines:
            items = line.strip().split()
            if len(items) != vectorDims + 1:
                wrongCnt += 1
                print(line)
                continue
            if items[0] in vocab:
                wrongCnt += 1
                print(line)
                continue
            vocab[items[0]] = id
            embeddings[id] = [float(v) for v in items[1:]]
            id += 1

        embeddings = embeddings[0: id, ]
        with open(dstPath, 'wb') as fout:
            pickle.dump([embeddings, vocab], fout)

        print('valid embedding nums : {0}, embeddings shape : {1},'
              ' wrong format embedding nums : {2}, total embedding nums : {3}'.format(len(vocab),
                                                                                      embeddings.shape,
                                                                                      wrongCnt,
                                                                                      wordNums))
        print('Original embeddings has been converted from {0} to {1}'.format(srcPath, dstPath))


# print log info on SCREEN and LOG file simultaneously
def print_log(*args, **kwargs):
    print(*args)
    if len(kwargs) > 0:
        print(*args, **kwargs)
    return None


# print all used hyper-parameters on both SCREEN an LOG file
def print_args(args, log_file):
    """
    :Param args: all used hyper-parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log("------------- HYPER PARAMETERS -------------", file=log_file)
    for a in argsList:
        print_log("%s: %s" % (a[0], str(a[1])), file=log_file)
    print("-----------------------------------------", file=log_file)
    return None


def build_word_pos(dataPath, vocabPath, posPath, threshold=0, lowercase=True):
    """
    :param dataPath: path of training data file
    :param vocabPath: path of vocabulary file
    :param threshold: mininum occurence of vocabulary, if a word occurence less than threshold, discard it
    :param lowercase: boolean, lower words or not
    """
    wordCnt = Counter()
    posCnt = Counter()
    with open(dataPath, mode='r', encoding='utf-8') as iF:
        for line in iF:
            try:
                if lowercase:
                    line = line.lower()
                tempLine = line.strip().split('\t')
                l1 = tempLine[1][:-1]
                l2 = tempLine[2][:-1]
                for word in list(l1):
                    wordCnt[word] += 1
                for word in list(l2):
                    wordCnt[word] += 1
            except:
                pass
    wordCntDict = [item for item in wordCnt.items() if item[1] >= threshold]
    wordCntDict = sorted(wordCntDict, key=lambda d: d[1], reverse=True)
    wordFreq = ['||'.join([word, str(freq)]) for word, freq in wordCntDict]

    with open(vocabPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(wordFreq) + '\n')
    print('word is stored in : {}'.format(vocabPath))


if __name__ == '__main__':
    build_word_pos('F:\python_work\github\pynlp\qaPairsRelationClassification\ESIM\data\\train.txt',
                   'F:\python_work\github\pynlp\qaPairsRelationClassification\ESIM\data\\vocab.txt',
                   "F:\python_work\github\pynlp\qaPairsRelationClassification\ESIM\data\pos.txt")
