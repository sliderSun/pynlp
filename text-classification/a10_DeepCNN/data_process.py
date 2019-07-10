import pandas as pd
from collections import defaultdict
import numpy as np


def load_data_from_csv(data_path):
    data = pd.read_csv(data_path)
    return data


def explore_data_analysis(data):
    sents = data.iloc[:, 0]
    print(sents)
    sents_len = np.array(list(map(lambda x: len(x), sents)))

    print("the index of the longest sentence is %s" % np.argmax(sents_len))
    print("the longest sentence is %s" % sents[np.argmax(sents_len)])
    print("the length of the longest sentence is %s" % np.max(sents_len))
    print("the length of 75%% sentence is %s" % np.percentile(sents_len, 75))
    print("the mean length is %s" % np.mean(sents_len))


def write_vocab(data, write_path, min_count=5):
    wordcount = defaultdict(int)
    data = data.tolist()
    for sent in data:
        for word in sent.replace('\n','').replace('"','').replace("\r",''):
            wordcount[word] += 1
    word2id = {word: freq for word, freq in wordcount.items() if freq >= min_count}
    word2id = sorted(word2id.items(), key=lambda x: x[1], reverse=True)
    with open(write_path, 'w', encoding='utf8') as f:
        for item in word2id:
            f.write(item[0] + ';;' + str(item[1]) + '\n')


def load_vocab(path):
    with open(path, 'r', encoding='utf8') as f:
        context = f.readlines()
    word2id = {}
    for ix, vocab in enumerate(context):
        word2id[vocab.split(';;')[0]] = ix + 2
    return word2id


def string2id(sent, word2id):
    id_ = [word2id.get(w, 1) for w in sent]
    return id_
