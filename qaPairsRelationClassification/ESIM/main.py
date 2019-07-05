# /usr/bin/env python
# coding=utf-8
import numpy as  np
import tensorflow as tf
import sys

UNKNOWN = '<UNK>'
PADDING = '<PAD>'
CATEGORIE_ID = {'0': 0, '1': 1}


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


def init_embeddings(vocab, embedding_dims):
    """
    :param vocab: word nums of the vocabulary
    :param embedding_dims: dimension of embedding vector
    :return: randomly init embeddings with shape (vocab, embedding_dims)
    """
    rng = np.random.RandomState(None)
    random_init_embeddings = rng.normal(size=(len(vocab), embedding_dims))
    return random_init_embeddings.astype(np.float32)


# normalize the word embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


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
    s1List, s2List = [], []
    with open(dataPath, mode='r', encoding='utf-8') as f:
        for line in f:
            lines = [v.strip() for v in line.strip().split('\t')]
            if len(lines) < 3:
                print("数据异常")
                continue
            try:
                l, s1, s2 = lines
                if lowercase:
                    s1, s2 = s1.lower(), s2.lower()
                s1 = [v.strip() for v in list(s1)]
                s2 = [v.strip() for v in list(s2)]
                if len(s1) > maxLen:
                    s1 = s1[:maxLen]
                if len(s2) > maxLen:
                    s2 = s2[:maxLen]
                if l in CATEGORIE_ID:
                    s1List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s1])
                    s2List.append([vocabDict[word] if word in vocabDict else vocabDict[UNKNOWN] for word in s2])
            except Exception as e:
                print(e.__repr__())
    s1Pad = tf.keras.preprocessing.sequence.pad_sequences(s1List, maxLen, padding='post')
    s2Pad = tf.keras.preprocessing.sequence.pad_sequences(s2List, maxLen, padding='post')
    return s1Pad, s2Pad


def next_batch(premise, hypothesis, batchSize=500):
    """
    :param premise_mask: actual length of premise
    :param hypothesis_mask: actual length of hypothesis
    :param shuffle: boolean, shuffle dataset or not
    :return: generate a batch of data (premise, premise_mask, hypothesis, hypothesis_mask, label)
    """
    sampleNums = len(premise)
    batchNums = int((sampleNums - 1) / batchSize) + 1
    for i in range(batchNums):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, sampleNums)
        yield (premise[startIndex: endIndex],
               hypothesis[startIndex: endIndex])


def process(test_path, output_path):
    vocab_path, model_path = "F:\python_work\siamese-lstm-network\ESIM\data\\vocab.txt", "F:\python_work\siamese-lstm-network\ESIM\model.pb"
    sess = tf.Session()
    with tf.gfile.GFile(model_path, 'rb') as f:  # 加载模型
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    premise = sess.graph.get_tensor_by_name("premise:0")
    hypothesis = sess.graph.get_operation_by_name("hypothesis").outputs[0]
    dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    logit = sess.graph.get_operation_by_name("composition/feed_forward/feed_foward_layer2/dense/Tanh").outputs[0]
    vocab_dict = load_vocab(vocab_path)
    premise_test, hypothesis_test = sentence2Index(test_path, vocab_dict)
    batches = next_batch(premise_test, hypothesis_test)
    with open(output_path, 'w+') as fout:
        i = 0
        for batch in batches:
            i += 1
            y_pred = []
            batch_premise_test, batch_hypothesis_test = batch
            feed_dict = {premise: batch_premise_test,
                         hypothesis: batch_hypothesis_test,
                         dropout_keep_prob: 1.0}
            logits = sess.run([logit], feed_dict=feed_dict)
            logits = np.array(logits)
            logits = logits.reshape([-1, logits.shape[-1]])
            y_pred.extend(logits)
            # evaluating
            y_pred = np.argmax(y_pred, 1)
            for index in range(len(y_pred)):
                fout.write(str(index + 1 + (i - 1) * 500) + "\t" + str(y_pred[index]) + "\n")


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
