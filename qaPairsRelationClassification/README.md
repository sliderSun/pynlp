# Deep Learning for Text Pairs Relation Classification

This repository is my bachelor graduation project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.).

The main objective of the project is to determine whether the two sentences are similar in sentence meaning (binary classification problems) by the two given sentences based on Neural Networks (Fasttext, CNN, LSTM, etc.).

## Requirements

- Python 3.6
- Tensorflow 1.12
- Numpy
- Gensim

## Innovation

### Data part
1. Make the data support **Chinese** and English.(Which use `jieba` seems easy)
2. Can use **your own pre-trained word vectors**.(Which use `gensim` seems easy)
3. Add embedding visualization based on the **tensorboard**.

### Model part
1. Deign **two subnetworks** to solve the task --- Text Pairs Similarity Classification.
2. Add the correct **L2 loss** calculation operation.
3. Add **gradients clip** operation to prevent gradient explosion.
4. Add **learning rate decay** with exponential decay.
5. Add a new **Highway Layer**.(Which is useful according to the model performance)
6. Add **Batch Normalization Layer**.
7. Add several performance measures(especially the **AUC**) since the data is imbalanced.

### Code part
0. 生成字向量 utils/ `python load_data.py`
1. 词向量的训练 word2vec/

    静态词向量，请执行:
`python word2vec_static.py`，该版本是采用gensim来训练词向量

    动态词向量，请执行:
`python word2vec_dynamic.py`，该版本是采用tensorflow来训练词向量，训练完成后会保存embedding矩阵、词典和词向量在二维矩阵的相对位置的图片，
2. run `python train.py` to train   or   run `python test.py` to test

## Data

See data format in `data` folder which including the data sample files.

### Text Segment

You can use `jieba` package if you are going to deal with the chinese text data.

### Data Format

This repository can be used in other datasets(text pairs similarity classification) by two ways:
1. Modify your datasets into the same format of the sample.

Anyway, it should depends on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors(based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Network Structure

### TextANN

![](https://farm2.staticflickr.com/1980/45660411461_afa9be1182_o.png)

References:

- **Personal ideas 🙃**

---


### TextCNN

![](https://farm2.staticflickr.com/1979/44907317574_cd090e115f_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1916/44745805425_f0e2e5edd0_o.png)

#### TODO

1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1966/30719730157_1c2f59a22c_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

![](https://farm2.staticflickr.com/1942/45660487191_32c9b40bc9_o.png)

References:

- **Personal ideas 🙃**

---

### TextHAN
![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/TextHAN.jpg)

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/TextSANN.jpg)

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Add attention penalization loss.
2. Add visualization.

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

### TextABCNN

![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/ABCNN-1.jpg)
![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/ABCNN-2.jpg)
![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/ABCNN-3.jpg)

**Warning: Only achieve the ABCNN-1 Model🤪!**

#### TODO

1. Add ABCNN-3 model.

References:

- [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf)

---

### DeepSiamesNet
![](https://github.com/sliderSun/pynlp/blob/master/qaPairsRelationClassification/image/DSN.jpg)

#### TODO

1. Add DeepSiamesNet model.

References:

- [Learning Text Similarity with Siamese Recurrent Networks](https://www.aclweb.org/anthology/W16-1617)

---


### ESIM

#### TODO

1. Add ESIM model.

References:

- [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf)

---

### Bert

#### TODO

1. Add Bert model.

References:

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

---
