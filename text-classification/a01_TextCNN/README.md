# Text Classification with CNN and RNN

使用卷积神经网络以及循环神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## 环境

- Python 2/3
- TensorFlow 1.12
- numpy
- scikit-learn
- scipy


## 预处理

`data_loader.py`为数据的预处理文件。

```

### CNN模型

具体参看`cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](images/cnn_architecture.png)
```

## 训练与测试

运行 `python run_cnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。



## 预测

为方便预测，repo 中 `predict.py` 提供了 CNN 模型的预测方法。
