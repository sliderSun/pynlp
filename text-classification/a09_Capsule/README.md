# Text Classification with Capsule Network
Implementation of our paper 
["Investigating Capsule Networks with Dynamic Routing for Text Classification"](https://arxiv.org/pdf/1804.00538.pdf) which is accepted by EMNLP-18.

Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

# Data Preparation
The reuters_process.py provides functions to clean the raw data and generate Reuters-Multilabel and Reuters-Full datasets. For quick start, please refer to [downloadDataset](https://drive.google.com/open?id=1a4rB6B1FDf7epZZlwXIppaSA7Nr8wSpt) for the Reuters-Multilabel dataset.

# More explanation 
The utils.py includes several wrapped and fundamental functions such as _conv2d_wrapper, _separable_conv2d_wrapper and _get_variable_wrapper etc.

The layers.py implements capsule network including Primary Capsule Layer, Convolutional Capsule Layer, Capsule Flatten Layer and FC Capsule Layer.

The network.py provides the implementation of two kinds of capsule network as well as baseline models for the comparison.

The loss.py provides the implementation of three kinds of loss function: cross entropy, margin loss and spread loss.

# Quick start

```bash

python ./main.py --model_type capsule-A --learning_rate 0.001
```

Notes: Val accuracy and loss are evaluated on dev (single-label), the metrics such as ER and Precision are evaluated on test (multi-label).

The main functions are already in this repository. For any questions, you can report issue here.

# Reference
If you find our source code useful, please consider citing our work.
```
@article{zhao2018investigating,
  title={Investigating Capsule Networks with Dynamic Routing for Text Classification},
  author={Zhao, Wei and Ye, Jianbo and Yang, Min and Lei, Zeyang and Zhang, Suofei and Zhao, Zhou},
  journal={arXiv preprint arXiv:1804.00538},
  year={2018}
}

@article{zhang2018fast,
  title={Fast Dynamic Routing Based on Weighted Kernel Density Estimation},
  author={Zhang, Suofei and Zhao, Wei and Wu, Xiaofu and Zhou, Quan},
  journal={arXiv preprint arXiv:1805.10807},
  year={2018}
}
```

Our second paper makes Capsule Network in relation with Kernel Density Estimation, and provides routing algorithm with explicit objective function to minimize.
