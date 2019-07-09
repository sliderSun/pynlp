"""
Created on @Time:2019/7/9 15:34
@Author:sliderSun 
@FileName: generate_char.py
"""
from qaPairsRelationClassification.utils.preprocess import MyVocabularyProcessor
import numpy as np
def generate(x2_text, x1_text, max_document_length):
    print("Building vocabulary")
    vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
    vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
    f = open("F:\python_work\github\pynlp\qaPairsRelationClassification\data\\vocab.txt", "w",encoding="utf-8")
    for i in range(len(vocab_processor.vocabulary_)):
        f.write(vocab_processor.vocabulary_.reverse(i)+"\n")
    print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
