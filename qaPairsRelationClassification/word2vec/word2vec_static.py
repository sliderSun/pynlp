from gensim.models import Word2Vec
import pandas as pd
import jieba
from qaPairsRelationClassification.BIMPM import args

df = pd.read_csv('F:\python_work\github\pynlp\qaPairsRelationClassification\data\\train.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts = []
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('F:\python_work\github\pynlp\qaPairsRelationClassification\data\dev.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)

df = pd.read_csv('F:\python_work\github\pynlp\qaPairsRelationClassification\data\\test.csv')
p = df['sentence1'].values
h = df['sentence2'].values
p_seg = list(map(lambda x: list(jieba.cut(x)), p))
h_seg = list(map(lambda x: list(jieba.cut(x)), h))
common_texts.extend(p_seg)
common_texts.extend(h_seg)
model = Word2Vec(common_texts, size=args.word_embedding_len, window=5, min_count=0, workers=12)

model.save("output/word2vec/word2vec.model")
