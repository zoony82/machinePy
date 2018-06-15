'''
https://www.kaggle.com/c/word2vec-nlp-tutorial
data : https://www.kaggle.com/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn/data

텍스트 이해를위한 심층 학습 :  파트 2 및 3 에서 Word2Vec을 사용하여 모델을 교육하는 방법과 정서 분석에 결과 단어 벡터를 사용하는 방법에 대해 알아 봅니다.
'''

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
#import gensim, logging


train = pd.read_csv('/home/insam/09_data/bagofwords/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('/home/insam/09_data/bagofwords/testData.tsv', header=0, delimiter="\t", quoting=3)

print("the first review is : ")
print(train["review"][0])
