'''
https://www.kaggle.com/c/word2vec-nlp-tutorial
data : https://www.kaggle.com/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn/data

기본 자연 언어 처리 :  이 자습서의 제 1 부는 초보자를 대상으로하며 자습서의 뒷부분에 필요한 기본 자연 언어 처리 기술을 다룹니다.
https://github.com/wendykan/DeepLearningMovies/blob/master/BagOfWords.py

'''

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import nltk
#import gensim, logging


train = pd.read_csv('/home/insam/09_data/bagofwords/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test = pd.read_csv('/home/insam/09_data/bagofwords/testData.tsv', header=0, delimiter="\t", quoting=3)

print("the first review is : ")
print(train["review"][0])

#nltk download
#nltk.download() # download text data sets, including stop words

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list

print("Cleaning and parsing the training set movie reviews...\n")
for i in range(0, len(train["review"])):
    clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


print(type(clean_train_reviews))
clean_train_reviews[0]
clean_train_reviews[0:2]


print("Creating the bag of words...")

# Initialize the "CountVectorizer" object, which is sckit-learn's
# bag of words tool.

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,max_features=5000)

# fit_transform() does two functions :
# First, it fits the model and learns the vocabulary;
# Second, it transforms out training data into feature vectors.
# The input to fit_transform should be a list of strings.

train_data_features = vectorizer.fit_transform(clean_train_reviews)
type(train_data_features) # scipy.sparse.csr.csr_matrix

# Numpy arrays are easy to work with, so convert the result to an array
np.asarray(train_data_features)
type(train_data_features)

train_data_features.shape


print("Training the random forest")
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set,
# using the bag of words as features and the sentiment labels as the response variable

forest = forest.fit(train_data_features, train["sentiment"])

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

for i in range(0, len(test["review"])):
    clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)

test_data_features.shape

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

type(result)
result[1:5]

# Copy the result to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame(data = {"id" : test["id"], "sentiment" : result})

# Use pandas to write the comma-separated output file
output.to_csv('/home/insam/09_data/bagofwords/bag_of_words_model_jjh.csv')