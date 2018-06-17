'''
https://www.kaggle.com/c/word2vec-nlp-tutorial
data : https://www.kaggle.com/ymanojkumar023/kumarmanoj-bag-of-words-meets-bags-of-popcorn/data

텍스트 이해를위한 심층 학습 :  파트 2 및 3 에서 Word2Vec을 사용하여 모델을 교육하는 방법과 정서 분석에 결과 단어 벡터를 사용하는 방법에 대해 알아 봅니다.

https://github.com/wendykan/DeepLearningMovies/blob/master/Word2Vec_AverageVectors.py

The code in this file is for part 2 of the tutorial bag of centroids for a word2vec model.
'''

# Load a pre-trained model
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

def makeFeatureVec(words, model, num_features):
    # function to average all of the word vectors in a given paragraph

    # pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.

    # index2word is a list that contains the names of the words in the modes's vocabulary
    # convert it to a set, for a speed
    index2word_set = set(model.wv.index2word)

    # loop over each word in the review and,
    # if it is an the modes's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    # divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)

    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # given a set of reviews (each one a list of words)
    # calculate the average feature vector for each one and return a 2D numpy array

    # initialize a counter
    counter = 0.

    # preallocate a 2d numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    # loop through the reviews
    for review in reviews:
        #print a status message every 100th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)

        # increment the counter
        counter = counter + 1.

    return reviewFeatureVecs

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

if __name__ == '__main__':
    # Read data from files
    train = pd.read_csv('/home/insam/09_data/bagofwords/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('/home/insam/09_data/bagofwords/testData.tsv', header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv('/home/insam/09_data/bagofwords/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    train["review"].size #25,000
    test["review"].size  #25,000
    unlabeled_train["review"].size #50,000

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # split the labeled and unlabeled training sets into clean sentences
    sentences = []

    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    # set parameters and train the word2vec model

    # import the built-in logging module and configure it so that word2vec
    # creates nice output messages
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for variaous parameters
    num_features = 300  # word vector dimensionality
    min_word_count = 40 # minimum word count
    num_workers = 4     # number of threads to run in parallel
    context = 10        # context window size
    downsampling = 1e-3 # downsample setting for frequent words

    # initialize and train the model (this will take some time)
    model = Word2Vec(sentences,workers=num_workers, size=num_features, min_count=min_word_count,window=context, sample=downsampling, seed=1)

    # if you don't plan to train the model any further,
    # calling init_sims will make the model much more memory much more memory-efficient.
    model.init_sims(replace=True)

    # it can be helpful to create a meaningful model name
    # and save the model for later use.
    # you caon load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.most_similar("man")
    model.most_similar("awful")

    # create average vectors for the training and test sets
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)
    type(trainDataVecs)
    trainDataVecs.shape # (25000,300)
    testDataVecs.shape # (25000,300)

    # fit a random forest to the training set, then make predictions
    # using 100 trees
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # test & extract results
    result = forest.predict(testDataVecs)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    output.to_csv('/home/insam/09_data/bagofwords/bag_of_words_model_AverageVectors_jjh.csv')
