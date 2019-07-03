import pyspark


#regression
#리눅스 머신에 아래 설치
'''
2.1 python-pip, python-devel 설치

    # sudo yum install -y epel-release 

    # sudo yum install python-pip python-devel

    # sudo yum install python-devel openldap-devel

   # pip install --ignore-installed pyldap dnspython
   
   # pip uninstall numpy 여러번

    # pip install --ignore-installed numpy matplotlib
    
    

    # yum install tkinter


'''

# 얼마나 자전거가 대여가 될지를 예측 하자.

import pandas as pd
bike_hour = pd.read_csv("D:\\regression\\hour.csv")
bike_hour.head(5)

bike_day = pd.read_csv("D:\\regression\\day.csv")
bike_day.head(5)

# coding: utf-8
# Extracting Features from the Bike Sharing Dataset
# In[1]:


# first remove the headers by using the 'sed' command:
# sed 1d hour.csv > hour_noheader.csv
path = "hdfs://192.168.56.11:9000/user/bigdata/bike/hour_noheader.csv"


conf = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '1'), ('spark.cores.max', '3'), ('spark.driver.memory','8g')])
sc = pyspark.SparkContext(conf=conf)

path = "D:\\regression\\hour_noheader.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(","))
first = records.first()
print(first)
print(num_data)

# ## Extract the variables we want to keep
#
# _All variables:_
#
# `instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt`
#
# _Variables to keep:_
#
# `season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,cnt`
# ## Create feature mappings for categorical features


# cache the dataset to speed up subsequent operations
records.cache()


# function to get the categorical feature mapping for a given variable column
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()


# we want to extract the feature mappings for columns 2 - 9
# try it out on column 2 first
print("Mapping of first categorical feasture column: %s" % get_mapping(records, 2))

# extract all the catgorical mappings
mappings = [get_mapping(records, i) for i in range(2, 10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11:15])
total_len = num_len + cat_len
print("Feature vector length for categorical features: %d" % cat_len) # 57
print("Feature vector length for numerical features: %d" % num_len) # 4
print("Total feature vector length: %d" % total_len) # 61

# ## Create Feature Vectors

# required imports
from pyspark.mllib.regression import LabeledPoint
import numpy as np


# function to use the feature mappings to extract binary feature vectors, and concatenate with
# the numerical feature vectors
# 원 핫 인코딩 => 스파크2에선 더 쉽게 될꺼다
# cat_vec(범주형을 57개로 인코딩) 과 num_vec 2개의 속성을 원핫인코딩으로 더하는 것
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2:9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)

    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))


# function to extract the label from the last column
def extract_label(record):
    return float(record[-1])


data = records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))
first_point = data.first()
print("Raw data: " + str(first[2:]))
print("Label: " + str(first_point.label))
print("Linear Model feature vector:\n" + str(first_point.features))
print("Linear Model feature vector length: " + str(len(first_point.features)))


# we need a separate set of feature vectors for the decision tree
def extract_features_dt(record):
    return np.array(map(float, record[2:14]))


data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))
# first_point_dt = data_dt.first() 여기서 에러
'''
print("Decision Tree feature vector: " + str(first_point_dt.features))
print("Decision Tree feature vector length: " + str(len(first_point_dt.features)))
'''

# # Training a Regression Model

from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree

help(LinearRegressionWithSGD.train)
help(DecisionTree.trainRegressor)

# ## Train a Regression Model on the Bike Sharing Dataset


linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
print("Linear Model predictions: " + str(true_vs_predicted.take(5)))

# we pass in an mepty mapping for categorical feature size {}
dt_model = DecisionTree.trainRegressor(data_dt, {}) #여기서 에러 뜨네...
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)
print("Decision Tree predictions: " + str(true_vs_predicted_dt.take(5)))
print("Decision Tree depth: " + str(dt_model.depth()))
print("Decision Tree number of nodes: " + str(dt_model.numNodes()))


# 자 이제, linear regression/decisiontree 의 잔차 제곱을 통해 성능을 비교해보자.

# ## Perfomance Metrics


# set up performance metrics functions

def squared_error(actual, pred):
    return (pred - actual) ** 2


def abs_error(actual, pred):
    return np.abs(pred - actual)


def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1)) ** 2


# ### Linear Model


# compute performance metrics for linear model
mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
print("Linear Model - Mean Squared Error: %2.4f" % mse)
print("Linear Model - Mean Absolute Error: %2.4f" % mae)
print("Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle)

# ### Decision Tree Model


# compute performance metrics for decision tree model
mse_dt = true_vs_predicted_dt.map(lambda (t, p): squared_error(t, p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
print("Decision Tree - Mean Squared Error: %2.4f" % mse_dt)
print("Decision Tree - Mean Absolute Error: %2.4f" % mae_dt)
print("Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt)

'''
제플린에서의 결과
잔차가 디시젼트리가 더 작다
Linear Model - Mean Squared Error: 30679.4539
Linear Model - Mean Absolute Error: 130.6429
Linear Model - Root Mean Squared Log Error: 1.4653
Decision Tree - Mean Squared Error: 11611.4860
Decision Tree - Mean Absolute Error: 71.1502
Decision Tree - Root Mean Squared Log Error: 0.6251
'''

# #############################################################
# #### Using the categorical feature mapping for Decision Tree


# we create the categorical feature mapping for decision trees
cat_features = dict([(i - 2, len(get_mapping(records, i)) + 1) for i in range(2, 10)])
print
"Categorical feature size mapping %s" % cat_features
# train the model again
dt_model_2 = DecisionTree.trainRegressor(data_dt, categoricalFeaturesInfo=cat_features)
preds_2 = dt_model_2.predict(data_dt.map(lambda p: p.features))
actual_2 = data.map(lambda p: p.label)
true_vs_predicted_dt_2 = actual_2.zip(preds_2)
# compute performance metrics for decision tree model
mse_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): squared_error(t, p)).mean()
mae_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_dt_2 = np.sqrt(true_vs_predicted_dt_2.map(lambda (t, p): squared_log_error(t, p)).mean())
print "Decision Tree - Mean Squared Error: %2.4f" % mse_dt_2
print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt_2
print "Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt_2

'''
카테고리를 하니까 에러가 더 줄어드네
확실히 디시젼트리는 범주형 데이터로 하는게 더 효율적이다.
Decision Tree - Mean Squared Error: 7912.5642
Decision Tree - Mean Absolute Error: 59.4409
Decision Tree - Root Mean Squared Log Error: 0.6192
'''

# # Transforming the Target Variable
# ## Distributon of Raw Target

import matplotlib.pyplot
import numpy
import StringIO


def show(p):
    img = StringIO.StringIO()
    p.savefig(img, format='svg')
    img.seek(0)
    print "%html <div style='width:1000px'>" + img.buf + "</div>"


targets = records.map(lambda r: float(r[-1])).collect()
matplotlib.pyplot.hist(targets, bins=40, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)
show(fig)
fig.clear()

# ## Distribution of Log Transformed Target
log_targets = records.map(lambda r: np.log(float(r[-1]))).collect()
matplotlib.pyplot.hist(log_targets, bins=40, color='lightblue', normed=True)
show(fig)
fig.clear()

# ## Distribution of Square-root Transformed target
sqrt_targets = records.map(lambda r: np.sqrt(float(r[-1]))).collect()
matplotlib.pyplot.hist(sqrt_targets, bins=40, color='lightblue', normed=True)
show(fig)
fig.clear()

# ## Impact of Training on Log Transformed Targets
# train a linear model on log-transformed targets
data_log = data.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)

true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label), np.exp(model_log.predict(p.features))))

mse_log = true_vs_predicted_log.map(lambda (t, p): squared_error(t, p)).mean()
mae_log = true_vs_predicted_log.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda (t, p): squared_log_error(t, p)).mean())
print "Mean Squared Error: %2.4f" % mse_log
print "Mean Absolute Error: %2.4f" % mae_log
print "Root Mean Squared Log Error: %2.4f" % rmsle_log
print "Non log-transformed predictions:\n" + str(true_vs_predicted.take(3))
print "Log-transformed predictions:\n" + str(true_vs_predicted_log.take(3))

'''
타겟 데이터를 정규화(로그를 통해... 근데 로그 해도 머 정규화 되지도 않네) 시켜서 해보면
타겟변수를 로그 적용해서 하면 오히려 잔차 제곱합이 늘어나고
루트민 스퀘어는 비슷하네..
'''

# train a decision tree model on log-transformed targets
data_dt_log = data_dt.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
dt_model_log = DecisionTree.trainRegressor(data_dt_log, {})

preds_log = dt_model_log.predict(data_dt_log.map(lambda p: p.features))
actual_log = data_dt_log.map(lambda p: p.label)
true_vs_predicted_dt_log = actual_log.zip(preds_log).map(lambda (t, p): (np.exp(t), np.exp(p)))

mse_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): squared_error(t, p)).mean()
mae_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_log_dt = np.sqrt(true_vs_predicted_dt_log.map(lambda (t, p): squared_log_error(t, p)).mean())
print
"Mean Squared Error: %2.4f" % mse_log_dt
print
"Mean Absolute Error: %2.4f" % mae_log_dt
print
"Root Mean Squared Log Error: %2.4f" % rmsle_log_dt
print
"Non log-transformed predictions:\n" + str(true_vs_predicted_dt.take(3))
print
"Log-transformed predictions:\n" + str(true_vs_predicted_dt_log.take(3))

# # Cross-validation
# ## Creating Training and Test Sets


# create training and testing sets for linear model
data_with_idx = data.zipWithIndex().map(lambda (k, v): (v, k))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (idx, p): p)
test_data = test.map(lambda (idx, p): p)

train_size = train_data.count()
test_size = test_data.count()
print
"Training data size: %d" % train_size
print
"Test data size: %d" % test_size
print
"Total data size: %d " % num_data
print
"Train + Test size : %d" % (train_size + test_size)

# create training and testing sets for decision tree
data_with_idx_dt = data_dt.zipWithIndex().map(lambda (k, v): (v, k))
test_dt = data_with_idx_dt.sample(False, 0.2, 42)
train_dt = data_with_idx_dt.subtractByKey(test_dt)

train_data_dt = train_dt.map(lambda (idx, p): p)
test_data_dt = test_dt.map(lambda (idx, p): p)


# ## Evaluating the Impact of Different Parameter Settings for Linear Model

# create a function to evaluate linear model
def evaluate(train, test, iterations, step, regParam, regType, intercept):
    model = LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType,
                                          intercept=intercept)
    tp = test.map(lambda p: (p.label, model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle


# ### Iterations
params = [1, 5, 10, 20, 50, 100]
metrics = [evaluate(train_data, test_data, param, 0.01, 0.0, 'l2', False) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')
show(fig)
fig.clear()

# ### Step Size
params = [0.01, 0.025, 0.05, 0.1, 1.0]
metrics = [evaluate(train_data, test_data, 10, param, 0.0, 'l2', False) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')
show(fig)
fig.clear()

# ### L2 Regularization
params = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l2', False) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')
show(fig)
fig.clear()

# ### L1 Regularization
params = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l1', False) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
matplotlib.pyplot.xscale('log')
show(fig)
fig.clear()

# Investigate sparsity of L1-regularized solution
model_l1 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=1.0, regType='l1', intercept=False)
model_l1_10 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=10.0, regType='l1', intercept=False)
model_l1_100 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=100.0, regType='l1', intercept=False)
print
"L1 (1.0) number of zero weights: " + str(sum(model_l1.weights.array == 0))
print
"L1 (10.0) number of zeros weights: " + str(sum(model_l1_10.weights.array == 0))
print
"L1 (100.0) number of zeros weights: " + str(sum(model_l1_100.weights.array == 0))

# ### Intercept
params = [False, True]
metrics = [evaluate(train_data, test_data, 10, 0.1, 1.0, 'l2', param) for param in params]
print
params
print
metrics
matplotlib.pyplot.bar(params, metrics, color='lightblue')
fig = matplotlib.pyplot.gcf()
show(fig)
fig.clear()


# ## Evaluating the Impact of Different Parameter Settings for Decision Tree Model

# create a function to evaluate decision tree model
def evaluate_dt(train, test, maxDepth, maxBins):
    model = DecisionTree.trainRegressor(train, {}, impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    preds = model.predict(test.map(lambda p: p.features))
    actual = test.map(lambda p: p.label)
    tp = actual.zip(preds)
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle


# ### Tree Depth
params = [1, 2, 3, 4, 5, 10, 20]
metrics = [evaluate_dt(train_data_dt, test_data_dt, param, 32) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
show(fig)
fig.clear()

# ### Max Bins
params = [2, 4, 8, 16, 32, 64, 100]
metrics = [evaluate_dt(train_data_dt, test_data_dt, 5, param) for param in params]
print
params
print
metrics
matplotlib.pyplot.plot(params, metrics)
fig = matplotlib.pyplot.gcf()
show(fig)
fig.clear()
