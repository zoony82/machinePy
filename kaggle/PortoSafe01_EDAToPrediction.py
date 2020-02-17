#https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/overview
#https://www.kaggle.com/gpreda/porto-seguro-exploratory-analysis-and-prediction

#Porto Seguro's Safe Driver Prediction
#prize = 25000*1200 = 30,000,000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

# pd.set_option('display.max_columns',100)

#load the data
trainset = pd.read_csv("/home/jjh/문서/dataset/porto_driver/train.csv")
testset = pd.read_csv("/home/jjh/문서/dataset/porto_driver/test.csv")

trainset.head()

#Few quick observation
'''
We can make few observations based on the data description in the competition:

Few groups are defined and features that belongs to these groups include patterns in the name (ind, reg, car, calc).
The ind indicates most probably individual, reg is probably registration, car is self-explanatory, calc suggests a calculated field;
The postfix bin is used for binary features;
The postfix cat to is used for categorical features;
Features without the bin or cat indications are real numbers (continous values) of integers (ordinal values);
A missing value is indicated by -1;
The value that is subject of prediction is in the target column. 
This one indicates whether or not a claim was filed for that insured person;
id is a data input ordinal number.
Let's glimpse the data to see if these interpretations are confirmed.
'''
trainset.columns
trainset['ps_car_01_cat']
trainset['ps_ind_15']
trainset['ps_ind_18_bin']
# Indeed, we can observe the cat values are categorical, integer values ranging from 0 to n, bin values are binary (either 0 or 1).

# Let's see how many rows and columns are in the data.
trainset.shape
testset.shape
len(trainset.columns)
len(testset.columns)

'''
There are 59 columns in the training dataset and only 58 in the testing dataset. 
Since from this dataset should have been extracted the target, this seems fine. 
Let's check the difference between the columns set in the two datasets, to make sure everything is fine.
'''
set(trainset.columns) - set(testset.columns)
#'target'

#Introduction of metadata
'''
To make easier the manipulation of data, 
we will associate few meta-information to the variables in the trainset. 
This will facilitate the selection of various types of features for analysis, inspection or modeling. 
We are using as well a category field for the car, ind, reg and calc types of features.

What metadata will be used:
use: input, ID, target
type: nominal, interval, ordinal, binary
preserve: True or False
dataType: int, float, char
category: ind, reg, car, calc
'''

data=[]

for feature in trainset.columns:
    # Defining the role
    if feature == 'target':
        use = 'target'
    elif feature == 'id':
        use = 'id'
    else:
        use = 'input'

    # Defining the type
    if 'bin' in feature or feature =='target':
        type = 'binary'
    elif 'cat' in feature or feature == 'id':
        type = 'categorical'
    elif trainset[feature].dtype == float or isinstance(trainset[feature].dtype, float):
        type = 'real'
    elif trainset[feature].dtype == int:
        type = 'integer'

    #Initialize preserve to True for all variable excepted for id
    preserve = True
    if feature == 'id':
        preserve = False

    # Defining the data type
    dtype = trainset[feature].dtype

    # Defining the category
    category = 'none'
    if 'ind' in feature:
        category = 'individual'
    elif 'reg' in feature:
        category = 'registration'
    elif 'car' in feature:
        category = 'car'
    elif 'calc' in feature:
        category = 'calculated'

    #creating a Dict that contains all the metadata for the variable
    feature_dictionary = {
        'varname' : feature,
        'use' : use,
        'type' : type,
        'preserve' : preserve,
        'dtype' : dtype,
        'category' : category
    }
    data.append(feature_dictionary)

metadata = pd.DataFrame(data, columns=['varname','use','type','preserve','dtype','category'])
metadata.set_index('varname',inplace=True)
metadata

# we can extract, for example, all category values
metadata.type.value_counts()
metadata.preserve.value_counts()
metadata[(metadata.type == 'categorical') & (metadata.preserve)].index

# Let's inspect all features, to see how many category distinct values do we have:
metadata.groupby('category').count()
{'count' : metadata.groupby(['category'])['category'].size()}
pd.DataFrame({'count' : metadata.groupby(['category'])['category'].size()}).reset_index()

'''
There are one nominal feature (the id), 20 binary values, 21 real (or float numbers), 16 categorical features 
- all these being as well input values and one target value, which is as well binary, the target.
'''

# Data analysis and statistics
