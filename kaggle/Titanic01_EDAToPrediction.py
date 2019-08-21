# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# **Part1. Exploratory Data Analysis**

# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')



# %% [code]
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()

# %% [code]
data.isnull().sum() # checking for total null values

# %% [markdown]
# The **Age, Cabin, Embarked** have null values. I will try to fix them.

# %% [markdown]
# How maby Survived?

# %% [code]
f, ax = plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()

# %% [markdown]
# Sex -< Categorical Feature

# %% [code]
type(data)
data.groupby(['Sex','Survived'])['Survived'].count()

# %% [code]
f, ax = plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()

# %% [markdown]
# Pclass --> Oridinal Feature

# %% [code]
pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')

# %% [code]
f,ax = plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of passengers by pclass')
ax[0].set_ylabel('count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass : Survived vs Dead')
plt.show()

# %% [markdown]
# Lets check survival rate with Sex and Pclass together

# %% [code]
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True)

# %% [code]
sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data)
plt.show()

# %% [markdown]
# Age --> Continous Feature

# %% [code]
print('Oldest Passenger was of :', data['Age'].max(),'Years')
print('Youngest Passenger was of :', data['Age'].min(),'Years')
print('Average Age on th ship :', data['Age'].mean(),'Years')

# %% [code]
f, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot(x="Pclass",y="Age",hue="Survived", data=data, split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,100,10))
sns.violinplot(x="Sex",y="Age",hue="Survived", data=data, split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,100,10))
plt.show()

# %% [markdown]
# Age feature has 177 null values,
# Name feature 2 Age ?

# %% [code]
data
data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

# %% [code]
pd.crosstab(data.Initial,data.Sex).T