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

# %% [code]
sns.countplot(x='Sex', hue='Survived',data=data)

# %% [markdown]
# [](http://)Pclass --> Oridinal Feature

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

# %% [code]
sns.factorplot(x='Pclass',y='Survived',data=data)
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
# # Name feature 2 Age ?

# %% [code]
data

# %% [code]

data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

# %% [code]
pd.crosstab(data.Initial,data.Sex).T

# %% [code]
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# %% [code]
data.groupby('Initial')['Age'].mean() # lets check the avaerage age by Initials

# %% [code]
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

# %% [code]
data.Age.isnull().any()

# %% [code]
f,ax = plt.subplots(1,2,figsize=(20,10))
data[data.Survived==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black',color='red')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data.Survived==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)


# %% [code]
sns.factorplot('Pclass','Survived',col='Initial',data=data)

# %% [markdown]
# Embarked --> Categorical VAlue

# %% [code]
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='summer_r')

# %% [markdown]
# chances for survival by port of embbarkation

# %% [code]
sns.factorplot('Embarked','Survived',data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()

# %% [code]
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()

# %% [code]
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)

# %% [code]
# filling embarked nan
data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()

# %% [markdown]
# sipsip--> Discrete Feature
# sibling = brother,sister,stepbrother,stepsister
# spouse = husband,wife

# %% [code]
pd.crosstab([data.SibSp],data.Survived)

# %% [code]
f, ax = plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])

# %% [code]
pd.crosstab(data.SibSp,data.Pclass)

# %% [markdown]
#
# Observations:
#
# The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. The graph roughly decreases if the number of siblings increase. This makes sense. That is, if I have a family on board, I will try to save them instead of saving myself first. Surprisingly the survival for families with 5-8 members is 0%. The reason may be Pclass??
#
# The reason is Pclass. The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.
#

# %% [markdown]
# Fare --> Continous Feature

# %% [code]
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data.Pclass==1].Fare,ax=ax[0])
sns.distplot(data[data.Pclass==2].Fare,ax=ax[1])
sns.distplot(data[data.Pclass==3].Fare,ax=ax[2])

# %% [markdown]
#
# Observations in a Nutshell for all features:
#
# Sex: The chance of survival for women is high as compared to men.
#
# Pclass:There is a visible trend that being a 1st class passenger gives you better chances of survival. The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!.
#
# Age: Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.
#
# Embarked: This is a very interesting feature. The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3.
#
# Parch+SibSp: Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.
#

# %% [markdown]
# Correlation Between The Features

# %% [code]
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# %% [markdown]
# Part2. Feature Engineering and DAta Cleaning