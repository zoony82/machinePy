import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

data = pd.read_csv('/home/jjh/문서/dataset/titanic/train.csv')
data.head()

data.isnull().sum() # checking for total null values

# The **Age, Cabin, Embarked** have null values. I will try to fix them.

# How maby Survived?

f, ax = plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()

# Sex -< Categorical Feature
type(data)
data.groupby(['Sex','Survived'])['Survived'].count()

f, ax = plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()

sns.countplot(x='Sex', hue='Survived',data=data)

# Pclass --> Oridinal Feature

pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')

f,ax = plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of passengers by pclass')
ax[0].set_ylabel('count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass : Survived vs Dead')
plt.show()

# Lets check survival rate with Sex and Pclass together
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True)

sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data)
plt.show()

sns.factorplot(x='Pclass',y='Survived',data=data)
plt.show()


# Age --> Continous Feature
print('Oldest Passenger was of :', data['Age'].max(),'Years')
print('Youngest Passenger was of :', data['Age'].min(),'Years')
print('Average Age on th ship :', data['Age'].mean(),'Years')

f, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot(x="Pclass",y="Age",hue="Survived", data=data, split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,100,10))
sns.violinplot(x="Sex",y="Age",hue="Survived", data=data, split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,100,10))
plt.show()

# Age feature has 177 null values,
# Name feature 2 Age ?
data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(data.Initial,data.Sex).T

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

data.groupby('Initial')['Age'].mean() # lets check the avaerage age by Initials

data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

data.Age.isnull().any()

f,ax = plt.subplots(1,2,figsize=(20,10))
data[data.Survived==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black',color='red')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data.Survived==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)

sns.factorplot('Pclass','Survived',col='Initial',data=data)

# Embarked --> Categorical VAlue
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='summer_r')

# chances for survival by port of embbarkation
sns.factorplot('Embarked','Survived',data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()

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


sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)

# filling embarked nan
data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()

# sipsip--> Discrete Feature
# sibling = brother,sister,stepbrother,stepsister
# spouse = husband,wife

pd.crosstab([data.SibSp],data.Survived)

f, ax = plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])


pd.crosstab(data.SibSp,data.Pclass)

# Observations:
#
# The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. The graph roughly decreases if the number of siblings increase. This makes sense. That is, if I have a family on board, I will try to save them instead of saving myself first. Surprisingly the survival for families with 5-8 members is 0%. The reason may be Pclass??
# The reason is Pclass. The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.

# Fare --> Continous Feature
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data.Pclass==1].Fare,ax=ax[0])
sns.distplot(data[data.Pclass==2].Fare,ax=ax[1])
sns.distplot(data[data.Pclass==3].Fare,ax=ax[2])

# Observations in a Nutshell for all features:
# Sex: The chance of survival for women is high as compared to men.
# Pclass:There is a visible trend that being a 1st class passenger gives you better chances of survival. The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!.
# Age: Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.
# Embarked: This is a very interesting feature. The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3.
# Parch+SibSp: Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.

# Correlation Between The Features
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

# Part2. Feature Engineering and Data Cleaning

# continous values into categorical values

data['Age']
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4

data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')

# statistics pclass & age to survived
sns.factorplot(x='Age_band',y='Survived',data=data, col='Pclass')

# famili size and alone
data['Family_size']=0
data['Family_size']=data['Parch'] + data['SibSp']
data['Alone'] = 0
data.loc[data.Family_size==0,'Alone'] = 1

f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=data,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()

sns.factorplot(x='Alone',y='Survived',data=data,hue='Sex',col='Pclass')

# Fare_range
data['Fare_range']=pd.qcut(data.Fare,4)

data.groupby('Fare_range')['Survived'].mean().to_frame()

del data['Fare_cat']

data['Fare_cat']=0
data.loc[data.Fare<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
