'''
Contents of the Notebook:
Part1: Exploratory Data Analysis(EDA):
1)Analysis of the features.

2)Finding any relations or trends considering multiple features.

Part2: Feature Engineering and Data Cleaning:
1)Adding any few features.

2)Removing redundant features.

3)Converting features into suitable form for modeling.

Part3: Predictive Modeling
1)Running Basic Algorithms.

2)Cross Validation.

3)Ensembling.

4)Important Features Extraction.
'''

# Part1: Exploratory Data Analysis(EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

data = pd.read_csv('C:\\04_dataset\\titanic\\train.csv')
data.head()

# checking for total null values
data.isnull().sum()

# The **Age, Cabin, Embarked** have null values. I will try to fix them.

# How maby Survived?

f, ax = plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()

'''
It is evident that not many passengers survived the accident.
Out of 891 passengers in training set, only around 350 survived i.e 
Only 38.4% of the total training set survived the crash. 
We need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.
We will try to check the survival rate by using the different features of the dataset. 
Some of the features being Sex, Port Of Embarcation, Age,etc.
First let us understand the different types of features.

Types Of Features
Categorical Features:
A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.
For example, gender is a categorical variable having two categories (male and female). 
Now we cannot sort or give any ordering to such variables. 
They are also known as Nominal Variables.

Categorical Features in the dataset: Sex,Embarked.

Ordinal Features:
An ordinal variable is similar to categorical values, 
but the difference between them is that we can have relative ordering or sorting between the values. 
For eg: If we have a feature like Height with values Tall, Medium, Short, then Height is a ordinal variable. 
Here we can have a relative sort in the variable.

Ordinal Features in the dataset: PClass

Continous Feature:
A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.
Continous Features in the dataset: Age

Analysing The Features
'''


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
'''
This looks interesting. 
The number of men on the ship is lot more than the number of women. 
Still the number of women saved is almost twice the number of males saved. 
The survival rates for a women on the ship is around 75% while that for men in around 18-19%.
This looks to be a very important feature for modeling. But is it the best?? Lets check other features.
'''


# Pclass --> Oridinal Feature

pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')

f,ax = plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of passengers by pclass')
ax[0].set_ylabel('count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass : Survived vs Dead')
plt.show()

'''
People say Money Can't Buy Everything. 
But we can clearly see that Passenegers Of Pclass 1 were given a very high priority while rescue. 
Even though the the number of Passengers in Pclass 3 were a lot higher, still the number of survival from them is very low, somewhere around 25%.
For Pclass 1 %survived is around 63% while for Pclass2 is around 48%. 
So money and status matters. Such a materialistic world.

Lets Dive in little bit more and check for other interesting observations. 
Lets check survival rate with Sex and Pclass Together.
'''

# Lets check survival rate with Sex and Pclass together
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True)

sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data)
plt.show()

sns.factorplot(x='Pclass',y='Survived',data=data)
plt.show()
'''
We use FactorPlot in this case, because they make the seperation of categorical values easy.
Looking at the CrossTab and the FactorPlot, we can easily infer that survival for Women from Pclass1 is about 95-96%, as only 3 out of 94 Women from Pclass1 died.
It is evident that irrespective of Pclass, Women were given first priority while rescue. 
Even Men from Pclass1 have a very low survival rate.
Looks like Pclass is also an important feature. 
Lets analyse other features.
'''

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

'''
Observations:
1)The number of children increases with Pclass and the survival rate for passenegers below Age 10(i.e children) looks to be good irrespective of the Pclass.
2)Survival chances for Passenegers aged 20-50 from Pclass1 is high and is even better for Women.
3)For males, the survival chances decreases with an increase in age.
As we had seen earlier, the Age feature has 177 null values. To replace these NaN values, we can assign them the mean age of the dataset.
But the problem is, there were many people with many different ages. 
We just cant assign a 4 year kid with the mean age that is 29 years. 
Is there any way to find out what age-band does the passenger lie??

Bingo!!!!, we can check the Name feature. 
Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. 
Thus we can assign the mean values of Mr and Mrs to the respective groups.
'''

data['Initial'] = 0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

'''
Okay so here we are using the Regex: [A-Za-z]+).. 
So what it does is, it looks for strings which lie between A-Z or a-z and followed by a .(dot). 
So we successfully extract the Initials from the Name.
'''
pd.crosstab(data.Initial,data.Sex).T

'''
Okay so there are some misspelled Initials like Mlle or Mme that stand for Miss. 
I will replace them with Miss and same thing for other values.
'''
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
# lets check the avaerage age by Initials
data.groupby('Initial')['Age'].mean()

#Filling NaN Ages
## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

##So no null values left finally
data.Age.isnull().any()

f,ax = plt.subplots(1,2,figsize=(20,10))
data[data.Survived==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black',color='red')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data.Survived==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
'''
Observations:
1)The Toddlers(age<5) were saved in large numbers(The Women and Child First Policy).
2)The oldest Passenger was saved(80 years).
3)Maximum number of deaths were in the age group of 30-40.
'''
sns.factorplot('Pclass','Survived',col='Initial',data=data)
'''
The Women and Child first policy thus holds true irrespective of the class.
'''

# Embarked --> Categorical VAlue
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True)


# chances for survival by port of embbarkation
sns.factorplot('Embarked','Survived',data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()

'''
The chances for survival for Port C is highest around 0.55 while it is lowest for S.
'''

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

'''
Observations:
1)Maximum passenegers boarded from S. Majority of them being from Pclass3.
2)The Passengers from C look to be lucky as a good proportion of them survived. 
The reason for this maybe the rescue of all the Pclass1 and Pclass2 Passengers.
3)The Embark S looks to the port from where majority of the rich people boarded. 
Still the chances for survival is low here, that is because many passengers from Pclass3 around 81% didn't survive.
4)Port Q had almost 95% of the passengers were from Pclass3.
'''

sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)
'''
Observations:
1)The survival chances are almost 1 for women for Pclass1 and Pclass2 irrespective of the Pclass.
2)Port S looks to be very unlucky for Pclass3 Passenegers as the survival rate for both men and women is very low.(Money Matters)
3)Port Q looks looks to be unlukiest for Men, as almost all were from Pclass 3.
'''

# filling embarked nan
#As we saw that maximum passengers boarded from Port S, we replace NaN with S
data['Embarked'].fillna('S',inplace=True)
data.Embarked.isnull().any()

# sipsip--> Discrete Feature
# sibling = brother,sister,stepbrother,stepsister
# spouse = husband,wife

'''
SibSip-->Discrete Feature
This feature represents whether a person is alone or with his family members.
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife
'''
pd.crosstab([data.SibSp],data.Survived)

f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()

pd.crosstab(data.SibSp,data.Pclass)

'''
Observations:
The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. 
The graph roughly decreases if the number of siblings increase. 
This makes sense. 
That is, if I have a family on board, I will try to save them instead of saving myself first. 
Surprisingly the survival for families with 5-8 members is 0%. The reason may be Pclass??
The reason is Pclass. 
The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.
'''

#Parch
pd.crosstab(data.Parch, data.Pclass)
'''
The corsstab again shows that large families were in pClass3
'''
f, ax = plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=data, ax=ax[0])
sns.factorplot('Parch','Survived',data=data, ax=ax[1])
plt.close(2)
'''
Observations:
Here too the results are quite similar. 
Passengers with their parents onboard have greater chance of survival. 
It however reduces as the number goes up.
The chances of survival is good for somebody who has 1-3 parents on the ship. 
Being alone also proves to be fatal and the chances for survival decreases when somebody has >4 parents on the ship.
'''

# Fare --> Continous Feature
print('Highest Fare was:',data['Fare'].max())
print('Lowest Fare was:',data['Fare'].min())
print('Average Fare was:',data['Fare'].mean())
#The lowest fare is 0.0. Wow!! a free luxorious ride.

f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()

'''
There looks to be a large distribution in the fares of Passengers in Pclass1 and this distribution goes on decreasing as the standards reduces. 
As this is also continous, we can convert into discrete values by using binning.
Observations in a Nutshell for all features:
Sex: The chance of survival for women is high as compared to men.
Pclass:There is a visible trend that being a 1st class passenger gives you better chances of survival. 
The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!.
Age: Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.
Embarked: This is a very interesting feature. 
The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3.
Parch+SibSp: Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.
'''


# Correlation Between The Features
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
'''
Interpreting The Heatmap
The first thing to note is that only the numeric features are compared as it is obvious that we cannot correlate between alphabets or strings. 
Before understanding the plot, let us see what exactly correlation is.
POSITIVE CORRELATION: If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.
NEGATIVE CORRELATION: If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.
Now lets say that two features are highly or perfectly correlated, so the increase in one leads to increase in the other. 
This means that both the features are containing highly similar information and there is very little or no variance in information. 
This is known as MultiColinearity as both of them contains almost the same information.
So do you think we should use both of them as one of them is redundant. 
While making or training models, we should try to eliminate redundant features as it reduces training time and many such advantages.
Now from the above heatmap,we can see that the features are not much correlated. The highest correlation is between SibSp and Parch i.e 0.41. So we can carry on with all features.
'''

#todo : 2020.01.23 할차례

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
