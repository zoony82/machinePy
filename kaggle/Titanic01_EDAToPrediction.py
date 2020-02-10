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
data = pd.read_csv('/home/jjh/문서/dataset/titanic/train.csv')
# data = pd.read_csv('C:\\04_dataset\\titanic\\train.csv')
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

pd.crosstab(data.Pclass, data.Survived, margins=True)#.style.background_gradient(cmap='summer_r')

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
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
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


# Part2. Feature Engineering and Data Cleaning
'''
Now what is Feature Engineering?
Whenever we are given a dataset with features, it is not necessary that all the features will be important. 
There maybe be many redundant features which should be eliminated. 
Also we can get or add new features by observing or extracting information from other features.
An example would be getting the Initials feature using the Name Feature. 
Lets see if we can get any new features and eliminate a few. 
Also we will transform the existing relevant features to suitable form for Predictive Modeling.
'''

'''
Age_band
Problem With Age Feature:
As I have mentioned earlier that Age is a continous feature, there is a problem with Continous Variables in Machine Learning Models.
Eg:If I say to group or arrange Sports Person by Sex, We can easily segregate them by Male and Female.
Now if I say to group them by their Age, then how would you do it? 
If there are 30 Persons, there may be 30 age values. Now this is problematic.
We need to convert these continous values into categorical values by either Binning or Normalisation. 
I will be using binning i.e group a range of ages into a single bin or assign them a single value.
Okay so the maximum age of a passenger was 80. So lets divide the range from 0-80 into 5 bins. So 80/5=16. So bins of size 16.
'''

data['Age']
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4

data['Age_band'].value_counts().to_frame()

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
'''
Clearly, as the Fare_cat increases, the survival chances increases. 
This feature may become an important feature during modeling along with the Sex.
'''


'''
Converting String Values into Numeric
Since we cannot pass strings to a machine learning model, 
we need to convert features loke Sex, Embarked, etc into numeric values.
'''

data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

'''
Dropping UnNeeded Features
Name--> We don't need name feature as it cannot be converted into any categorical value.
Age--> We have the Age_band feature, so no need of this.
Ticket--> It is any random string that cannot be categorised.
Fare--> We have the Fare_cat feature, so unneeded
Cabin--> A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.
Fare_Range--> We have the fare_cat feature.
PassengerId--> Cannot be categorised.
'''

data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
'''
Now the above correlation plot, we can see some positively related features. 
Some of them being SibSp andd Family_Size and Parch and Family_Size and some negative ones like Alone and Family_Size.
'''

data.to_csv('/home/jjh/문서/dataset/titanic/data_clean.csv',index=False)

############################
# Part3: Predictive Modeling
############################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_clean = pd.read_csv('/home/jjh/문서/dataset/titanic/data_clean.csv')

'''
We have gained some insights from the EDA part. 
But with that, we cannot accurately predict or tell whether a passenger will survive or die. 
So now we will predict the whether the Passenger will survive or not using some great Classification Algorithms.
Following are the algorithms I will use to make the model:
1)Logistic Regression
2)Support Vector Machines(Linear and radial)
3)Random Forest
4)K-Nearest Neighbours
5)Naive Bayes
6)Decision Tree
7)Logistic Regression
'''

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

train, test = train_test_split(data_clean, test_size=0.3, random_state=0, stratify=data_clean['Survived'])
# stratify : 지정한 데이터의 비율 유지
train_x=train[train.columns[1:]]
train_y=train[train.columns[:1]]
test_x=test[test.columns[1:]]
test_y=test[test.columns[:1]]
all_x=data_clean[data_clean.columns[1:]]
all_y=data_clean[data_clean.columns[:1]]

#Radial Support Vector Machines : rbf-SVM
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_x,train_y)
pred1 = model.predict(test_x)
print(metrics.accuracy_score(pred1,test_y))
#0.835

#Linear Support Vector Machines : linear-SVM
model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
model.fit(train_x,train_y)
pred2 = model.predict(test_x)
print(metrics.accuracy_score(pred2,test_y))
#0.817

#Logistic Regression
model = LogisticRegression()
model.fit(train_x,train_y)
pred3 = model.predict(test_x)
print(metrics.accuracy_score(pred3,test_y))
#0.817

#Decision Tree
model = DecisionTreeClassifier()
model.fit(train_x,train_y)
pred4 = model.predict(test_x)
print(metrics.accuracy_score(pred4,test_y))
#0.802

#K-Nearest Neighbors(KNN)
model = KNeighborsClassifier()
model.fit(train_x,train_y)
pred5 = model.predict(test_x)
print(metrics.accuracy_score(pred5,test_y))
#0.832
'''
Now the accuracy for the KNN model changes as we change the values for n_neighbours attribute. 
The default value is 5. Lets check the accuracies over various values of n_neighbours.
'''
a_index = list(range(1,11))
a = pd.Series()
xt = [0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_x,train_y)
    pred = model.predict(test_x)
    a = a.append(pd.Series(metrics.accuracy_score(pred,test_y)))

plt.plot(a_index,a)
plt.xticks(xt)
fig.plt.gcf()
fig.set_size_inches(12,6)
plt.show()

#Gussian Naive Bayes
model = GaussianNB()
model.fit(train_x,train_y)
pred6 = model.predict(test_x)
print(metrics.accuracy_score(pred6,test_y))
#0.813


#Random Forests
model = RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y)
pred7 = model.predict(test_x)
print(metrics.accuracy_score(pred7,test_y))
#0.820

'''
The accuracy of a model is not the only factor that determines the robustness of the classifier. 
Let's say that a classifier is trained over a training data and tested over the test data and it scores an accuracy of 90%.
Now this seems to be very good accuracy for a classifier, but can we confirm that it will be 90% for all the new test sets that come over??. 
The answer is No, because we can't determine which all instances will the classifier will use to train itself. 
As the training and testing data changes, the accuracy will also change. 
It may increase or decrease. This is known as model variance.

To overcome this and get a generalized model,we use Cross Validation.
'''


'''
Cross Validation
Many a times, the data is imbalanced, i.e there may be a high number of class1 instances but less number of other class instances. 
Thus we should train and test our algorithm on each and every instance of the dataset. 
Then we can take an average of all the noted accuracies over the dataset.

1)The K-Fold Cross Validation works by first dividing the dataset into k-subsets.
2)Let's say we divide the dataset into (k=5) parts. We reserve 1 part for testing and train the algorithm over the 4 parts.
3)We continue the process by changing the testing part in each iteration and training the algorithm over the other parts. 
The accuracies and errors are then averaged to get a average accuracy of the algorithm.
This is called K-Fold Cross Validation.
4)An algorithm may underfit over a dataset for some training data and sometimes also overfit the data for other training set. 
Thus with cross-validation, we can achieve a generalised model.
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
kfold = KFold(n_splits=10, random_state=22)
xyz=[]
accuracy=[]
std=[]
classifiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, all_x, all_y, cv=kfold, scoring="accuracy")
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)

new_models_df2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)
new_models_df2
'''
                      CV Mean       Std
Linear Svm           0.793471  0.047797
Radial Svm           0.828290  0.034427
Logistic Regression  0.805843  0.021861
KNN                  0.813783  0.041210
Decision Tree        0.808115  0.029225
Naive Bayes          0.801386  0.028999
Random Forest        0.808115  0.030077
'''

plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()

new_models_df2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()

'''
The classification accuracy can be sometimes misleading due to imbalance. 
We can get a summarized result with the help of confusion matrix, which shows where did the model go wrong, 
or which class did the model predict wrong.
'''

f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred=cross_val_predict(svm.SVC(kernel='rbf'), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[0, 0], annot=True, fmt='2.0f')
ax[0,0].set_title('Matrix for rfb-svm')
y_pred = cross_val_predict(svm.SVC(kernel='linear'), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[0, 1], annot=True, fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[0, 2], annot=True, fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[1, 0], annot=True, fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[1, 1], annot=True, fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[1, 2], annot=True, fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(), all_x, all_y, cv=10)
sns.heatmap(confusion_matrix(all_y, y_pred), ax=ax[2, 0], annot=True, fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()


'''
Interpreting Confusion Matrix
The left diagonal shows the number of correct predictions made for each class while the right diagonal shows the number of wrong prredictions made. 
Lets consider the first plot for rbf-SVM:
1)The no. of correct predictions are 491(for dead) + 247(for survived) with the mean CV accuracy being (491+247)/891 = 82.8% 
which we did get earlier.
2)Errors--> Wrongly Classified 58 dead people as survived and 95 survived as dead. 
Thus it has made more mistakes by predicting dead as survived.
By looking at all the matrices, we can say that rbf-SVM has a higher chance in correctly predicting dead passengers 
but NaiveBayes has a higher chance in correctly predicting passengers who survived.
'''

'''
Hyper-Parameters Tuning
The machine learning models are like a Black-Box. 
There are some default parameter values for this Black-Box, which we can tune or change to get a better model. 
Like the C and gamma in the SVM model and similarly different parameters for different classifiers, are called the hyper-parameters, 
which we can tune to change the learning rate of the algorithm and get a better model. 
This is known as Hyper-Parameter Tuning.
We will tune the hyper-parameters for the 2 best classifiers i.e the SVM and RandomForests.
'''

#SVM
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':gamma}
gd=GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
gd.fit(all_x, all_y)
print(gd.best_score_)
print(gd.best_estimator_)
'''
0.8282828282828283
SVC(C=0.5, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
'''

#Random Forest
n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(all_x, all_y)
print(gd.best_score_)
print(gd.best_estimator_)
'''
0.8170594837261503
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=900,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
'''
#The best score for Rbf-Svm is 82.82% with C=0.05 and gamma=0.1. For RandomForest, score is abt 81.8% with n_estimators=900.


'''
Ensembling
Ensembling is a good way to increase the accuracy or performance of a model. 
In simple words, it is the combination of various simple models to create a single powerful model.

Lets say we want to buy a phone and ask many people about it based on various parameters. 
So then we can make a strong judgement about a single product after analysing all different parameters. 
This is Ensembling, which improves the stability of the model. 
Ensembling can be done in ways like:

1)Voting Classifier
2)Bagging
3)Boosting
'''

'''
Voting Classifier
It is the simplest way of combining predictions from many different simple machine learning models. 
It gives an average prediction result based on the prediction of all the submodels. 
The submodels or the basemodels are all of diiferent types.
'''

from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                                ('RBF', svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),
                                                ('RFor', RandomForestClassifier(n_estimators=500, random_state=0)),
                                                ('LR', LogisticRegression(C=0.05)),
                                                ('DT', DecisionTreeClassifier(random_state=0)),
                                                ('NB', GaussianNB()),
                                                ('svm', svm.SVC(kernel='linear', probability=True))
                                                ],voting='soft').fit(train_x,train_y)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_x,test_y))
#0.824
cross = cross_val_score(ensemble_lin_rbf, all_x, all_y, cv=10, scoring='accuracy')
print('The corss validation score is:',cross.mean())
#0.823


'''
Bagging
Bagging is a general ensemble method. 
It works by applying similar classifiers on small partitions of the dataset and then taking the average of all the predictions. 
Due to the averaging,there is reduction in variance. 
Unlike Voting Classifier, Bagging makes use of similar classifiers.
'''

'''
Bagged KNN
Bagging works best with models with high variance. 
An example for this can be Decision Tree or Random Forests. 
We can use KNN with small value of n_neighbours, as small value of n_neighbours.
'''

from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_x,train_y)
prediction = model.predict(test_x)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_y))
#0.835
result=cross_val_score(model, all_x, all_y, cv=10, scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
#0.814

# Bagged DecisionTree
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_x,train_y)
prediction=model.predict(test_x)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_y))
#0.824
result=cross_val_score(model, all_x, all_y, cv=10, scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())
#0.820

'''
Boosting
Boosting is an ensembling technique which uses sequential learning of classifiers. 
It is a step by step enhancement of a weak model.
Boosting works as follows:
A model is first trained on the complete dataset. 
Now the model will get some instances right while some wrong. 
Now in the next iteration, the learner will focus more on the wrongly predicted instances or give more weight to it. 
Thus it will try to predict the wrong instance correctly. 
Now this iterative process continous, and new classifers are added to the model until the limit is reached on the accuracy.
'''

'''
AdaBoost(Adaptive Boosting)
The weak learner or estimator in this case is a Decsion Tree. 
But we can change the dafault base_estimator to any algorithm of our choice.
'''
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,all_x,all_y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
# 0.824

'''
Stochastic Gradient Boosting
Here too the weak learner is a Decision Tree.
'''
from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,all_x,all_y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())
#0.818

#XGBoost
import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,all_x,all_y,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())

#Hyper-Parameter Tuning for AdaBoost
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(all_x,all_y)
print(gd.best_score_)
print(gd.best_estimator_)
'''
0.8316498316498316
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.05,
                   n_estimators=200, random_state=None)
'''


#Confusion Matrix for the Best Model
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)
result=cross_val_predict(ada,all_x,all_y,cv=10)
sns.heatmap(confusion_matrix(all_y,result),cmap='winter',annot=True,fmt='2.0f')
plt.show()

#Feature Importance
f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(all_x, all_y)
pd.Series(model.feature_importances_, all_x.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0, 0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(all_x, all_y)
pd.Series(model.feature_importances_, all_x.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0, 1], color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(all_x, all_y)
pd.Series(model.feature_importances_, all_x.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1, 0], cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(all_x, all_y)
pd.Series(model.feature_importances_, all_x.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[1, 1], color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()

'''
Observations:
1)Some of the common important features are Initial,Fare_cat,Pclass,Family_Size.
2)The Sex feature doesn't seem to give any importance, which is shocking as we had seen earlier that Sex combined with Pclass was giving a very good differentiating factor. 
Sex looks to be important only in RandomForests.
However, we can see the feature Initial, which is at the top in many classifiers.
We had already seen the positive correlation between Sex and Initial, so they both refer to the gender.
3)Similarly the Pclass and Fare_cat refer to the status of the passengers and Family_Size with Alone,Parch and SibSp.
'''
