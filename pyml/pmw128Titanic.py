import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# %matplotlib inline

titanic_df = pd.read_csv("/home/jjh/문서/dataset/titanic/train.csv")
titanic_df.head(3)
type(titanic_df)
titanic_df.shape
titanic_df.info()

#Series 객체만 반환
titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))

#null 값 처리
titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)
print('데이터 셋트 null 값 개수',titanic_df.isnull().sum().sum())

#문자열 핏쳐 데이터 분포도 확인
print(titanic_df['Sex'].value_counts())
print(titanic_df['Cabin'].value_counts())
print(titanic_df['Embarked'].value_counts())

#선실의 경우 첫번째 알파벳이 영향을 끼칠것으로 보임
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

#성별에 따른 생존자 수 비교
titanic_df.groupby(['Sex','Survived'])['Survived'].count()
sns.barplot(x='Sex', y='Survived', data=titanic_df)

#부자 가난한 사람간 생존 확률
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)

#연령별 생존 확률
def get_category(age):
    cat =''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'YoungAdult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    return  cat

# 막대그래프의 크기를 더 크게 설정
plt.figure(figsize=(10,6))
#x축의 값을 순차적으로 표시
groups_names = ['Unknown','Baby','Child','Teenager','Student','YoungAdult','Adult','Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda v:get_category(v))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=groups_names)
titanic_df.drop('Age_cat',axis=1, inplace=True)

#LabelEncoder 를 통해 0 ~ (카테고리유형수-1) 까지의 숫자 값으로 변환

def encode_features(dataDF):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le_fit = le.fit(dataDF[feature])
        dataDF[feature] = le_fit.transform(dataDF[feature])
    return dataDF

titanic_df=encode_features(titanic_df)
titanic_df.head()



#지금까지 만든것을 함수로 생성
#null 처리 함수
def fillna(df):
    df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

#불필요 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
    return df

#레이블 인코딩 수행
def format_features(df):
    df['Cabin'] = titanic_df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le_fit = le.fit(df[feature])
        df[feature] = le_fit.transform(df[feature])

    return df


#앞에서 설정한 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


#다시 처음부터 로딩 및 시작
titanic_df = pd.read_csv("/home/jjh/문서/dataset/titanic/train.csv")
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis = 1)
x_titanic_df = transform_features(x_titanic_df)
df = x_titanic_df

