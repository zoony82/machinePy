import pandas as pd

data_frame = pd.read_csv('C:\\04_dataset\\test.csv')
data_frame.head()

type(data_frame.Pclass)

# 컬럼은 하나의 시리즈로 불린다
# 시리즈는 파이썬 리스트로 만든다
s1 = pd.core.series.Series([1,2,3])
s2 = pd.core.series.Series(['one','two','three'])
pd.DataFrame(data=dict(num=s1,word=s2))

# 다른 방식으로 만들어보자
friend_dict =[
    {'name':'jang','age':20,'job':'stud'},
    {'name':'kim','age':30,'job':'jjj'},
    {'name':'lee','age':40,'job':'stud'}
]

type(friend_dict)

df = pd.DataFrame(friend_dict)
df.head()

# 컬럼 순서를 바꿔보자.
df = df[['name','age']]
df.head()

# 필터
# 행 선택
df[1:3] # 기본적으로 첫번째 항목은 포함, 두번째 항목은 미포함

# 특정 위치의 열만 가져오고 싶을때
df.loc[[0,2]]

# 조건절
df[df.age > 30]
df.query('age>30')
df[(df.age>20) & (df.job=='stud')]

#컬럼 필터
df.iloc[:] # 모든 행/컬럼을 원할때
df.iloc[:,0:2] # 모든 행과 0,1 컬럼만
df.iloc[0:2,0:2]
df[['name']]
df.filter(items=['name'])

#컬럼에 대한 조건
df.filter(like='a', axis=1)

# 열 삭제
df_temp = df.drop([1])
df_temp = df.drop(df.index[[1]])
df_temp = df[df.age>20]

# 컬럼 삭제
df_temp = df.drop('age',axis=1)

# 컬럼 생성
df['sal'] = 0

# 컬럼 조작
import numpy as np
df['sal'] = np.where(df['job'] != 'stud', 'yes','no')
df

dict2 =[
    {'name':'jang','mid':50,'final':40},
    {'name':'kim','mid':60,'final':100},
    {'name':'lee','mid':100,'final':70}
]

df = pd.DataFrame(dict2)

df['total'] = df['mid'] + df['final']
df['avg'] = df['total'] / 2
df

grades =[]
for row in df['avg']:
    if row>50:
        grades.append('a')
    else:
        grades.append('b')

df['grades'] = grades

# 각 row별로 함수 적용하기
def fn_pass(row):
    if row == 'a':
        return 'pass'
    else :
        return 'fail'


df.grades = df.grades.apply(fn_pass)

#Feature Extraction
date_list =[
    {
        'yyyy-mm-dd' : '2000-09-09'
    },
{
        'yyyy-mm-dd' : '2010-10-09'
    }
]

df = pd.DataFrame(date_list,columns=['yyyy-mm-dd'])
df
# 연도만 빼고 싶다면
def extract_year(row):
    return row.split('-')[0]

df['year'] = df['yyyy-mm-dd'].apply(extract_year)
df

#행을 추가하기
dict2 =[
    {'name':'jang','mid':50,'final':40},
    {'name':'kim','mid':60,'final':100},
    {'name':'lee','mid':100,'final':70}
]

df = pd.DataFrame(dict2)

df2 = pd.DataFrame([
    ['ben',66,99],
    ],columns=['name','mid','final']
)

df.append(df2)

df.append(df2,ignore_index=True)


 #todo : 데이터 그룹 만들기 할 차례