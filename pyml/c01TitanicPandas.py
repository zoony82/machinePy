import pandas as pd

titanic_df = pd.read_csv("/home/jjh/문서/dataset/titanic/train.csv")
titanic_df.head(3)
type(titanic_df)
titanic_df.shape
titanic_df.info()

#데이터 분포도 확인
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)

#Series 객체만 반환
titanic_pclass = titanic_df['Pclass']
print(type(titanic_pclass))

