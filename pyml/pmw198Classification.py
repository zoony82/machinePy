#사용자 행동 인식 데이터 세트를 통한 결정트리 테스트
#https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

import pandas as pd
import matplotlib.pyplot as plt

# 피처 인덱스,이름 가져오기
# 1 tBodyAcc-mean()-X
# 2 tBodyAcc-mean()-Y
# 3 tBodyAcc-mean()-Z
feautre_name_df = pd.read_csv('/home/jjh/문서/dataset/human_activity/features.txt',sep='\s',header=None, names=['column_index','column_name'])

#피처명 index를 제거 하고, 피처명만 리스트 객체로 생성하고, 샘플로 10개만 추출
feautre_name = feautre_name_df.iloc[:,1].values.tolist()
feautre_name[:10]

def get_human_dataset():
    feature_name_df = pd.read_csv('/home/jjh/문서/dataset/human_activity/features.txt',sep='\s',header=None, names=['column_index','column_name'])
    # dataframe에 피처명을 컬럼으로 부여하기 위해 리스트 객체로 다시 변환
    feature_name = feautre_name_df.iloc[:,1].values.tolist()

    #피처 데이터 로드
    x_train = pd.read_csv('/home/jjh/문서/dataset/human_activity/train/X_train.txt',sep='\s+',names=feautre_name)
    x_test = pd.read_csv('/home/jjh/문서/dataset/human_activity/test/X_test.txt', sep='\s+', names=feautre_name)

    #레이블 데이터 로드
    y_train = pd.read_csv('/home/jjh/문서/dataset/human_activity/train/y_train.txt',sep='\s+',header=None, names=['action'])
    y_test = pd.read_csv('/home/jjh/문서/dataset/human_activity/test/y_test.txt', sep='\s+', header=None, names=['action'])

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_human_dataset()

x_train.info()
# 피처 데이터의 경우 7,352 레코드를 가지고 있고, 561개의 피처로 구성되어있다
# 피처가 전부 float 형이므로, 카테고리 인코딩은 할 필요 없다

y_train['action'].value_counts()
# 레이블값은 6개이며 골고루 분포되어 있다


# 결정트리 학습을 default로 진행
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(x_train,y_train)
pred = dt_clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
print(dt_clf.get_params())

#결정트리의 깊이가 예측 정확도에 미치는 영향(5개의 세트로 진행)
from sklearn.model_selection import GridSearchCV

params = {'max_depth':[6,8,10,12,16,20,24]}
grid_cv = GridSearchCV(dt_clf,param_grid=params, scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(x_train,y_train)
#최고 평균 정확도 수치
grid_cv.best_score_
#최적 하이퍼 파라메터
grid_cv.best_params_
# => max_detph가 8일때 평균 정확도가 85프로


# cv_results_ 속성을 통해 파라메터에 따른 평균 정확도 수치 파악 가능
cv_results_df = pd.DataFrame(grid_cv.cv_results_)

cv_results_df[['param_max_depth','mean_test_score']]#,'mean_train_score']]

# max_depth와 min_samples_split 같이 변경하면서 테스트
params = {'max_depth':[6,8,10,12,16,20,24], 'min_samples_split':[16,24]}
grid_cv = GridSearchCV(dt_clf,param_grid=params, scoring='accuracy', cv=5, verbose=1)
grid_cv.fit(x_train,y_train)
#최고 평균 정확도 수치
grid_cv.best_score_
#최적 하이퍼 파라메터
grid_cv.best_params_
# => max_detph가 8일때, min_samples_split가 16일때 평균 확도가 85프로

#위의 최적 파라메터로 학습이 완료된 Estimator 객체로 테스트 세트 예측 수행
best_df_clf = grid_cv.best_estimator_
pred1 = best_df_clf.predict(x_test)
accuracy = accuracy_score(y_test,pred1)
accuracy # 87.17


# 중요도가 높은 순으로 피처 표시
import seaborn as sns
ftr_importances_values = best_df_clf.feature_importances_
# top 중요도로 정렬을 쉽게 하고, 시본의 막대 그래프로 쉽게 표현하기 위해 시리즈로 변환
ftr_importances = pd.Series(ftr_importances_values, index=x_train.columns)
#중요도값으로 시리즈 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure(figsize=(8,6))
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()