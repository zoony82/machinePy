import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import Binarizer

from sklearn.metrics import precision_recall_curve

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
    df['Cabin'] = df['Cabin'].str[:1]
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


#전처리 전체 과정 처음부터 로딩 및 시작
titanic_df = pd.read_csv("/home/jjh/문서/dataset/titanic/train.csv")
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis = 1)
x_titanic_df = transform_features(x_titanic_df)
# 인코딩 된 Feature 확인
x_titanic_df['Cabin'].unique() #[7, 2, 4, 6, 3, 0, 1, 5, 8]
x_titanic_df['Sex'].unique() # [0,1]
x_titanic_df['Embarked'].unique() # [0,1,2,3]

#테스트 셋 추출
x_train, x_test, y_train, y_test = train_test_split(x_titanic_df,y_titanic_df,test_size=0.2, random_state=11)

#결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()


#Decision 학습/예측/평가
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
print(accuracy_score(y_test,dt_pred))
# 78.77

#RandomForest 학습/예측/평가
rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
print(accuracy_score(y_test,rf_pred))
# 83.24

#LogisticRecression 학습/예측/평가
lr_clf.fit(x_train, y_train)
lr_pred = lr_clf.predict(x_test)
print(accuracy_score(y_test,lr_pred))
# 86.59


#교차 검증으로 결정트리 모델을 좀 더 평가해보겠다.
def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 kfold객체를 생성, 폴드 수만큼 예측결과 저장을 위한 리스트 객체 생성
    kfold = KFold(n_splits=folds)
    scores = []

    #kfold 교차 검증 수행, enumerate => 인덱스와 함께 tuple형태로 반환
    for iter_count, (train_index, test_index) in enumerate(kfold.split(x_titanic_df)):
        # x_titanic_df 에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
        x_train, x_test = x_titanic_df.values[train_index], x_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        #Classifier 학습, 예측, 정확도 계산
        clf.fit(x_train,y_train)
        prediction = clf.predict(x_test)
        accuracy = accuracy_score(y_test,prediction)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도 : {1:.4f}".format(iter_count,accuracy))

    #5개 fold에서의 평균 정확도 계산
    mean_score = np.mean(scores)
    print("평균 정확도 :{0:.4f}".format(mean_score)) # 0.7823


#exec kfold 호출
exec_kfold(dt_clf,folds=5)

#이번에는 corss_val_sore 를 통해 교차 검증
scores = cross_val_score(dt_clf, x_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도 : {1:.4f}".format(iter_count, accuracy))
print("평균 정확도 :{0:.4f}".format(np.mean(scores))) # 0.7835
# 평균 정확도가 차이나는 이유는 cross 함수의 경우 StratifiedKFold를 사용하기 때문

# DT의 최적 하이퍼 파라메터를 찾고, 예측 성능을 측정해보자.
parameters = {
    'max_depth':[2,3,5,10],
    'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]
}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(x_train,y_train)

print('최적 하이퍼 파라메터:', grid_dclf.best_params_)
print('최고 정확도:{0:.4f}'.format(grid_dclf.best_score_))

best_dclf=grid_dclf.best_estimator_

#gridSearchCv 의 최적 하이퍼 파라메터로 학습된 Estimator로 예측 및 평가 수행
dpredictions = best_dclf.predict(x_test)
accuracy = accuracy_score(y_test,dpredictions)
print('테스트 셋트에서 dt 정확도 :{0:.4f}'.format(accuracy)) # 0.8715

# 일반적으로 하이퍼 파라메터를 튜닝해도 이렇게 향상되긴 힘들다.
# 테스트용 데이터라서 작기 때문에, 너무 많이 향상된것으로 판단 된다.

#평가(정확도/정밀도/재현율)
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred) # 정확도 : tn+tp / tn+fp+fn+tp
    precision = precision_score(y_test,pred) # 정밀도 : tp / fp+tp -> 예측 성능을 더욱 정밀하게 측정
    recall = recall_score(y_test,pred) # 재현율 : tp / tp+fn -> 실제 정답이 Negative 로 판별된것이 많은지 확인
    print('오차행렬')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}'.format(accuracy,precision,recall))


#로지스틱 회귀 기반 예측한것으로 평가하기
get_clf_eval(y_test,lr_pred)
'''
오차행렬
[[108  10]
 [ 14  47]]
정확도 : 0.8659, 정밀도 : 0.8246, 재현율 : 0.7705
=> 정밀도에 비해 재현율이 낮음 -> 트레이드 오프
'''



#LogisticRecression 결과를 확률로 나오게 변경 : .predict => .predict_proba (분류 항목별 확률을 리턴)
lr_pred_proba = lr_clf.predict_proba(x_test) # 이진 분류일 경우 (0에대한 확률, 1에대한확률)

#예측 확률과 예측값 한번에 표시
pred_proba_result = np.concatenate([lr_pred_proba, lr_pred.reshape(-1,1)],axis=1)
print(pred_proba_result)

# 확률을 판단하는 threshold값을 조정해보자.
custom_threshold = 0.4
lr_pred_proba_1 = lr_pred_proba[:,-1].reshape(-1,1)
binarizer = Binarizer(threshold=custom_threshold).fit(lr_pred_proba_1)
lr_custom_predict = binarizer.transform(lr_pred_proba_1)

# 재현율은 올라가고, 정밀도가 떨어짐
# 분류 결정 임계값은 Positive 예측값의 확률을 봄
get_clf_eval(y_test,lr_custom_predict)
'''
오차행렬
[[97 21]
 [11 50]]
정확도 : 0.8212, 정밀도 : 0.7042, 재현율 : 0.8197
'''

# 임계값을 0.4~0.6까지 0.05씩 증가시키며 평가지표를 조사
thresholds = [0.4,0.45,0.5,0.55,0.6]
def get_eval_by_threshold(y_test, pred_proba_c1, thersholds):
    for t in thersholds:
        binarizer = Binarizer(t).fit(pred_proba_c1)
        predict = binarizer.transform(pred_proba_c1)
        print('임계값:',t)
        get_clf_eval(y_test,predict)

get_eval_by_threshold(y_test,lr_pred_proba[:,-1].reshape(-1,1),thresholds)
# 0.45 임계치가 0.5 대비 정확도는 비슷하고 정밀도가 약간 떨어지고, 재현율이 크게 상승=> 0.45를 선택

# 이와 유사한 사이킷런 api 테스트

# 레이블 값이 1일때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(x_test)[:,1]

# 실제값 데이터 세트와 레이블 값이 1일때의 예측 확률을 인자로 입력
precisions, recalls, thresholds = precision_recall_curve(y_test,pred_proba_class1)
print('반환된 분류 결정 임곗값 배열의 shape : ', thresholds.shape )

# 샘플 10건만 추출, 임계값은 15step로 추출
thr_idx = np.arange(0, thresholds.shape[0], 15)
print('샘플용 임계값 10개 :', np.round(thresholds[thr_idx],3))
print('샘플용 정밀도 10개 :', np.round(precisions[thr_idx],3))
print('샘플용 재현율 10개 :', np.round(recalls[thr_idx],3))

# 시각화
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def precision_recall_curve_plot(y_test, pred_proba_c1):
    #x축을 임계치로, y축을 정밀도(점선),재현율로
    plt.figure(figsize=(8,6))
    thre_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:thre_boundary],linestyle='--',label='precision')
    plt.plot(thresholds, recalls[0:thre_boundary], label='recall')

    # threshold 값 x축의 scale를 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))

    # x축 y축 label과 lebend 그리고 grid 설정
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall')
    plt.legend()
    plt.grid()
    plt.show()

precision_recall_curve_plot(y_test, lr_clf.predict_proba(x_test)[:,1])

#f1 score : 정밀도 재현율이 한쪽으로 치우침을 판단하는 지표
from sklearn.metrics import f1_score
f1 = f1_score(y_test, lr_pred)
print(f1)

#roc 그리기
from sklearn.metrics import roc_curve
#레이블 값이 1일 때의 예측 확률을 추출
pred_proba_class1 = lr_clf.predict_proba(x_test)[:,1]
# fprs : 1-특이도 = fp/fp+tn => fp:0인 케이스에 대해 1로 잘못 예측한 비율
# tprs : 민감도(재현율) = tp/tp+fn => tp:1인 케이스에 대해 1로 예측한 비율
# 임계값
fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)

def roc_curve_plot(y_test, pred_proba_c1):
    #임계값에 따른 fpr,ptr 값
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_class1)
    #roc 곡선 그리기
    plt.plot(fprs, tprs, label='ROC')
    #가운데 대각선 직선
    plt.plot([0,1],[0,1], 'k--', label='RanDom')

roc_curve_plot(y_test,pred_proba_class1)

# roc 곡선 자체는 fpr/tpr의 변화 추이를 보는데 이용함
# 성능 지표로 활용된느 것은 roc 곡선 아래 면적인 auc(Area Under Curve)
# 일반적으로 1에 가까울 수록 좋은 수치

from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test,lr_clf.predict(x_test))
print(auc_score)