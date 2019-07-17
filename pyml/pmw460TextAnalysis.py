from sklearn.datasets import fetch_20newsgroups

news_data = fetch_20newsgroups(subset='all', random_state=156)

type(news_data) #딕셔너리와 유사한 Bunch 객체 리턴
print(news_data.keys())


import pandas as pd

#target 정보 확인
print('target 클래스의 값과 분포도 \n', pd.Series(news_data.target).value_counts().sort_index())
print('target 클래스의 이름들\n', len(news_data.target_names), news_data.target_names)

#개별 데이터 확인
print(news_data.data[0])

#내용을 제외한 제목 등의 다른정보는 모두 삭제(only 텍스트문 만으로 분석을 진행하기 위해서) - remove함수
#학습/테스트 셋 분류
train_news = fetch_20newsgroups(subset='train',remove=('headers','footers','quotes'), random_state=156)
x_train = train_news.data
y_train = train_news.target
test_news = fetch_20newsgroups(subset='test',remove=('headers','footers','quotes'), random_state=156)
x_test = test_news.data
y_test  = test_news.target
print('학습 데이터 크기 {0}, 테스트 데이터 크기{1]',len(x_train),len(x_test))


#피처 벡터화 변환과 머신러닝 모델 학습/예측/평가
from sklearn.feature_extraction.text import CountVectorizer

#Count Vectorization으로 피처 벡터화 변환 수행
cnt_vect = CountVectorizer()
cnt_vect.fit(x_train,y_train)
x_train_cnt_vect = cnt_vect.transform(x_train)
type(x_train_cnt_vect) #CSR(Compressed Sparse Row) 형식의 Sparse Matrix
print('학습 데이터의 CountVect shape : ', x_train_cnt_vect.shape) # (11314개의 문서 101631개의 단어)

#학습 데이터로 fit된 CountVectorizer를 이용해 테스트 데이터를 피쳐 벡터화 변환 수행
x_test_cnt_vect = cnt_vect.transform(x_test)
print('테스트 데이터의 CountVect shape : ', x_test_cnt_vect.shape) # (7532개의 문서 101631개의 단어)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr_clf = LogisticRegression()
lr_clf.fit(x_train_cnt_vect,y_train)
pred = lr_clf.predict(x_test_cnt_vect)
print('예측 정확도 : ', accuracy_score(y_test,pred))

#tf-idf 기반 측정
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
x_train_tfidf_vect = tfidf_vect.transform(x_train)
x_test_tfidf_vect = tfidf_vect.transform(x_test)

lr_clf = LogisticRegression()
lr_clf.fit(x_train_tfidf_vect,y_train)
pred = lr_clf.predict(x_test_tfidf_vect)
print('예측 정확도 : ', accuracy_score(y_test,pred))

#Stop-word 적용, bigram 적용, max_df=300 적용
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300)
tfidf_vect.fit(x_train)
x_train_tfidf_vect = tfidf_vect.transform(x_train)
x_test_tfidf_vect = tfidf_vect.transform(x_test)

lr_clf = LogisticRegression()
lr_clf.fit(x_train_tfidf_vect,y_train)
pred = lr_clf.predict(x_test_tfidf_vect)
print('예측 정확도 : ', accuracy_score(y_test,pred))


#GridSearchCV를 활용한 파라미터 최적화
from sklearn.model_selection import GridSearchCV

#최적 C값 도출 튜닝 수행, CV는 3폴드 세트로 설정
params = {'C':[0.01, 0.1, 1,5,10]}
grid_cv_lr = GridSearchCV(lr_clf, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_lr.fit(x_train_tfidf_vect, y_train)
print('Best C parameter : ', grid_cv_lr.best_params_)

# 최적 C로 학습된 grid_cv로 예측 및 평가
pred = grid_cv_lr.predict(x_test_tfidf_vect)
print('Best accuracy : ', accuracy_score(y_test,pred))

#pipeline을 사용
from sklearn.pipeline import Pipeline
#TfidfVector 객체를 tfidf_vect로, LogisticResgression 객체를 lr_clf로 생성하는 pipeline생성
pipeline = Pipeline([
    ('tfidf_vect',TfidfVectorizer(stop_words='english',ngram_range=(1,2), max_df=300)),
    ('lr_clf',LogisticRegression(C=10))
])
pipeline.fit(x_train,y_train)
pred = pipeline.predict(x_test)
print('accuracy : ',accuracy_score(y_test,pred))