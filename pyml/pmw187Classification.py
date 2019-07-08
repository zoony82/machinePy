from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(random_state=156)

#Load Iris Data
iris_data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

#Decision Tree 학습
dt_clf.fit(x_train,y_train)


# 우분투에서 graphviz 설치 못함
# export_graphviz 의 호출 결과로 out_file로 지정된 tree.dot파일을 생성함
# from sklearn.tree import export_graphviz
# export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)
#
# # 위에서 생성된 출력파일 tree.dot 파일을 grahpviz가 읽어서 주피터 노트북상에 시각화
# import graphviz

#결정트리의 속성을 가져와 피쳐별로 중요도값 매핑 해보자.
import seaborn as sns
import numpy as np

#feature importance 추출
print('Feature importances:\n{0}'.format(np.round(dt_clf.feature_importances_, 3)))

#feature 매핑
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print('{0} : {1:.3f}'.format(name,value))

sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)

## 과적합 테스트
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

plt.title('3 class values with 2 features sample')

#2차원 시각화를 위해서 피처는 2개, 클래스는 3가지 유형의 분류 샘플 데이터 생성
x_features, y_lables = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1, random_state=0)
#그래프 형태로 2개의 피처로 2차원 좌표 시각화, 각 클래스 값은 다른 색깔로 표시됨
plt.scatter(x_features[:,0], x_features[:,1], marker='o',c=y_lables, s=25, edgecolors='k')


# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig, ax = plt.subplots()

    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start, xlim_end = ax.get_xlim()
    ylim_start, ylim_end = ax.get_ylim()

    # 호출 파라미터로 들어온 training 데이타로 model 학습 .
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행.
    xx, yy = np.meshgrid(np.linspace(xlim_start, xlim_end, num=200), np.linspace(ylim_start, ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # contourf() 를 이용하여 class boundary 를 visualization 수행.
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)

#특정한 트리 생성 제약 없는 결정 트리의 학습과 결정 경계 시각화
dt_clf=DecisionTreeClassifier().fit(x_features,y_lables)
visualize_boundary(dt_clf,x_features,y_lables)
#=> 일부 이상치까지 분류하기 위해 분할이 자주 일어남(결정 기준 경계가 매우 많아짐)

# 리프노드 생성 규칙을 완화
dt_clf=DecisionTreeClassifier(min_samples_leaf=6).fit(x_features,y_lables)
visualize_boundary(dt_clf,x_features,y_lables)
#=> 이상치에 크게 반응하지 않으면서 좀 더 일반화된 분류규칙에 따라 분류됨
