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

# todo : 199 page