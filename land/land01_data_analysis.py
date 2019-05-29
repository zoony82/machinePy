import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
# from plotnine import *


#C:\04_land

pre_sale = pd.read_csv('C:\\04_land\\전국_평균_분양가격_2019.2월_.csv',encoding='euc-kr')
pre_sale.shape
pre_sale.head()
pre_sale.tail()
pre_sale.info()
pre_sale.dtypes

# 각 컬럼별 결측치 현황
pre_sale.isnull().sum()

# 데이터 타입 변경
pre_sale['연도'] = pre_sale['연도'].astype(str)
pre_sale['월'] = pre_sale['월'].astype(str)

pre_sale_price = pre_sale['분양가격(㎡)']
pre_sale['분양가격'] = pd.to_numeric(pre_sale_price, errors='coerce')

#평당 분양가격
pre_sale['평당분양가격'] = pre_sale['분양가격'] * 3.3

pre_sale.info()
# 공백이 널값으로 들어가면서 결측치가 더 많아짐
pre_sale.isnull().sum()
pre_sale.describe()
pre_sale.describe(include=[np.object])
pre_sale.describe(include='all')

pre_sale_2017 = pre_sale.loc[pre_sale['연도'] == '2017']
pre_sale_2017.head()

#집계
pre_sale['규모구분'].value_counts()
pre_sale['지역명'].value_counts()

