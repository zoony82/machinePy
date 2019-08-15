import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re


pre_sale = pd.read_csv('C:\\04_land\\전국_평균_분양가격_2019.2월_.csv',encoding='euc-kr')
#pd.options.display.float_format = '{:,.0f}'.format

pre_sale.dtypes

pre_sale_price = pre_sale['분양가격(㎡)']
pre_sale['분양가격'] = pd.to_numeric(pre_sale_price,errors='coerce')

pre_sale.groupby(pre_sale.연도).describe()
pre_sale.groupby(pre_sale.연도).describe().T

pre_sale.pivot_table('분양가격','규모구분','연도')

pre_sale_all = pre_sale.loc[pre_sale['규모구분']=='전체']
pre_sale_all_reset_index = pre_sale_all.pivot_table('분양가격','지역명','연도').reset_index()
pre_sale_all_reset_index.head()

pre_sale_all_reset_index.dtypes
pre_sale_all_reset_index['변동액'] = (pre_sale_all_reset_index[2019] - pre_sale_all_reset_index[2015]) #.astype(int)
pre_sale_all_reset_index.head()
max_delta_price = np.max(pre_sale_all_reset_index['변동액'])
print(max_delta_price)

