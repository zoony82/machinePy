import pandas as pd
import matplotlib.pyplot as plt


web_stats = {'day': [1, 2, 3, 4, 5, 6, 7],
             'visitors': [43, 44, 45, 46, 47, 48, 58],
             'bounce_rate': [89, 67, 90, 73, 28, 89, 41]
             }
df = pd.DataFrame(web_stats)
print(df)
df.plot()
plt.xlabel("day")
plt.ylabel("visitors")

df.plot.bar()

pre_sale = pd.read_csv('C:\\04_land\\전국_평균_분양가격_2019.2월_.csv',encoding='euc-kr')
pre_sale.shape
pre_sale.head()

pre_sale["분양가격"] = pd.to_numeric(pre_sale["분양가격(㎡)"],errors='')
pre_sale.plot()
