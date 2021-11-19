import pandas as pd
df = pd.read_csv('./model/moel_log/model_v2/performance.csv',converters={'precision':lambda x: x.strip("['']").strip("'', ").split(", ")[-1],
                                                                         'recall':lambda x: x.strip("['']").strip("'', ").split(", ")[-1],
                                                                         'f1_score':lambda x: x.strip("['']").strip("'', ").split(", ")[-1],
                                                                         'support' : lambda x: x.strip("['']").strip("'', ").split(", ")[-1]})
df.columns
df = df[['Unnamed: 0','precision','recall','f1_score','support']]
df.to_csv('model_performance.csv',encoding='utf-8-sig')
sum(df['precision'][:23].astype(float)*df['support'][:23].astype(float))/sum(df['support'][:23].astype(float))


data = pd.read_csv('./data/train_data_v10_0715.csv',converters={"division": lambda x: x.strip("['']").strip("'', ").split("', '")})
division = [d for subd in data['division'].tolist() for d in subd]
division = pd.DataFrame(division)
dd = division.value_counts()
division_list = [d[0] for d in dd.keys()]
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
x = np.arange(23)

plt.rcParams['font.sans-serif']=['Noto Sans CJK TC']   # 將字體設定為思源宋體
plt.rcParams['axes.unicode_minus']=False
cmap = cm.jet(np.linspace(0, 1, 23))
plt.barh(x, dd.values)
for a,b in zip(x,dd.values):
    plt.text(b+50, a-0.5 , '%.0f' % b, ha='center', va= 'bottom',fontsize=10)
plt.yticks(x, division_list)
plt.ylabel('Division')
plt.xlabel('Count')
plt.title('data distribution')
plt.show()