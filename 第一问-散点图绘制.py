import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns    #导入散点图所用的库

#从Excel文件读取数据
df = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')

#设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#创建散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Y染色体浓度', y='孕妇BMI')
plt.title('相关关系')
plt.xlabel('Y染色体浓度')
plt.ylabel('孕妇BMI')

#打印散点图
plt.show()