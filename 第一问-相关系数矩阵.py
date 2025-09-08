import pandas as pd

#从Excel文件读取数据
df = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')

#读取需要计算的两组数据
column1 = 'Y染色体浓度'
column2 = input("请输入需要检测的数据：")

# 计算Pearson相关系数
correlation = df[column1].corr(df[column2], method='pearson')
print(f"{column1}和{column2}的Pearson相关系数为: {correlation:.4f}")