import pandas as pd
import matplotlib.pyplot as plt

#定义清洗Z不合格的数据的函数
def clean_GC(GC_value):
    if GC_value < 0.4 or GC_value > 0.6:
        return 0
    else:
        return GC_value

#定义处理孕期数据的函数
def processed_value(origin_value):
    if pd.isna(origin_value):
        return None               #增强函数的健壮性

    #转换为字符串并处理
    str_value = str(origin_value)

    # 检查是否包含'w'
    if "w+" in str_value:
        # 按'w'分割字符串
        parts = str_value.split("w+")

        # 确保分割后有两部分且都是数字
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            main_num = int(parts[0])  # 两位数部分
            digit = int(parts[1])  # 个位数部分

            #周数乘7加上后续天数即为总天数
            return ( main_num * 7 + digit )

    # 如果没有后续数字，返回周数乘7
    elif 'w' in str_value:
        # 找到'w'的位置
        w_index = str_value.find('w')

        # 提取'w'前面的部分
        before_w = str_value[:w_index]
        return int(before_w)*7

    else:
        return None


#从Excel文件读取数据
df = pd.read_excel('男胎检测数据_清洗后.xlsx', sheet_name='Sheet1')


#将不合格的Z含量清除为0，填充到新一列中
df['清洗GC'] = df['GC含量'].apply(lambda x: clean_GC(x))

#清洗数据
df = df[df['清洗GC'] != 0].reset_index(drop=True)

# 删除过渡列
df = df.drop('清洗GC', axis=1)



#标准化孕期数据
df['标准化孕期'] = df['检测孕周'].apply(lambda x: processed_value(x))

# 生成输出文件
output_file = '男胎检测数据_清洗后.xlsx'
# 保存到新的Excel文件
df.to_excel(output_file, index=False)