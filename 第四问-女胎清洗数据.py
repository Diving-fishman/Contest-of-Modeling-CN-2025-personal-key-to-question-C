import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

female_data = pd.read_excel('女胎检测数据.xlsx')

female_data = female_data[
    (female_data['原始读段数'] >= 3000000) &
    (female_data['在参考基因组上比对的比例'] >= 0.75) &
    (female_data['重复读段的比例'] <= 0.15) &
    (female_data['GC含量'] >= 0.4) &
    (female_data['GC含量'] <= 0.6) &

    (female_data['被过滤掉读段数的比例'] <= 0.1) &
    (female_data['X染色体的Z值'] >= -3) &
    (female_data['X染色体的Z值'] <= 3) &
    (female_data['13号染色体的Z值'] >= -3) &
    (female_data['13号染色体的Z值'] <= 3) &
    (female_data['18号染色体的Z值'] >= -3) &
    (female_data['18号染色体的Z值'] <= 3) &
    (female_data['21号染色体的Z值'] >= -3) &
    (female_data['21号染色体的Z值'] <= 3)
]

female_data['染色体的非整倍体'] = female_data['染色体的非整倍体'].apply(
    lambda x: '无' if (pd.isna(x) or str(x).strip() == '') else x
)

"""#数据标准化
scaler = StandardScaler()
female_data_scaled = scaler.fit_transform(female_data)"""

female_data.to_excel('整理后的数据——女.xlsx', index=False)

"""(female_data['13号染色体的GC含量'] >= 0.4) &
    (female_data['13号染色体的GC含量'] <= 0.6) &
    (female_data['18号染色体的GC含量'] >= 0.4) &
    (female_data['18号染色体的GC含量'] <= 0.6) &
    (female_data['21号染色体的GC含量'] >= 0.4) &
    (female_data['21号染色体的GC含量'] <= 0.6) &"""

