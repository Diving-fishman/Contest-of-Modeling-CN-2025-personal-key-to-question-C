import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据和模型
# 假设已有训练好的模型和标准化器
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# 加载数据
file_path = '整理后的数据——女.xlsx'
df = pd.read_excel(file_path)

# 2. 数据预处理
target_column = '染色体的非整倍体'
feature_columns = [
    '孕妇BMI', '原始读段数', 'GC含量', '13号染色体的Z值',
    '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'
]

# 处理缺失值
for col in feature_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# 将目标变量转换为数值型
if df[target_column].dtype == 'object':
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

# 准备特征和目标变量
X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化特征
X_test_scaled = scaler.transform(X_test)

# 3. 计算预测概率
y_pred_proba = model.predict_proba(X_test_scaled)

# 4. 定义异常风险分数
# 计算每个样本的异常风险分数（所有异常类别的概率之和）
normal_class_index = np.where(model.classes_ == list(le.classes_).index('无'))[0][0] if 'le' in locals() else 5
abnormal_risk_scores = 1 - y_pred_proba[:, normal_class_index]

# 5. 选择高灵敏度的阈值
# 将问题转化为二分类：正常 vs 异常
y_binary = (y_test != normal_class_index).astype(int)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_binary, abnormal_risk_scores)
roc_auc = auc(fpr, tpr)

# 寻找高灵敏度的阈值（灵敏度 > 0.95）
high_sensitivity_thresholds = []
for i, threshold in enumerate(thresholds):
    if tpr[i] >= 0.95:  # 灵敏度 ≥ 95%
        high_sensitivity_thresholds.append((threshold, tpr[i], fpr[i]))

# 选择特异性最高的阈值（假阳性率最低）
if high_sensitivity_thresholds:
    # 按假阳性率排序，选择最低的
    high_sensitivity_thresholds.sort(key=lambda x: x[2])
    selected_threshold = high_sensitivity_thresholds[0][0]
    selected_sensitivity = high_sensitivity_thresholds[0][1]
    selected_specificity = 1 - high_sensitivity_thresholds[0][2]
else:
    # 如果没有达到95%灵敏度的阈值，选择最接近的
    closest_idx = np.argmin(np.abs(tpr - 0.95))
    selected_threshold = thresholds[closest_idx]
    selected_sensitivity = tpr[closest_idx]
    selected_specificity = 1 - fpr[closest_idx]

print(f"选择的阈值: {selected_threshold:.4f}")
print(f"对应的灵敏度: {selected_sensitivity:.4f}")
print(f"对应的特异性: {selected_specificity:.4f}")

# 6. 定义风险分层阈值
# 使用百分位数定义中风险和高风险的阈值
low_medium_threshold = np.percentile(abnormal_risk_scores, 70)  # 低风险与中风险的分界
medium_high_threshold = np.percentile(abnormal_risk_scores, 90)  # 中风险与高风险的分界

# 确保高风险阈值至少高于选择的灵敏度阈值
medium_high_threshold = max(medium_high_threshold, selected_threshold)

print(f"\n风险分层阈值:")
print(f"低风险与中风险分界: {low_medium_threshold:.4f}")
print(f"中风险与高风险分界: {medium_high_threshold:.4f}")

# 7. 进行风险分层
risk_levels = []
for score in abnormal_risk_scores:
    if score < low_medium_threshold:
        risk_levels.append("低风险")
    elif score < medium_high_threshold:
        risk_levels.append("中风险")
    else:
        risk_levels.append("高风险")

# 8. 评估风险分层效果
# 计算各风险层中实际异常的比例
risk_df = pd.DataFrame({
    '异常风险分数': abnormal_risk_scores,
    '风险等级': risk_levels,
    '实际标签': y_test
})

# 将实际标签转换为二分类（正常/异常）
risk_df['实际状态'] = risk_df['实际标签'].apply(
    lambda x: '异常' if x != normal_class_index else '正常'
)

# 计算各风险等级的异常比例
risk_summary = risk_df.groupby('风险等级')['实际状态'].value_counts(normalize=True).unstack().fillna(0)
print("\n各风险等级中异常样本的比例:")
print(risk_summary)

# 9. 可视化结果
plt.figure(figsize=(12, 5))

# 绘制风险分数分布
plt.subplot(1, 2, 1)
for risk_level in ['低风险', '中风险', '高风险']:
    subset = risk_df[risk_df['风险等级'] == risk_level]
    plt.hist(subset['异常风险分数'], alpha=0.7, label=risk_level, bins=30)
plt.axvline(x=low_medium_threshold, color='orange', linestyle='--', label='低/中风险分界')
plt.axvline(x=medium_high_threshold, color='red', linestyle='--', label='中/高风险分界')
plt.xlabel('异常风险分数')
plt.ylabel('频数')
plt.title('风险分数分布')
plt.legend()

# 绘制各风险等级的异常比例
plt.subplot(1, 2, 2)
risk_summary.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('各风险等级中异常样本比例')
plt.ylabel('比例')
plt.xticks(rotation=0)
plt.legend(title='实际状态')

plt.tight_layout()
plt.show()

# 10. 输出风险分层结果
print("\n前20个样本的风险评估结果:")
result_df = pd.DataFrame({
    '样本ID': X_test.index[:20],
    '异常风险分数': abnormal_risk_scores[:20],
    '风险等级': risk_levels[:20],
    '实际标签': [le.inverse_transform([label])[0] if 'le' in locals() else label for label in y_test[:20]]
})
print(result_df)

# 11. 保存结果
risk_df.to_csv('风险分层结果.csv', index=False)
print("\n风险分层结果已保存到 '风险分层结果.csv'")