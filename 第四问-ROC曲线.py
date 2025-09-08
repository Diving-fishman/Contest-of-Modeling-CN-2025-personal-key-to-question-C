import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载数据
# 假设你的Excel文件名为'女_副本(2).xlsx'
file_path = '整理后的数据——女.xlsx'
df = pd.read_excel(file_path)

# 2. 数据预处理
# 查看数据基本信息
print("数据形状:", df.shape)
print("\n数据前5行:")
print(df.head())
print("\n数据列名:")
print(df.columns.tolist())

# 假设目标变量列名为'染色体的非整倍体'
target_column = '染色体的非整倍体'

# 检查目标变量的分布
print(f"\n目标变量'{target_column}'的分布:")
print(df[target_column].value_counts())

# 定义特征列（根据你的文档）
feature_columns = [
    '孕妇BMI', '原始读段数', 'GC含量', '13号染色体的Z值',
    '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
    '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'
]

# 检查是否存在缺失值
print("\n缺失值统计:")
print(df[feature_columns + [target_column]].isnull().sum())

# 如果有缺失值，进行简单处理（这里用均值填充）
for col in feature_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# 将目标变量转换为数值型（如果是分类变量）
if df[target_column].dtype == 'object':
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_column])
    # 保存标签映射关系
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n目标变量编码映射: {class_mapping}")
else:
    y_encoded = df[target_column].values
    class_mapping = {str(i): i for i in np.unique(y_encoded)}

# 3. 准备特征和目标变量
X = df[feature_columns]
y = y_encoded

# 获取类别列表
classes = np.unique(y)
n_classes = len(classes)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 训练逻辑回归模型
model = LogisticRegression(
    max_iter=1000,
    tol=0.001,
    fit_intercept=True,
    random_state=42,
    multi_class='ovr'  # 使用一对多策略处理多分类
)

model.fit(X_train_scaled, y_train)

# 7. 模型评估
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 8. 绘制混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 9. 绘制ROC曲线（多分类）
# 将标签二值化
y_test_bin = label_binarize(y_test, classes=classes)

# 为每个类别计算ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算微平均ROC曲线和AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 绘制所有ROC曲线
plt.figure(figsize=(10, 8))

# 绘制微平均ROC曲线
plt.plot(fpr["micro"], tpr["micro"],
         label='微平均ROC曲线 (AUC = {0:0.3f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# 绘制每个类别的ROC曲线
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
class_names = list(class_mapping.keys())

for i, color in zip(range(n_classes), colors):
    if i < len(class_names):
        class_label = class_names[i]
    else:
        class_label = f'Class {i}'

    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (AUC = {1:0.3f})'
                   ''.format(class_label, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('多分类ROC曲线')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 10. 输出模型性能指标
print("\n模型性能指标:")
print(f"训练集准确率: {model.score(X_train_scaled, y_train):.3f}")
print(f"测试集准确率: {model.score(X_test_scaled, y_test):.3f}")

# 11. 特征重要性分析
if hasattr(model, 'coef_'):
    # 对于多分类问题，coef_是一个二维数组，每行对应一个类别
    # 我们可以取所有类别系数的平均值作为特征重要性
    feature_importance = pd.DataFrame({
        '特征': feature_columns,
        '重要性': np.mean(np.abs(model.coef_), axis=0)  # 取所有类别系数的绝对值的平均值
    })
    feature_importance = feature_importance.sort_values('重要性', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=feature_importance)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.show()

    print("\n特征重要性排序:")
    print(feature_importance)

# 12. 保存模型（可选）
import joblib

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n模型已保存为 'logistic_regression_model.pkl'")
print("标准化器已保存为 'scaler.pkl'")