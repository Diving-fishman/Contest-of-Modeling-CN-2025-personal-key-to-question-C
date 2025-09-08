import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np  # 需要导入numpy


def kmeans_clustering_excel(input_file, output_file, n_clusters=3):
    """
    使用K-Means对Excel表格中的某一列数据进行聚类分析

    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    n_clusters: 聚类的分组数量
    """

    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 1. 从Excel文件读取数据
    df = pd.read_excel(input_file)  # 使用传入的参数input_file

    # 删缺失值的行
    df_clean = df.dropna(subset=['BMI', '预测达到时间'])
    filtered_rows = df_clean[( df_clean['预测达到时间'] > 0 ) & ( df_clean['预测达到时间'] < 300 ) & ( df_clean['BMI'] > 20 )]

    # 使用筛选后的数据进行后续分析
    X = filtered_rows[['BMI', '预测达到时间']].values

    # 3. 数据标准化（推荐，特别是当两列数据的量纲不同时）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 使用肘部法则确定最佳K值
    inertias = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('K值')
    plt.ylabel('误差平方和')
    plt.title('肘部法则选择K值')
    plt.show()

    # 5. 根据肘部图或先验知识选择K值
    n_clusters = int(input("请输入决策后的k值："))

    # 6. 执行K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 获取聚类结果
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 将聚类中心转换回原始尺度
    centroids_original = scaler.inverse_transform(centroids)

    # 计算BMI的分位数作为分组边界
    bmi_quantiles = np.percentile(X[:, 0], [25, 50, 75])

    # 在控制台输出BMI分组区间
    print("\nBMI分组区间:")
    print(f"低BMI: < {bmi_quantiles[0]:.2f}")
    print(f"中BMI: {bmi_quantiles[0]:.2f} - {bmi_quantiles[1]:.2f}")
    print(f"中高BMI: {bmi_quantiles[1]:.2f} - {bmi_quantiles[2]:.2f}")
    print(f"高BMI: > {bmi_quantiles[2]:.2f}")

    # 7. 可视化聚类结果
    plt.figure(figsize=(10, 6))

    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', alpha=0.8)

    # 绘制聚类中心
    plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
                c='red', s=200, alpha=0.8, marker='X')

    plt.title('K-Means聚类结果')
    plt.xlabel('BMI')
    plt.ylabel('预测达到时间')
    plt.legend(['数据点', '聚类中心'])
    plt.show()

    # 8. 将聚类结果添加到原始DataFrame中
    # 创建一个新列并初始化为NaN
    df['Cluster'] = np.nan
    # 只将有效行的标签分配回去（使用filtered_rows.index而不是df_clean.index）
    df.loc[filtered_rows.index, 'Cluster'] = labels

    # 9. 将结果保存回Excel文件
    df.to_excel(output_file, index=False)

    return df  # 返回处理后的DataFrame


# 设置参数
input_file = "生存分析预测结果_详细报告.xlsx"  # 输入Excel文件路径
output_file = "K-Means结果.xlsx"  # 输出Excel文件路径

# 执行聚类分析
result_df = kmeans_clustering_excel(input_file, output_file)