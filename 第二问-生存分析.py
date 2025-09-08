import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, CoxPHFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import math

warnings.filterwarnings("ignore")

# 可配置参数
INPUT_FILE = "男胎检测数据_清洗后.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_FILE = "生存分析预测结果_详细报告_单变量.xlsx"
THRESHOLD = 0.04  # Y染色体浓度阈值


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 去除列名中的前后空白和换行
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace("\n", "", regex=False)
    return df


def coerce_numeric(series: pd.Series) -> pd.Series:
    # 将'≥3'、' ≥3 '、等样式统一为数字，无法转为数字的置为NaN
    s = series.astype(str).str.strip()
    s = s.str.replace("≥", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    return s


def find_time_to_event(group: pd.DataFrame, threshold: float = THRESHOLD):
    """
    对单个孕妇按"标准化孕期"排序，判断是否达到阈值；
    - 达到：返回首次达到阈值的精确时间（线性插值）与事件=1
    - 未达：返回最后一次观察时间与事件=0（删失）
    """
    g = group.sort_values("标准化孕期")
    y = g["Y染色体浓度"].to_numpy(dtype=float)
    t = g["标准化孕期"].to_numpy(dtype=float)

    reached_idx = np.where(y >= threshold)[0]
    if reached_idx.size > 0:
        i = int(reached_idx[0])
        if i == 0:
            return float(t[0]), 1, "首次测量即达到"
        # 线性插值
        x1, x2 = float(t[i - 1]), float(t[i])
        y1, y2 = float(y[i - 1]), float(y[i])
        if y2 == y1:
            return float(x2), 1, "平坦段取上界"
        et = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
        return float(et), 1, "线性插值"
    else:
        return float(t[-1]), 0, "未达到(删失)"


def main():
    # 1) 读取并清洗
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    df = clean_columns(df)

    required = ["孕妇代码", "Y染色体浓度", "标准化孕期"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"缺少必需列：{col}。请检查输入文件列名。")

    # 关键列清洗
    df = df.dropna(subset=required).copy()
    df = df[(df["Y染色体浓度"] >= 0) & (df["标准化孕期"] > 0)].copy()

    # 协变量列（只使用BMI）
    covariate_col = "孕妇BMI"
    if covariate_col in df.columns:
        df[covariate_col] = coerce_numeric(df[covariate_col])
    else:
        raise KeyError(f"缺少BMI列：{covariate_col}")

    # 2) 逐孕妇求事件时间与删失状态
    records = []
    data_point_counts = {}

    for pid, grp in df.groupby("孕妇代码", sort=False):
        data_point_counts[pid] = len(grp)
        duration, event, method = find_time_to_event(grp, THRESHOLD)
        first = grp.sort_values("标准化孕期").iloc[0]

        rec = {
            "孕妇代码": pid,
            "实际/删失时间": float(duration),
            "是否达到0.04(1=是,0=否)": int(event),
            "确定方法": method,
        }

        # 只使用BMI作为协变量
        rec["BMI"] = pd.to_numeric(first.get(covariate_col), errors="coerce")

        records.append(rec)

    survival_df = pd.DataFrame(records)

    # 3) 拟合 Weibull AFT 和 Cox 模型（只使用BMI）
    covariates = ["BMI"]

    # 准备训练数据（去掉协变量缺失的样本）
    train = survival_df[["实际/删失时间", "是否达到0.04(1=是,0=否)"] + covariates].dropna().copy()

    # Weibull AFT 模型
    aft = WeibullAFTFitter()
    aft.fit(train, duration_col="实际/删失时间", event_col="是否达到0.04(1=是,0=否)")

    # Cox 模型
    cph = CoxPHFitter()
    cph.fit(train, duration_col="实际/删失时间", event_col="是否达到0.04(1=是,0=否)")

    # 4) 预测所有个体的达到时间（使用中位数填补缺失的BMI）
    X_all = survival_df[covariates].copy()
    medians = train[covariates].median()
    X_all = X_all.fillna(medians)

    # Weibull AFT 预测
    pred_median_aft = aft.predict_median(X_all).astype(float).values
    survival_df["预测达到时间"] = pred_median_aft

    # 5) 计算模型评估指标
    # 只对事件=1的样本计算误差
    event_mask = survival_df["是否达到0.04(1=是,0=否)"] == 1
    actual_times = survival_df.loc[event_mask, "实际/删失时间"]
    predicted_times = survival_df.loc[event_mask, "预测达到时间"]

    mae = mean_absolute_error(actual_times, predicted_times)
    rmse = np.sqrt(mean_squared_error(actual_times, predicted_times))

    # 中位生存时间
    median_survival_time = np.median(actual_times) if len(actual_times) > 0 else np.nan

    # C-index
    c_index_aft = aft.concordance_index_
    c_index_cox = cph.concordance_index_

    # 样本统计
    total_samples = len(survival_df)
    event_count = survival_df["是否达到0.04(1=是,0=否)"].sum()
    censored_count = total_samples - event_count

    # 6) 准备输出到多个工作表
    # 工作表1: 个体预测结果
    individual_results = survival_df.copy()

    # 工作表2: 模型评估
    evaluation_metrics = pd.DataFrame({
        "指标": ["中位生存时间(天)", "平均绝对误差(MAE)", "均方根误差(RMSE)",
                 "Cox模型C-index", "Weibull AFT模型C-index", "总样本数", "事件发生数", "删失数"],
        "值": [median_survival_time, mae, rmse, c_index_cox, c_index_aft,
               total_samples, event_count, censored_count]
    })

    # 工作表3: Cox模型系数
    cox_summary = cph.summary
    cox_coefficients = pd.DataFrame({
        "covariate": cox_summary.index,
        "coef": cox_summary["coef"],
        "exp(coef)": cox_summary["exp(coef)"],
        "se(coef)": cox_summary["se(coef)"],
        "coef lower 95%": cox_summary["coef lower 95%"],
        "coef upper 95%": cox_summary["coef upper 95%"],
        "exp(coef) lower 95%": cox_summary["exp(coef) lower 95%"],
        "exp(coef) upper 95%": cox_summary["exp(coef) upper 95%"],
        "cmp to": 0,  # 参考水平
        "z": cox_summary["z"],
        "p": cox_summary["p"],
        "-log2(p)": -np.log2(cox_summary["p"])
    })

    # 工作表4: Weibull AFT模型系数
    aft_summary = aft.summary
    aft_coefficients = pd.DataFrame({
        "param": aft_summary.index,
        "covariate": [idx[1] if isinstance(idx, tuple) and len(idx) > 1 else "Intercept"
                      for idx in aft_summary.index],
        "coef": aft_summary["coef"],
        "exp(coef)": aft_summary["exp(coef)"],
        "se(coef)": aft_summary["se(coef)"],
        "coef lower 95%": aft_summary["coef lower 95%"],
        "coef upper 95%": aft_summary["coef upper 95%"],
        "exp(coef) lower 95%": aft_summary["exp(coef) lower 95%"],
        "exp(coef) upper 95%": aft_summary["exp(coef) upper 95%"],
        "cmp to": 0,  # 参考水平
        "z": aft_summary["z"],
        "p": aft_summary["p"],
        "-log2(p)": -np.log2(aft_summary["p"])
    })

    # 工作表5: 数据概览
    overview_df = survival_df.copy()
    overview_df["数据点数量"] = overview_df["孕妇代码"].map(data_point_counts)

    # 工作表6: 预测误差分析
    error_analysis = survival_df[event_mask].copy()
    error_analysis["绝对误差"] = abs(error_analysis["实际/删失时间"] - error_analysis["预测达到时间"])
    error_analysis["相对误差(%)"] = (error_analysis["绝对误差"] / error_analysis["实际/删失时间"]) * 100

    # 7) 写入Excel文件
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        individual_results.to_excel(writer, sheet_name='个体预测结果', index=False)
        evaluation_metrics.to_excel(writer, sheet_name='模型评估', index=False)
        cox_coefficients.to_excel(writer, sheet_name='Cox模型系数', index=False)
        aft_coefficients.to_excel(writer, sheet_name='Weibull AFT模型系数', index=False)
        overview_df.to_excel(writer, sheet_name='数据概览', index=False)
        error_analysis.to_excel(writer, sheet_name='预测误差分析', index=False)

    print(f"已保存：{OUTPUT_FILE}")
    print(f"样本量：{total_samples}，其中已达到阈值（事件=1）：{event_count}，删失：{censored_count}。")


if __name__ == "__main__":
    main()