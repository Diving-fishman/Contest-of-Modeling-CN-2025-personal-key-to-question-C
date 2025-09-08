import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
import warnings

warnings.filterwarnings("ignore")

# 可配置参数
INPUT_FILE = "男胎检测数据_清洗后.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_FILE = "所有孕妇_达到0.04_预测时间.xlsx"
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
    对单个孕妇按“标准化孕期”排序，判断是否达到阈值；
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

    # 协变量列（若存在就使用；缺失则自动忽略）
    raw_covs = ["年龄","孕妇BMI","身高","怀孕次数","生产次数"]
    for col in raw_covs:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    # 2) 逐孕妇求事件时间与删失状态
    records = []
    for pid, grp in df.groupby("孕妇代码", sort=False):
        duration, event, method = find_time_to_event(grp, THRESHOLD)
        first = grp.sort_values("标准化孕期").iloc[0]

        rec = {
            "孕妇代码": pid,
            "时间": float(duration),
            "事件": int(event),
            "确定方法": method,
        }
        # 取第一次测量的基线特征
        # 将“孕妇BMI”映射为“BMI”
        if "年龄" in grp.columns:
            rec["年龄"] = pd.to_numeric(first.get("年龄"), errors="coerce")
        if "孕妇BMI" in grp.columns:
            rec["BMI"] = pd.to_numeric(first.get("孕妇BMI"), errors="coerce")
        if "身高" in grp.columns:
            rec["身高"] = pd.to_numeric(first.get("身高"), errors="coerce")
        if "体重" in grp.columns:
            rec["体重"] = pd.to_numeric(first.get("体重"), errors="coerce")
        if "怀孕次数" in grp.columns:
            rec["怀孕次数"] = pd.to_numeric(first.get("怀孕次数"), errors="coerce")
        if "生产次数" in grp.columns:
            rec["生产次数"] = pd.to_numeric(first.get("生产次数"), errors="coerce")

        records.append(rec)

    survival_df = pd.DataFrame(records)

    # 3) 拟合 Weibull AFT（个体化预测），并给出全体预测
    # 动态选择可用协变量
    covariates = [c for c in ["年龄", "BMI", "身高", "体重", "怀孕次数", "生产次数"] if c in survival_df.columns]

    # 训练集：仅用于拟合（需去掉协变量缺失）
    if covariates:
        train = survival_df[["时间", "事件"] + covariates].dropna().copy()
    else:
        train = survival_df[["时间", "事件"]].copy()

    aft = WeibullAFTFitter()

    if len(train) >= 5:
        # 正常拟合（带协变量或仅截距）
        aft.fit(train, duration_col="时间", event_col="事件")
        covs_used = covariates  # 可能为空（仅截距）
    else:
        # 数据太少，退化为仅截距模型
        aft.fit(survival_df[["时间", "事件"]], duration_col="时间", event_col="事件")
        covs_used = []

    # 预测用全体矩阵，并以训练集中位数填补缺失，确保“所有孕妇”都有预测值
    if covs_used:
        X_all = survival_df[covs_used].copy()
        medians = train[covs_used].median()
        X_all = X_all.fillna(medians)
    else:
        X_all = pd.DataFrame(index=survival_df.index)  # 截距模型

    # 预测中位数与期望时间（两者都给，便于比较/选择）
    try:
        pred_median = aft.predict_median(X_all).astype(float).values
    except Exception:
        pred_median = np.full(len(survival_df), np.nan)

    try:
        pred_mean = aft.predict_expectation(X_all).astype(float).values
    except Exception:
        pred_mean = np.full(len(survival_df), np.nan)

    survival_df["预测达到时间_中位数"] = pred_median
    survival_df["预测达到时间_期望"] = pred_mean
    survival_df["观察到达到时间"] = np.where(survival_df["事件"] == 1, survival_df["时间"], np.nan)

    # 4) 导出：所有孕妇都有预测值
    out_cols = ["孕妇代码", "预测达到时间_中位数", "预测达到时间_期望", "事件", "观察到达到时间", "确定方法"]
    out_df = survival_df[out_cols].sort_values("孕妇代码")

    out_df.to_excel(OUTPUT_FILE, index=False)
    print(f"已保存：{OUTPUT_FILE}")
    print(f"样本量：{len(out_df)}，其中已达到阈值（事件=1）：{int(survival_df['事件'].sum())}。")


if __name__ == "__main__":
    main()
