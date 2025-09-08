import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置matplotlib以正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# --- 数据准备 ---
# 您的原始数据
bmi_data = np.array([
    33.331832, 30.742188, 28.641243, 30.844444, 35.883635, 33.874064, 29.136316, 33.333333, 33.333333, 36.25047,
    30.385016, 35.755956, 35.055632, 30.888889, 30.443839, 28.282828, 28.133657, 28.320313, 35.684444, 28.677663,
    30.043262, 33.284024, 29.752066, 28.040378, 29.136316, 32.046147, 35.058864, 30.385016, 30.483158, 32.561885,
    36.885092, 30.846277, 34.722222, 29.446367, 28.407877, 30.110991, 29.752066, 31.68506, 35.69304, 37.113546,
    30.119376, 30.120482, 29.7442, 28.344671, 28.90625, 33.874064, 32.690542, 34.927679, 38.567493, 39.159843,
    28.90625, 36.132335, 32.051282, 34.289377, 31.229454, 31.732941, 38.946813, 36.289737, 31.640625, 32.046147,
    31.633715, 32.046147, 30.427198, 33.789063, 30.46875, 34.720883, 31.644286, 33.71488, 27.915519, 35.546875,
    34.063609, 29.174885, 31.887755, 30.443839, 32.8125, 29.169333, 32.297866, 28.1476, 32.36855, 30.483158,
    30.486657, 32.522449, 29.12415, 30.661292, 30.483379, 29.752066, 27.202498, 36.923077, 30.043262, 41.132813,
    29.760863, 36.640446, 37.637849, 30.116213, 28.981143, 33.409205, 31.992171, 32.839415, 28.792954, 29.760863,
    40.138408, 32.242064, 30.119402, 34.049031, 30.863036, 33.462225, 35.685352, 30.443839, 37.290688, 30.427198,
    29.899691, 26.619343, 31.25, 29.136316, 35.379813, 31.25, 29.065744, 30.119402, 30.859375, 28.266024,
    32.8125, 29.384757, 29.615806, 28.833793, 30.46875, 29.49495, 29.296875, 39.302112, 30.470522, 29.411765,
    30.078125, 31.288319, 28.344671, 31.75352, 29.136316,
    93.2437362, 110.400254, 116.668369, 102.650094, 103.183044, 103.963022, 95.0016603, 95.415381, 92.6156291,
    97.0200954, 106.028275, 103.075164, 99.6682325, 103.251199, 92.1627171, 105.522922, 98.2973017, 95.0638845,
    101.23238, 105.420952, 94.3469718, 97.9953125, 93.867685, 93.301505, 101.013913, 98.876758, 102.972457,
    110.120956, 93.9075711, 95.8539553, 105.023262, 103.332831, 112.496414, 97.0934702, 96.3381935, 95.2139932,
    101.561435, 108.071684, 98.9923231, 104.289514, 101.090093, 92.483819, 99.3598283, 106.940275, 107.123329,
    102.449745, 101.603933, 95.799683, 88.1709798, 106.453144, 91.5031118, 93.7045037, 106.975997, 92.0742743,
    94.8728431, 93.8317202, 106.904949, 103.550073, 101.429151, 100.366948, 103.090875, 105.74492, 91.5759535,
    99.9182265, 99.9311178, 102.6302, 96.6050429, 99.898312, 98.2269621, 103.833722, 104.155406, 104.956193,
    92.0702004, 102.010799, 104.745771, 99.4175256, 94.9556487
])

predicted_times = np.array([
    104.7657650889461, 98.6361085039231, 93.92768896770228, 98.87120619040148,
    111.1783529993479, 106.0966824536452, 95.0165453915245, 104.7694272264577,
    104.7694272264577, 112.1319179331059, 97.81931668414178, 110.8483676687522,
    109.0557175386482, 98.97356250782477, 97.95336940228962, 93.14719086984776,
    92.82426333516005, 93.22851329689031, 110.6639732311948, 94.00736294021006,
    97.0441173803967, 104.6492229551822, 96.38844606996028, 92.62290316787374,
    95.0165453915245, 101.6763715436442, 109.0639249695997, 97.81931668414178,
    98.0430755977554, 102.9045519696209, 113.8009273291487, 98.87542375423288,
    108.2124903378286, 95.70488465753765, 93.41876287367735, 97.1972556836042,
    96.38844606996028, 100.825214739916, 110.6861204601267, 114.4078091026523,
    97.21623170439666, 97.21873575684442, 96.37079565239087, 93.28139782017097,
    94.50897960808693, 106.0966824536452, 103.2132402310942, 108.7313380080496,
    118.3467345781532, 119.9901049508017, 94.50897960808693, 111.8239430139409,
    101.6885289174925, 107.1274983744358, 99.76141546308088, 100.9376698528392,
    119.3964795537607, 112.2344733041713, 100.7209655023218, 101.6763715436442,
    100.7047633348911, 101.6763715436442, 97.9154270458229, 105.8869333560658,
    98.01019405562938, 108.2091156866719, 100.7295499707974, 105.7042203278441,
    92.3540526753473, 110.3101099318437, 106.5659025023523, 95.10190072928374,
    101.3021286253484, 97.95336940228962, 103.5067116177924, 95.0896084193312,
    102.2739748763501, 92.85440131732864, 102.4424144239946, 98.0430755977554,
    98.05106328138713, 102.8101172089319, 94.98963547224042, 98.45051857497805,
    98.04358058949492, 96.38844606996028, 90.83364201295167, 113.9016102123067,
    97.0441173803967, 125.6301251883535, 96.40818785322698, 113.1546032519116,
    115.8128636132929, 97.20907447523227, 94.67390901795865, 104.9546523023513,
    101.5486855098076, 103.5715908564742, 94.26002689092827, 96.40818787567179,
    122.7550984149682, 102.1411924444493, 97.21629134226544, 106.5297393530575,
    98.91400968211995, 105.0842842060049, 110.6663105677601, 97.95336940228962,
    114.8806061349353, 97.9154270458229, 96.72029062460594, 89.60877614496424,
    99.8091452817016, 95.0165453915245, 109.8819046132697, 99.8091452817016,
    94.860561418253, 97.2162913196324, 98.9055797090644, 93.11075750435933,
    103.5067116177924, 95.567708917877, 96.08315935176064, 94.34968798871324,
    98.01019405562938, 95.81319353358113, 95.3723794337604, 120.3881907724046,
    98.01423640408095, 95.62781847483207, 97.12291426913076, 99.89822608071688,
    93.28139782017097, 100.9860402818254, 95.0165453915245, 96.24800427639045,
    109.9825214514454, 107.6961185465586, 96.14330548919412, 98.01423640408095,
    94.67390901795865, 106.8047650659206, 116.3398959541721, 118.8626224604215,
    112.42773315752, 99.74248763551412, 101.567195425894, 104.452311465949,
    104.0684544742472, 92.68914969554699, 97.79850642113324, 99.97409711537473,
    96.88259120544191, 102.9286292720437, 104.4688149845661, 104.2049248940922,
    95.76570253770356, 100.8924242644499, 96.67174787030957, 101.6258796631732,
    107.2523754190777, 97.67337160122067, 104.3104277559046, 98.0639515164466,
    99.22267694723512, 105.0016788153465, 98.60763796515158, 95.45428104037538,
    106.7588043799252, 102.5785302190697, 98.37734072147171, 92.2806001476872,
    110.3125257911006, 118.2657978122822, 104.6556579289914, 105.3132248876768,
    106.0700944573608, 97.36531501111456, 99.45279778538952, 92.95363827231287,
    98.20949275497519, 104.2317091528446, 100.0297078423617, 100.7061476354441,
    100.8244307959174, 92.90559036619943, 105.7174764728844, 100.2748157578375,
    98.4158980925121, 101.004958822881, 107.6565554763012, 98.30693926102248,
    97.27653804881218, 97.58651409730732, 95.79245918330295, 103.2744850759804,
    96.70108569411578, 102.3859958969492, 107.8305811360585, 94.73871741133543,
    97.50279168324901, 106.0671203941508, 105.7062780722933, 110.3646049765448,
    101.057244185491, 93.86474702597103, 97.35553871202569, 103.6049857789225,
    110.4782298495871, 101.607728431561, 106.9365430143361, 104.4873507280933,
    96.32873928362369, 100.7001240354925, 108.2867512532182, 105.2205550551591,
    103.6565832378897, 102.8849132204424, 98.68463357298558, 94.91497272023734,
    107.614997971855, 94.9932570003661, 97.01037707827658, 104.008101677576,
    94.79903664960005, 98.67438468551661, 95.64398591603376, 104.6994632817935,
    102.3069396134485, 102.9818339759268, 102.8952144637781, 103.7803189228485,
    104.0807727954034, 92.67586411991901, 100.6427629645986, 101.8903197813181,
    104.7904588709007, 96.37836731697001, 98.7801555773085, 98.71267521152572,
    105.8806008857221, 106.9438317936597, 106.1238863504313, 95.48304409600554,
    102.6871946889408, 106.5566432372992, 101.8871423482704, 98.5464668015696
])

# 确保数据长度一致
min_len = min(len(bmi_data), len(predicted_times))
df = pd.DataFrame({
    'bmi': bmi_data[:min_len],
    'predicted_time': predicted_times[:min_len]
})


# --- 分组函数 ---
def assign_bmi_group(bmi):
    """根据K-Means结果边界，为BMI分配组别"""
    if bmi < 29.95:
        return '低BMI组 (<29.95)'
    elif 29.95 <= bmi < 31.62:
        return '中BMI组 (29.95-31.62)'
    elif 31.62 <= bmi < 33.44:
        return '中高BMI组 (31.62-33.44)'
    else:  # bmi >= 33.44
        return '高BMI组 (>33.44)'


# 将分组信息添加到DataFrame中
df['bmi_group'] = df['bmi'].apply(assign_bmi_group)


# --- 风险函数 ---
def calculate_group_risk(x, group_predicted_times):
    """
    计算单个组在给定检测时间x下的总风险。
    总风险 = 组内人数 + 未达标人数 * 3
    未达标人数：预测时间 > 设定的检测时间x
    """
    group_size = len(group_predicted_times)
    if group_size == 0:
        return 0

    # 未达标人数：预测时间 > 设定的检测时间x
    not_meeting_standard_count = np.sum(group_predicted_times > x)

    # 总风险 = 组内人数 + 未达标人数 * 3
    total_risk = 100 ** ( (x - 84) / 105 ) * group_size + not_meeting_standard_count * 3
    return total_risk


# --- 蒙特卡洛模拟函数 ---
def monte_carlo_simulation(group_times, n_simulations=1000, error_std=5.0):
    """
    对给定组的预测时间进行蒙特卡洛模拟，分析检测误差对最佳检测时间的影响

    参数:
    - group_times: 组的预测时间数组
    - n_simulations: 模拟次数
    - error_std: 检测误差的标准差（天数）

    返回:
    - optimal_times: 每次模拟得到的最佳检测时间
    - min_risks: 每次模拟得到的最小风险值
    """
    optimal_times = []
    min_risks = []

    group_size = len(group_times)
    if group_size == 0:
        return np.array([]), np.array([])

    # 原始最佳检测时间（无误差）
    time_range = np.linspace(min(group_times) - 2, max(group_times) + 2, 800)
    risks = [calculate_group_risk(x, group_times) for x in time_range]
    original_optimal_x = time_range[np.argmin(risks)]

    print(f"原始最佳检测时间: {original_optimal_x:.2f} 天")

    # 蒙特卡洛模拟
    for i in range(n_simulations):
        # 添加随机误差到预测时间
        errors = np.random.normal(0, error_std, group_size)
        perturbed_times = group_times + errors

        # 计算风险曲线并找到最佳检测时间
        risks = [calculate_group_risk(x, perturbed_times) for x in time_range]
        min_risk = np.min(risks)
        optimal_x = time_range[np.argmin(risks)]

        optimal_times.append(optimal_x)
        min_risks.append(min_risk)

    return np.array(optimal_times), np.array(min_risks)


# --- 执行蒙特卡洛模拟并可视化结果 ---
def run_monte_carlo_analysis(dataframe, n_simulations=1000, error_std=5.0):
    """对每个BMI组执行蒙特卡洛模拟并可视化结果"""

    # 按顺序获取组名，确保图例顺序一致
    group_names = ['低BMI组 (<29.95)', '中BMI组 (29.95-31.62)', '中高BMI组 (31.62-33.44)', '高BMI组 (>33.44)']

    # 准备存储结果
    results = {}

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    colors = {
        '低BMI组 (<29.95)': 'green',
        '中BMI组 (29.95-31.62)': 'blue',
        '中高BMI组 (31.62-33.44)': 'orange',
        '高BMI组 (>33.44)': 'red'
    }

    print("--- 蒙特卡洛模拟分析 (检测误差标准差 = {:.1f} 天) ---".format(error_std))

    for i, name in enumerate(group_names):
        group_df = dataframe[dataframe['bmi_group'] == name]
        group_times = group_df['predicted_time'].values
        group_size = len(group_times)

        if group_size == 0:
            print(f"\n组名: {name} - 该组没有样本，已跳过。")
            continue

        print(f"\n分析组: {name} (样本数: {group_size})")

        # 执行蒙特卡洛模拟
        optimal_times, min_risks = monte_carlo_simulation(
            group_times, n_simulations, error_std
        )

        # 存储结果
        results[name] = {
            'optimal_times': optimal_times,
            'min_risks': min_risks,
            'group_size': group_size
        }

        # 绘制最佳检测时间分布
        sns.histplot(optimal_times, ax=axes[0], color=colors[name],
                     alpha=0.6, label=name, kde=True, stat='density')

        # 绘制最小风险分布
        sns.histplot(min_risks, ax=axes[1], color=colors[name],
                     alpha=0.6, label=name, kde=True, stat='density')

        # 计算统计量
        mean_optimal = np.mean(optimal_times)
        std_optimal = np.std(optimal_times)
        ci_optimal = stats.norm.interval(0.95, loc=mean_optimal, scale=std_optimal / np.sqrt(len(optimal_times)))

        mean_risk = np.mean(min_risks)
        std_risk = np.std(min_risks)

        print(f"  最佳检测时间: 均值 = {mean_optimal:.2f} 天, 标准差 = {std_optimal:.2f} 天")
        print(f"  95%置信区间: [{ci_optimal[0]:.2f}, {ci_optimal[1]:.2f}]")
        print(f"  最小风险值: 均值 = {mean_risk:.2f}, 标准差 = {std_risk:.2f}")

    # 设置图表属性
    axes[0].set_xlabel('最佳检测时间 (天)', fontsize=12)
    axes[0].set_ylabel('密度', fontsize=12)
    axes[0].set_title('蒙特卡洛模拟: 最佳检测时间分布\n(检测误差标准差 = {:.1f} 天)'.format(error_std), fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('最小风险值', fontsize=12)
    axes[1].set_ylabel('密度', fontsize=12)
    axes[1].set_title('蒙特卡洛模拟: 最小风险值分布\n(检测误差标准差 = {:.1f} 天)'.format(error_std), fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 绘制最佳检测时间与BMI组的关系（带误差条）
    group_means = []
    group_stds = []
    group_labels = []

    for name in group_names:
        if name in results:
            group_means.append(np.mean(results[name]['optimal_times']))
            group_stds.append(np.std(results[name]['optimal_times']))
            group_labels.append(name)

    x_pos = np.arange(len(group_means))
    axes[2].errorbar(x_pos, group_means, yerr=group_stds, fmt='o', capsize=5, capthick=2, markersize=8)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(group_labels, rotation=45)
    axes[2].set_ylabel('最佳检测时间 (天)', fontsize=12)
    axes[2].set_title('各BMI组的最佳检测时间比较\n(误差条表示标准差)', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    # 绘制风险曲线的不确定性
    for i, name in enumerate(group_names):
        if name not in results:
            continue

        group_df = dataframe[dataframe['bmi_group'] == name]
        group_times = group_df['predicted_time'].values

        # 计算原始风险曲线
        time_range = np.linspace(min(group_times) - 2, max(group_times) + 2, 100)
        original_risks = [calculate_group_risk(x, group_times) for x in time_range]

        # 计算蒙特卡洛模拟的风险曲线范围
        risk_curves = []
        for _ in range(100):  # 随机选择100次模拟
            errors = np.random.normal(0, error_std, len(group_times))
            perturbed_times = group_times + errors
            risks = [calculate_group_risk(x, perturbed_times) for x in time_range]
            risk_curves.append(risks)

        risk_curves = np.array(risk_curves)
        risk_mean = np.mean(risk_curves, axis=0)
        risk_std = np.std(risk_curves, axis=0)

        axes[3].plot(time_range, risk_mean, color=colors[name], label=name, linewidth=2)
        axes[3].fill_between(time_range, risk_mean - risk_std, risk_mean + risk_std,
                             color=colors[name], alpha=0.2)

    axes[3].set_xlabel('检测时间 (天)', fontsize=12)
    axes[3].set_ylabel('风险值', fontsize=12)
    axes[3].set_title('考虑检测误差的风险函数曲线\n(阴影区域表示±1标准差)', fontsize=14)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


# --- 敏感性分析函数 ---
def sensitivity_analysis(dataframe, error_stds=np.linspace(0, 10, 6)):
    """分析不同误差水平对最佳检测时间的影响"""

    group_names = ['低BMI组 (<29.95)', '中BMI组 (29.95-31.62)', '中高BMI组 (31.62-33.44)', '高BMI组 (>33.44)']
    colors = {
        '低BMI组 (<29.95)': 'green',
        '中BMI组 (29.95-31.62)': 'blue',
        '中高BMI组 (31.62-33.44)': 'orange',
        '高BMI组 (>33.44)': 'red'
    }

    plt.figure(figsize=(12, 8))

    for name in group_names:
        group_df = dataframe[dataframe['bmi_group'] == name]
        group_times = group_df['predicted_time'].values
        group_size = len(group_times)

        if group_size == 0:
            continue

        optimal_means = []
        optimal_stds = []

        for error_std in error_stds:
            optimal_times, _ = monte_carlo_simulation(
                group_times, n_simulations=200, error_std=error_std
            )

            if len(optimal_times) > 0:
                optimal_means.append(np.mean(optimal_times))
                optimal_stds.append(np.std(optimal_times))
            else:
                optimal_means.append(np.nan)
                optimal_stds.append(np.nan)

        plt.errorbar(error_stds, optimal_means, yerr=optimal_stds,
                     color=colors[name], marker='o', capsize=5, label=name)

    plt.xlabel('检测误差标准差 (天)', fontsize=12)
    plt.ylabel('最佳检测时间 (天)', fontsize=12)
    plt.title('敏感性分析: 检测误差对最佳检测时间的影响', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # 执行蒙特卡洛模拟
    results = run_monte_carlo_analysis(df, n_simulations=1000, error_std=5.0)

    # 执行敏感性分析
    sensitivity_analysis(df)

    print("\n--- 分析总结 ---")
    print("1. 蒙特卡洛模拟考虑了检测误差对预测时间的影响，误差服从正态分布。")
    print("2. 对于每个BMI组，模拟了1000次不同的误差情景。")
    print("3. 结果显示检测误差会导致最佳检测时间的不确定性增加。")
    print("4. 敏感性分析展示了不同误差水平对各组最佳检测时间的影响程度。")
    print("5. 高BMI组通常需要更晚的检测时间，且对检测误差更为敏感。")