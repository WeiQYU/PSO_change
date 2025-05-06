import cupy as cp
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import pandas as pd
from PSO_main_new import *

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30
pc = 3.086e16

print("加载数据...")
TrainingData = scio.loadmat('../../generate_ligo/noise.mat')
analysisData = scio.loadmat('../../generate_ligo/data.mat')
print("透镜的数据")
print("加载完毕")

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
training_noise = cp.asarray(TrainingData['noise'][0])  # Move to GPU
dataY_only_signal = dataY - training_noise  # Extract signal part (for comparison)

nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])

#                r    mc   tc   phi    A      Δtd
rmin = cp.array([-2, 0, 1, 0, 0, 0])
rmax = cp.array([4, 2, 8.0, np.pi, 1.0, 4.0])  # Changed A upper limit to 2.0

# Time domain settings
dt = 1 / Fs  # sampling rate Hz
t = cp.arange(0, 8, dt)  # Using CuPy for t array
T = nSamples / Fs
df = 1 / T
Nyq = Fs / 2

# Calculate PSD
psdHigh = cp.asarray(TrainingData['psd'][0])  # Get PSD directly from training data

# PSO input parameters - keep everything on GPU
inParams = {
    'dataX': t,
    'dataY': dataY,
    'dataY_only_signal': dataY_only_signal,  # 明确添加纯信号数据用于匹配滤波
    'sampFreq': Fs,
    'psdHigh': psdHigh,
    'rmin': rmin,
    'rmax': rmax,
}

# 设置平衡维度PSO参数
balanced_pso = True  # 启用平衡维度PSO

# Enable Bayesian approach with blind priors
use_bayesian = True  # 启用贝叶斯方法

# 增加运行次数
nRuns = 8

# PSO配置参数
pso_config = {
    'popsize': 100,  # 增加粒子数量以确保每个维度有足够的覆盖
    'maxSteps': 3000,  # 迭代次数
    'c1': 2.0,  # 个体学习因子
    'c2': 2.0,  # 社会学习因子
    'w_start': 0.9,  # 初始惯性权重
    'w_end': 0.5,  # 最终惯性权重
    'max_velocity': 0.4,  # 速度限制
    'nbrhdSz': 6,  # 邻域大小
    'disable_early_stop': True,  # 禁用早停

    # 平衡维度PSO特定参数
    'use_dimension_balance': balanced_pso,  # 使用维度平衡策略
    'use_dimension_groups': True,  # 使用维度分组
    'use_dimension_restart': True,  # 使用维度特定的重启策略
    'use_specialized_init': True,  # 使用专门化的初始化

    # 定义维度分组
    'dimension_groups': [
        [0, 1],  # r, m_c: 距离和质量参数
        [2, 3],  # tc, phi_c: 时间和相位参数
        [4, 5]  # A, delta_t: 透镜参数
    ],

    # 维度名称，用于分析报告
    'dim_names': ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t'],

    # 自适应策略参数
    'use_adaptive_strategy': True,  # 启用自适应策略
    'adaptation_interval': 100,  # 适应间隔
    'exploration_threshold': 0.6,  # 探索阈值
}

print("PSO已部署完毕,芜湖！！！！")
if balanced_pso:
    print("使用平衡维度PSO策略，确保6维参数空间的充分搜索")
if use_bayesian:
    print("使用贝叶斯方法，引入物理先验知识（不使用真实参数）")

# 设置固定种子以确保可重复性
np.random.seed(42)
seeds = np.random.randint(0, 10000, nRuns)

# 定义实际参数用于结果比较（仅评估用）
actual_params = {
    'chirp_mass': 30.09,  # Solar masses
    'merger_time': 7.5000,  # seconds
    'source_distance': 3100.0,  # Mpc
    'lens_mass': 5.0000e+04,  # 单位未知
    'y': 0.5000,  # 无量纲
    'flux_ratio': 0.3333,  # A参数
    'time_delay': 0.9854,  # seconds (delta_t)
    'phase': 0.2500  # 2π的分数
}

# 打印真实参数 - 仅用于结果对比，不用于拟合
print("\n真实参数 (仅用于结果比较):")
print(f"Chirp Mass: {actual_params['chirp_mass']:.2f} M⊙")
print(f"Merger Time: {actual_params['merger_time']:.4f} s")
print(f"Source Distance: {actual_params['source_distance']:.1f} Mpc")
print(f"Flux Ratio (A): {actual_params['flux_ratio']:.4f}")
print(f"Time Delay (delta_t): {actual_params['time_delay']:.4f} s")
print(f"Phase: {actual_params['phase']:.4f} (fraction of 2π)")

# 运行PSO优化，启用两步匹配过程和平衡维度策略
# 传入actual_params仅用于评估，不用于构建先验
outResults, outStruct = crcbqcpsopsd(inParams, pso_config, nRuns,
                                     use_two_step=True,
                                     actual_params=actual_params,  # 仅用于评估，不用于参数估计
                                     balanced_pso=balanced_pso,
                                     use_bayesian=use_bayesian)

# 将数据移到CPU用于绘图
t_cpu = cp.asnumpy(t)

# 绘制PSO收敛历史图
plt.figure(figsize=(12, 8), dpi=200)
for lpruns in range(nRuns):
    if 'fitnessHistory' in outStruct[lpruns]:
        plt.plot(outStruct[lpruns]['fitnessHistory'], label=f'Run {lpruns + 1}')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value (SNR)')
title = 'PSO Fitness History'
if balanced_pso:
    title = 'Balanced Dimension ' + title
if use_bayesian:
    title = 'Bayesian ' + title
plt.title(title)
plt.legend()
plt.grid(True)
plt.savefig('pso_convergence_plot.png')
plt.close()

# 可视化不同运行的参数收敛情况
param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
param_values = [[] for _ in range(len(param_names))]

for lpruns in range(nRuns):
    for i, param in enumerate(param_names):
        # Convert CuPy arrays to NumPy/Python types if needed
        value = outResults['allRunsOutput'][lpruns][param]
        if isinstance(value, cp.ndarray):
            value = float(value.get())  # Explicitly convert CuPy array to float
        param_values[i].append(value)

# 创建参数散点图
fig, axs = plt.subplots(3, 2, figsize=(15, 12), dpi=200)
for i, (param, values) in enumerate(zip(param_names, param_values)):
    row, col = i // 2, i % 2
    axs[row, col].scatter(range(1, nRuns + 1), values, color='blue', alpha=0.7)
    if i < 2:  # Log scale for r and m_c parameters
        axs[row, col].set_ylabel(f'10^({param})')
    else:
        axs[row, col].set_ylabel(param)
    axs[row, col].set_xlabel('Run Number')
    axs[row, col].set_title(f'Parameter {param} Across Runs')
    axs[row, col].grid(True, alpha=0.3)
    # Highlight best run
    best_run = outResults['bestRun'] + 1
    axs[row, col].scatter([best_run], [values[best_run - 1]], color='red', s=100,
                          label=f'Best Run ({best_run})', zorder=10)

    # 添加实际值标记（如果可用）
    if param == 'r' and 'source_distance' in actual_params:
        actual_val = np.log10(actual_params['source_distance'])
        axs[row, col].axhline(y=actual_val, color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["source_distance"]:.1f} Mpc')
    elif param == 'm_c' and 'chirp_mass' in actual_params:
        actual_val = np.log10(actual_params['chirp_mass'])
        axs[row, col].axhline(y=actual_val, color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["chirp_mass"]:.2f} M⊙')
    elif param == 'tc' and 'merger_time' in actual_params:
        axs[row, col].axhline(y=actual_params['merger_time'], color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["merger_time"]:.2f} s')
    elif param == 'phi_c' and 'phase' in actual_params:
        axs[row, col].axhline(y=actual_params['phase'] * 2 * np.pi, color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["phase"] * 2 * np.pi:.2f} rad')
    elif param == 'A' and 'flux_ratio' in actual_params:
        axs[row, col].axhline(y=actual_params['flux_ratio'], color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["flux_ratio"]:.4f}')
    elif param == 'delta_t' and 'time_delay' in actual_params:
        axs[row, col].axhline(y=actual_params['time_delay'], color='green', linestyle='--', alpha=0.7,
                              label=f'Actual: {actual_params["time_delay"]:.4f} s')

    axs[row, col].legend()

plt.tight_layout()
title = 'Parameter Convergence'
if balanced_pso:
    title = 'Balanced Dimension PSO ' + title
if use_bayesian:
    title = 'Bayesian ' + title
plt.suptitle(title, y=1.02)
plt.savefig('parameter_convergence.png')
plt.close()

# 在一个图上显示所有运行的信号
fig = plt.figure(figsize=(15, 10), dpi=200)
ax = fig.add_subplot(111)

# 将数据转换为CPU进行绘图
dataY_real_np = cp.asnumpy(cp.real(dataY))  # 使用实部
dataY_only_signal_np = cp.asnumpy(cp.real(dataY_only_signal))  # 纯信号部分

# 绘制观测数据和纯信号数据
ax.scatter(t_cpu, dataY_real_np, marker='.', s=1, color='gray', alpha=0.3, label='Observed Data (Signal + Noise)')
ax.plot(t_cpu, dataY_only_signal_np, color='black', lw=1.0, label='Actual Signal (No Noise)')

# 绘制所有估计的信号
colors = plt.cm.tab10(np.linspace(0, 1, nRuns))
for lpruns in range(nRuns):
    # 获取信号的实部并转移到CPU
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))

    # 添加标签显示分类结果
    classification = outResults['allRunsOutput'][lpruns]['classification']
    snr_pycbc = outResults['allRunsOutput'][lpruns]['SNR_pycbc']

    # 确保SNR是Python浮点数
    if isinstance(snr_pycbc, cp.ndarray):
        snr_pycbc = float(snr_pycbc.get())

    # 仅绘制SNR > 8的运行结果，避免图表过于拥挤
    if snr_pycbc > 8:
        ax.plot(t_cpu, est_sig, color=colors[lpruns], lw=0.8,
                label=f'Run {lpruns + 1} ({classification}, SNR={snr_pycbc:.1f})')

# 突出显示最佳信号
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
best_classification = outResults['classification']
best_snr = outResults['allRunsOutput'][outResults['bestRun']]['SNR_pycbc']
# 确保SNR是Python浮点数
if isinstance(best_snr, cp.ndarray):
    best_snr = float(best_snr.get())
best_run_number = outResults['bestRun'] + 1
ax.plot(t_cpu, best_sig, 'red', lw=2.0,
        label=f'Best Fit (Run {best_run_number}, {best_classification}, SNR={best_snr:.1f})')

# 计算相关系数并添加到图表
def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

corr_coef = correlation_coefficient(dataY_only_signal_np, best_sig)
title = f'All PSO Runs Comparison (Best Run {best_run_number}, Correlation: {corr_coef:.4f})'
if balanced_pso:
    title = f'Balanced Dimension PSO: {title}'
if use_bayesian:
    title = f'Bayesian Approach: {title}'
ax.set_title(title)

# 设置标签和图例
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')

# 保存图表
plt.savefig('all_pso_runs_comparison.png')
plt.close()

# 处理最佳运行结果进行可视化
best_run_idx = outResults['bestRun']
bestSig_real = cp.real(outResults['bestSig'])

# 计算SNR和不匹配度
best_snr_pycbc = calculate_matched_filter_snr(bestSig_real, dataY_only_signal, psdHigh, Fs)
# 确保SNR是Python浮点数
if isinstance(best_snr_pycbc, cp.ndarray):
    best_snr_pycbc = float(best_snr_pycbc.get())
best_epsilon = analyze_mismatch(bestSig_real, dataY_only_signal, Fs, psdHigh)
# 确保epsilon是Python浮点数
if isinstance(best_epsilon, cp.ndarray):
    best_epsilon = float(best_epsilon.get())

# 提取最佳运行的参数
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # A参数作为透镜放大比例
if isinstance(best_flux_ratio, cp.ndarray):
    best_flux_ratio = float(best_flux_ratio.get())
best_time_delay = outResults['allRunsOutput'][best_run_idx]['delta_t']  # 延迟时间（秒）
if isinstance(best_time_delay, cp.ndarray):
    best_time_delay = float(best_time_delay.get())

# 计算最佳估计与实际信号之间的相关性
best_corr = correlation_coefficient(
    cp.asnumpy(cp.real(dataY_only_signal)),
    cp.asnumpy(bestSig_real)
)

# 打印结果
print('\n============= Final Results =============')
if balanced_pso:
    print("Using Balanced Dimension PSO")
if use_bayesian:
    print("Using Bayesian Approach with physically motivated priors")
print(f"Best Fitness (Inner Product): {outResults['bestFitness']:.4f}")
print(f"PyCBC SNR (Independently Calculated): {best_snr_pycbc:.2f}")
print(f"Correlation with actual signal: {best_corr:.4f}")
print(f"r : {10 ** float(outResults['allRunsOutput'][outResults['bestRun']]['r']):.4f} Mpc")
print(f"Mc: {10 ** float(outResults['allRunsOutput'][outResults['bestRun']]['m_c']):.4f} M_sun")
print(f"tc: {float(outResults['allRunsOutput'][outResults['bestRun']]['tc']):.4f} s")
print(f"phi_c: {float(outResults['allRunsOutput'][outResults['bestRun']]['phi_c']) / np.pi:.4f} π")
print(f"A: {float(outResults['allRunsOutput'][outResults['bestRun']]['A']):.4f}")
print(f"delta_t: {float(outResults['allRunsOutput'][outResults['bestRun']]['delta_t']):.4f} s")

# 打印透镜分析结果
print(f"\n============= Lensing Analysis =============")
print(f"Two-step Matching Result: {outResults['lensing_message']}")
print(f"Classification: {outResults['classification']}")
print(f"Is Lensed: {outResults['is_lensed']}")
print(f"Unlensed Mismatch: {outResults['allRunsOutput'][best_run_idx].get('unlensed_mismatch', best_epsilon):.6f}")
if outResults['is_lensed']:
    print(f"Lensed Mismatch: {outResults['allRunsOutput'][best_run_idx].get('lensed_mismatch', 'Not calculated')}")
print(f"Matched Filter SNR: {best_snr_pycbc:.2f}")

# 添加与实际参数的比较
if 'actual_comparison' in outResults and outResults['actual_comparison']:
    print("\n============= Comparison with Actual Values (For Evaluation Only) =============")
    print(f"Actual signal is {'lensed' if outResults['actual_comparison']['actual_is_lensed'] else 'unlensed'}")
    print(f"Estimated signal is {'lensed' if outResults['actual_comparison']['estimated_is_lensed'] else 'unlensed'}")
    print(f"Classification matches actual: {outResults['actual_comparison']['classification_matches_actual']}")

    # 遍历所有参数进行比较
    for param_name, param_data in outResults['actual_comparison']['parameters'].items():
        if param_name in ['r', 'm_c']:
            # 对数参数需要转换
            est_val = 10 ** float(param_data['estimated'])
            act_val = param_data['actual']
            print(f"{param_name}: Estimated={est_val:.4f}, Actual={act_val:.4f}, " +
                  f"Error={(est_val - act_val) / act_val * 100:.2f}%")
        else:
            est_val = float(param_data['estimated'])
            act_val = param_data.get('actual', param_data.get('actual_radians', 0))
            print(f"{param_name}: Estimated={est_val:.4f}, Actual={act_val:.4f}, " +
                  f"Error={(est_val - act_val) / act_val * 100 if act_val != 0 else 0:.2f}%")

# 如果有维度分析结果，打印维度探索情况
if 'best_dimension_analysis' in outResults:
    print("\n============= Dimension Exploration Analysis =============")
    dim_analysis = outResults['best_dimension_analysis']

    # 打印总体维度平衡情况
    if 'overall' in dim_analysis:
        overall = dim_analysis['overall']
        print(f"Mean Coverage: {overall['mean_coverage']:.2f}")
        print(f"Coverage Imbalance: {overall['coverage_imbalance']:.2f}")
        print(f"Mean Variance: {overall['mean_variance']:.4f}")
        print(f"Variance Imbalance: {overall['variance_imbalance']:.2f}")

    # 打印每个维度的探索情况
    for dim in ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']:
        if dim in dim_analysis:
            info = dim_analysis[dim]
            print(f"\nDimension {dim}:")
            print(f"  Coverage: {info['coverage']:.2f}")
            print(f"  Variance: {info['variance']:.4f}")
            print(f"  Concentration: {info['concentration']:.2f}")
            print(f"  Range: [{info['min']:.2f}, {info['max']:.2f}]")

# 如果使用了自适应策略，打印策略转换
if 'strategy_shifts' in outResults:
    print("\n============= Adaptive Strategy Shifts =============")
    shifts = outResults['strategy_shifts']
    for i, shift in enumerate(shifts):
        print(f"Shift {i + 1} at iteration {shift.get('iteration', 'N/A')}:")
        print(f"  Old strategy: {shift.get('old_strategy', 'N/A')}")
        print(f"  New strategy: {shift.get('new_strategy', 'N/A')}")
        print(f"  Exploration score: {shift.get('exploration_score', 'N/A')}")
        print(f"  Fitness improvement: {shift.get('fitness_improvement', 'N/A')}")
        print(f"  Reason: {shift.get('reason', 'N/A')}")

# 最终比较图 - 转换为CPU进行绘图
bestData_cpu = cp.asnumpy(cp.real(dataY))
bestSig_cpu = cp.asnumpy(cp.real(outResults['bestSig']))
bestSignal_only_cpu = cp.asnumpy(cp.real(dataY_only_signal))

# 增强的最终比较图，显示运行编号和是否透镜化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

# 顶部图：完整数据比较
ax1.plot(t_cpu, bestData_cpu, 'gray', alpha=0.5, label='Observed Data (Signal + Noise)')
ax1.plot(t_cpu, bestSig_cpu, 'r', label=f'Best Estimated Signal (Run {best_run_number})')
ax1.plot(t_cpu, bestSignal_only_cpu, 'b', alpha=0.7, label='Actual Signal (No Noise)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Strain')
ax1.legend()
title = f'Signal Comparison: {outResults["lensing_message"]}'
if balanced_pso:
    title = f'Balanced Dimension PSO: {title}'
if use_bayesian:
    title = f'Bayesian Approach: {title}'
ax1.set_title(title)

# 底部图：残差分析
residual = bestSig_cpu - bestSignal_only_cpu
ax2.plot(t_cpu, residual, 'g', label='Residual (Estimate - Actual)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Strain Difference')
ax2.legend()
ax2.set_title(f'Residual Analysis (Correlation: {best_corr:.4f}, SNR: {best_snr_pycbc:.2f})')
plt.tight_layout()
plt.savefig('signal_comparison_plot.png')
plt.close()

# 初始化存储所有结果的列表
all_results = []

# 为每次运行添加更详细的分析指标
for lpruns in range(nRuns):
    # 获取该运行的参数
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])

    # 获取SNR值
    run_snr_optimal = -float(outStruct[lpruns]['bestFitness'])  # 基于优化的SNR
    run_snr_pycbc = outResults['allRunsOutput'][lpruns]['SNR_pycbc']  # 已计算的SNR
    if isinstance(run_snr_pycbc, cp.ndarray):
        run_snr_pycbc = float(run_snr_pycbc.get())

    # 确保所有参数都是Python类型
    run_mass = 10 ** float(outResults['allRunsOutput'][lpruns]['m_c']) * M_sun
    run_flux_ratio = float(outResults['allRunsOutput'][lpruns]['A'])
    run_time_delay = float(outResults['allRunsOutput'][lpruns]['delta_t'])

    # 计算该运行的不匹配度
    run_epsilon = analyze_mismatch(run_sig, dataY_only_signal, Fs, psdHigh)
    if isinstance(run_epsilon, cp.ndarray):
        run_epsilon = float(run_epsilon.get())

    # 计算与实际信号的相关性
    run_corr = correlation_coefficient(
        cp.asnumpy(cp.real(dataY_only_signal)),
        cp.asnumpy(run_sig)
    )

    # 打印每次运行的分类结果
    print(f"\nRun {lpruns + 1} Results:")
    print(f"  Classification: {outResults['allRunsOutput'][lpruns]['classification']}")
    print(f"  Two-step Matching Result: {outResults['allRunsOutput'][lpruns]['lensing_message']}")
    print(f"  Is Lensed: {outResults['allRunsOutput'][lpruns]['is_lensed']}")
    print(f"  A value: {run_flux_ratio:.4f}")
    print(f"  delta_t value: {run_time_delay:.4f}")
    print(f"  SNR: {run_snr_pycbc:.2f}, Correlation: {run_corr:.4f}")
    print(f"  Mismatch: {run_epsilon:.6f}")

    # 检查是否有与实际参数的比较结果
    has_actual_comparison = 'actual_comparison' in outResults['allRunsOutput'][lpruns]
    if has_actual_comparison:
        actual_comp = outResults['allRunsOutput'][lpruns]['actual_comparison']
        is_correct = actual_comp.get('classification_matches_actual', False)
        print(f"  Classification correct: {is_correct}")

    # 添加到结果 - 确保所有值都是Python类型而不是CuPy数组
    run_result = {
        'run': lpruns + 1,
        'fitness': float(outStruct[lpruns]['bestFitness']),
        'r': float(10 ** outResults['allRunsOutput'][lpruns]['r']),
        'm_c': float(10 ** outResults['allRunsOutput'][lpruns]['m_c']),
        'tc': float(outResults['allRunsOutput'][lpruns]['tc']),
        'phi_c': float(outResults['allRunsOutput'][lpruns]['phi_c']) / np.pi,
        'A': float(outResults['allRunsOutput'][lpruns]['A']),
        'delta_t': float(outResults['allRunsOutput'][lpruns]['delta_t']),
        'SNR_optimal': float(run_snr_optimal),
        'SNR_pycbc': float(run_snr_pycbc),
        'correlation': float(run_corr),
        'mismatch': float(run_epsilon),
        'two_step_match_result': outResults['allRunsOutput'][lpruns]['lensing_message'],
        'two_step_is_lensed': outResults['allRunsOutput'][lpruns]['is_lensed'],
        'classification': outResults['allRunsOutput'][lpruns]['classification'],
        'bayesian_used': use_bayesian
    }

    # 添加平衡维度PSO信息
    if balanced_pso:
        run_result['balanced_pso'] = True

    # 如果有维度分析，添加维度探索指标
    if 'dimension_analysis' in outResults['allRunsOutput'][lpruns]:
        dim_analysis = outResults['allRunsOutput'][lpruns]['dimension_analysis']
        if 'overall' in dim_analysis:
            run_result.update({
                'mean_dimension_coverage': dim_analysis['overall']['mean_coverage'],
                'dimension_coverage_imbalance': dim_analysis['overall']['coverage_imbalance'],
                'dimension_exploration_score': dim_analysis['overall']['mean_coverage'] / (
                            1 + dim_analysis['overall']['coverage_imbalance'])
            })

    # 添加与实际参数比较的指标（如果可用）
    if has_actual_comparison:
        # 确保错误值是Python浮点数
        def get_error_value(param_name):
            value = outResults['allRunsOutput'][lpruns].get('param_errors', {}).get(param_name, 0)
            if isinstance(value, cp.ndarray):
                return float(value.get()) * 100
            return float(value) * 100


        run_result.update({
            'classification_correct': is_correct,
            'actual_is_lensed': actual_comp.get('actual_is_lensed', False),
            'parameter_r_error': get_error_value('r_error'),
            'parameter_m_c_error': get_error_value('m_c_error'),
            'parameter_tc_error': get_error_value('tc_error'),
            'parameter_A_error': get_error_value('A_error'),
            'parameter_delta_t_error': get_error_value('delta_t_error')
        })

    all_results.append(run_result)

# 添加最佳结果作为摘要条目（标记为"best"）
best_result = {
    'run': 'best',
    'fitness': float(outResults['bestFitness']),
    'r': float(10 ** outResults['allRunsOutput'][best_run_idx]['r']),
    'm_c': float(10 ** outResults['allRunsOutput'][best_run_idx]['m_c']),
    'tc': float(outResults['allRunsOutput'][best_run_idx]['tc']),
    'phi_c': float(outResults['allRunsOutput'][best_run_idx]['phi_c']) / np.pi,
    'A': float(outResults['allRunsOutput'][best_run_idx]['A']),
    'delta_t': float(outResults['allRunsOutput'][best_run_idx]['delta_t']),
    'SNR_pycbc': float(best_snr_pycbc),
    'correlation': float(best_corr),
    'mismatch': float(best_epsilon),
    'two_step_match_result': outResults['lensing_message'],
    'two_step_is_lensed': outResults['is_lensed'],
    'classification': outResults['classification'],
    'bayesian_used': use_bayesian
}

# 添加平衡维度PSO信息到最佳结果
if balanced_pso:
    best_result['balanced_pso'] = True

    # 如果有最佳维度分析，添加维度探索分析指标
    if 'best_dimension_analysis' in outResults:
        dim_analysis = outResults['best_dimension_analysis']
        if 'overall' in dim_analysis:
            best_result.update({
                'mean_dimension_coverage': dim_analysis['overall']['mean_coverage'],
                'dimension_coverage_imbalance': dim_analysis['overall']['coverage_imbalance']
            })

# 添加与实际参数比较的指标（如果可用）
if 'actual_comparison' in outResults:
    # 确保错误值是Python浮点数
    def get_param_error(param_name):
        value = outResults.get('param_errors', {}).get(param_name, 0)
        if isinstance(value, cp.ndarray):
            return float(value.get()) * 100
        return float(value) * 100


    best_result.update({
        'classification_correct': outResults['actual_comparison'].get('classification_matches_actual', False),
        'actual_is_lensed': outResults['actual_comparison'].get('actual_is_lensed', False),
        'parameter_r_error': get_param_error('r_error'),
        'parameter_m_c_error': get_param_error('m_c_error'),
        'parameter_tc_error': get_param_error('tc_error'),
        'parameter_A_error': get_param_error('A_error'),
        'parameter_delta_t_error': get_param_error('delta_t_error')
    })

all_results.append(best_result)

# 定义CSV列 - 增加更多列以包含与实际参数比较的指标
columns = [
    'run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t',
    'SNR_pycbc', 'correlation', 'mismatch',
    'two_step_match_result', 'two_step_is_lensed', 'classification',
    'bayesian_used'
]

# 使用pandas保存为CSV以获得更好的格式
df = pd.DataFrame(all_results)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)


# 创建一个单独的参数比较CSV
# 确保所有值都是Python浮点数
def get_estimated_value(param_name):
    if param_name == 'm_c':
        value = outResults['m_c']
        if isinstance(value, cp.ndarray):
            return 10 ** float(value.get())
        return 10 ** float(value)
    elif param_name == 'r':
        value = outResults['r']
        if isinstance(value, cp.ndarray):
            return 10 ** float(value.get())
        return 10 ** float(value)
    elif param_name == 'phi_c':
        value = outResults['phi_c']
        if isinstance(value, cp.ndarray):
            return float(value.get()) / (2 * np.pi)
        return float(value) / (2 * np.pi)
    else:
        value = outResults[param_name]
        if isinstance(value, cp.ndarray):
            return float(value.get())
        return float(value)


comparison_df = pd.DataFrame({
    'Parameter': ['Chirp Mass (M⊙)', 'Merger Time (s)', 'Source Distance (Mpc)',
                  'Flux Ratio (A)', 'Time Delay (s)', 'Phase (rad/2π)'],
    'Actual Value': [
        actual_params['chirp_mass'],
        actual_params['merger_time'],
        actual_params['source_distance'],
        actual_params['flux_ratio'],
        actual_params['time_delay'],
        actual_params['phase']
    ],
    'Estimated Value': [
        get_estimated_value('m_c'),
        get_estimated_value('tc'),
        get_estimated_value('r'),
        get_estimated_value('A'),
        get_estimated_value('delta_t'),
        get_estimated_value('phi_c')
    ],
    'Relative Error (%)': [
        abs(get_estimated_value('m_c') - actual_params['chirp_mass']) / actual_params['chirp_mass'] * 100,
        abs(get_estimated_value('tc') - actual_params['merger_time']) / actual_params['merger_time'] * 100,
        abs(get_estimated_value('r') - actual_params['source_distance']) / actual_params['source_distance'] * 100,
        abs(get_estimated_value('A') - actual_params['flux_ratio']) / actual_params['flux_ratio'] * 100,
        abs(get_estimated_value('delta_t') - actual_params['time_delay']) / actual_params['time_delay'] * 100,
        min(abs(get_estimated_value('phi_c') - actual_params['phase']),
            abs(get_estimated_value('phi_c') - actual_params['phase'] - 1)) * 100
    ]
})

# 添加PSO方法信息
method_str = ""
if balanced_pso:
    method_str += "Balanced Dimension "
if use_bayesian:
    method_str += "Bayesian "
method_str += "PSO"

comparison_df['Method'] = method_str

# 保存参数比较表
comparison_df.to_csv('parameter_comparison.csv', index=False)

if balanced_pso and use_bayesian:
    print(f"\n本次贝叶斯平衡维度PSO飞行结束，使用物理先验和自适应优化策略")
elif balanced_pso:
    print(f"\n本次平衡维度PSO飞行结束，确保6维参数空间的充分搜索")
elif use_bayesian:
    print(f"\n本次贝叶斯PSO飞行结束，使用物理先验信息")
else:
    print(f"\n本次标准PSO飞行结束")
print(f"飞行结果报告已保存在：{csv_filename}和parameter_comparison.csv")