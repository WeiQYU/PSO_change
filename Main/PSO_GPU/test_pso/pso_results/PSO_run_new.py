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
# Load data
TrainingData = scio.loadmat('../../generate_ligo/noise.mat')
analysisData = scio.loadmat('../../generate_ligo/data.mat')
print("加载完毕")

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
training_noise = cp.asarray(TrainingData['noise'][0])  # 移动到GPU
dataY_only_signal = dataY - training_noise  # 提取信号部分（用于比较）

nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])
# 搜索范围参数
#                r  mc tc phi A  Δtd
rmin = cp.array([1, 0, 0, 0, 0, 0])  # parameter range lower bounds
rmax = cp.array([4, 3, 8, np.pi, 1, 7])  # parameter range upper bounds
# 时间域设置
dt = 1 / Fs  # sampling rate Hz
t = cp.arange(0, 8, dt)  # Using CuPy for t array
T = nSamples / Fs
df = 1 / T
Nyq = Fs / 2
# 计算PSD
psdHigh = cp.asarray(TrainingData['psd'][0])  # 直接从训练数据获取PSD

# PSO输入参数 - 保持所有内容在GPU上
inParams = {
    'dataX': t,
    'dataY': dataY,
    'sampFreq': Fs,
    'psdHigh': psdHigh,
    'rmin': rmin,
    'rmax': rmax,
}

# PSO运行次数
nRuns = 8

# PSO参数配置
pso_config = {
    'popsize': 80,  # 种群大小
    'maxSteps': 3000,  # 迭代次数
    'c1': 2,  # 个体学习因子
    'c2': 2,  # 社会学习因子
    'w_start': 0.9,  # 初始惯性权重
    'w_end': 0.4,  # 最终惯性权重
    'max_velocity': 0.5,  # 最大速度限制
    'nbrhdSz': 5  # 邻域大小
}

print("PSO已部署完毕,芜湖！！！！")
# 运行PSO优化，启用两步匹配过程
outResults, outStruct = crcbqcpsopsd(inParams, pso_config, nRuns, use_two_step=True)

# 新增：在每次PSO运行结束后输出匹配结果
print("\n============= 每次PSO运行的匹配结果 =============")
for lpruns in range(nRuns):
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])
    run_snr_pycbc = calculate_snr_pycbc(run_sig, psdHigh, Fs)

    print(f"\n运行 {lpruns + 1}:")
    if outResults['allRunsOutput'][lpruns]['is_noise']:
        print(f"  消息: {outResults['allRunsOutput'][lpruns]['lensing_message']}")
        print(f"  SNR: {run_snr_pycbc:.4f} (小于阈值8)")
        print(f"  状态: 噪声")
    else:
        run_epsilon = analyze_mismatch(run_sig, dataY_only_signal, Fs, psdHigh)
        mismatch_threshold = 1.0 / run_snr_pycbc
        print(f"  消息: {outResults['allRunsOutput'][lpruns]['lensing_message']}")
        print(f"  SNR: {run_snr_pycbc:.4f}")
        print(f"  失配度 (Mismatch): {run_epsilon:.6f}")
        print(f"  阈值 (Threshold): {mismatch_threshold:.6f}")

        if outResults['allRunsOutput'][lpruns]['is_lensed']:
            print(f"  状态: 透镜信号")
        else:
            print(f"  状态: 非透镜信号")

# 对于绘图，我们需要将数据移回CPU
# 只在需要可视化时进行转换
t_cpu = cp.asnumpy(t)

# 绘制每次运行的PSO收敛情况
plt.figure(figsize=(12, 8), dpi=200)
for lpruns in range(nRuns):
    if 'fitnessHistory' in outStruct[lpruns]:
        plt.plot(outStruct[lpruns]['fitnessHistory'], label=f'Run {lpruns + 1}')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value (SNR)')
plt.title('PSO Fitness History Across Iterations')
plt.legend()
plt.grid(True)
plt.savefig('pso_convergence_plot.png')
plt.close()

# 在一个图上显示所有运行的信号
fig = plt.figure(figsize=(15, 10), dpi=200)
ax = fig.add_subplot(111)

# 将数据转换为CPU以进行绘图
dataY_real_np = cp.asnumpy(cp.real(dataY))  # 使用实部进行绘图

# 绘制观测数据
ax.scatter(t_cpu, dataY_real_np, marker='.', s=1, color='gray', alpha=0.5, label='Observed Data')

# 使用不同颜色绘制所有估计信号
colors = plt.cm.tab10(np.linspace(0, 1, nRuns))
for lpruns in range(nRuns):
    # 获取估计信号的实部并转移到CPU进行绘图
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))

    # 添加标签以显示信号类型
    if outResults['allRunsOutput'][lpruns]['is_noise']:
        signal_status = "Noise"
    elif outResults['allRunsOutput'][lpruns]['is_lensed']:
        signal_status = "Lensed"
    else:
        signal_status = "Unlensed"

    ax.plot(t_cpu, est_sig, color=colors[lpruns], lw=0.8,
            label=f'Run {lpruns + 1} ({signal_status})')

# 突出显示最佳信号
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
if outResults['is_noise']:
    best_signal_status = "Noise"
elif outResults['is_lensed']:
    best_signal_status = "Lensed"
else:
    best_signal_status = "Unlensed"

ax.plot(t_cpu, best_sig, 'red', lw=1.5,
        label=f'Best Fit (Run {outResults["bestRun"] + 1}, {best_signal_status})')

# 设置标签和图例
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')
plt.title('All PSO Runs Comparison')

# 保存图表
plt.savefig('all_pso_runs_comparison.png')
plt.close()

# 保留原始单图功能
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

# 绘制观测数据
ax.scatter(t_cpu, dataY_real_np, marker='.', s=5, label='Observed Data')

# 绘制所有估计信号
for lpruns in range(nRuns):
    # 获取估计信号的实部并转移到CPU进行绘图
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t_cpu, est_sig,
            color=[51 / 255, 255 / 255, 153 / 255],
            lw=0.8,
            alpha=0.5,
            label='Estimated Signal' if lpruns == 0 else "_nolegend_")

# 突出显示最佳信号
ax.plot(t_cpu, best_sig, 'red', lw=0.4, label='Best Fit')

# 设置标签和图例
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')

# 保存图表
plt.savefig('pso_optimization_results.png')
plt.close()

# 首先处理最佳运行进行可视化
best_run_idx = outResults['bestRun']
bestSig_real = cp.real(outResults['bestSig'])

# 计算SNR
best_snr_pycbc = calculate_snr_pycbc(bestSig_real, psdHigh, Fs)

# 使用PyCBC计算失配度（如果不是噪声）
if not outResults['is_noise']:
    best_epsilon = analyze_mismatch(bestSig_real, dataY_only_signal, Fs, psdHigh)

# 从最佳运行中提取参数
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # 将振幅A作为透镜振幅比例
best_time_delay = outResults['allRunsOutput'][best_run_idx]['delta_t']  # 单位：秒

# 打印结果
print('\n============= 最终结果 =============')
print(f"最佳适应度（内积结果）: {outResults['bestFitness']:.4f}")
print(f"PyCBC SNR（独立计算）: {best_snr_pycbc:.2f}")
print(f"r : {10 ** outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}")
print(f"Mc: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}")
print(f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}")
print(f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c'] / np.pi:.4f}")
print(f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}")
print(f"delta_t: {outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")

# 打印分类结果
print(f"\n============= 信号分类 =============")
if outResults['is_noise']:
    print(f"分类结果: 噪声 (SNR < 8)")
    print(f"SNR: {best_snr_pycbc:.2f} (低于阈值8)")
else:
    # 打印两步匹配结果
    print(f"两步匹配结果: {outResults['lensing_message']}")
    print(f"是否为透镜波形: {outResults['is_lensed']}")
    print(f"失配度: {best_epsilon:.6f}")
    print(f"失配度阈值 (1/SNR): {1 / best_snr_pycbc:.6f}")
    print(f"变化率: {best_flux_ratio:.4f}, 时间延迟: {best_time_delay:.4f} s")

# 最终比较图 - 仅为绘图转换为CPU
bestData_cpu = cp.asnumpy(cp.real(dataY))
bestSig_cpu = cp.asnumpy(cp.real(outResults['bestSig']))

# 绘制最佳信号与数据对比图
fig = plt.figure(figsize=(20, 8))
plt.plot(t_cpu, bestData_cpu, 'gray', alpha=0.5, label='Observed Data')
plt.plot(t_cpu, bestSig_cpu, 'r', label='Best Signal')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()

# 根据信号类型设置标题
if outResults['is_noise']:
    plt.title(f'Best Signal Comparison (Noise): SNR = {best_snr_pycbc:.2f} < 8')
else:
    signal_type = "Lensed" if outResults['is_lensed'] else "Unlensed"
    plt.title(f'Best Signal Comparison ({signal_type}): {outResults["lensing_message"]}')

# 保存最终比较图
plt.savefig('signal_comparison_plot.png')
plt.close()

# 初始化列表以存储所有结果
all_results = []

# 分析所有运行并添加到结果中
for lpruns in range(nRuns):
    # 获取此运行的参数
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])

    # 计算SNR
    run_snr_optimal = -float(outStruct[lpruns]['bestFitness'])  # 基于优化的SNR（取负值）
    run_snr_pycbc = calculate_snr_pycbc(run_sig, psdHigh, Fs)  # PyCBC方法SNR

    run_mass = 10 ** outResults['allRunsOutput'][lpruns]['m_c'] * M_sun
    run_flux_ratio = outResults['allRunsOutput'][lpruns]['A']
    run_time_delay = outResults['allRunsOutput'][lpruns]['delta_t']

    # 计算此运行的失配度（如果不是噪声）
    if not outResults['allRunsOutput'][lpruns]['is_noise']:
        run_epsilon = analyze_mismatch(run_sig, dataY_only_signal, Fs, psdHigh)
    else:
        run_epsilon = None

    # 失配度阈值
    mismatch_threshold = 1.0 / run_snr_pycbc

    # 添加到结果中 - 确保所有值都是Python类型，而不是CuPy数组
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
        'is_noise': outResults['allRunsOutput'][lpruns]['is_noise'],
        'mismatch': float(run_epsilon) if run_epsilon is not None else None,
        'mismatch_threshold': float(mismatch_threshold),
        'two_step_match_result': outResults['allRunsOutput'][lpruns]['lensing_message'],
        'two_step_is_lensed': outResults['allRunsOutput'][lpruns]['is_lensed'],
    }
    all_results.append(run_result)

# 将最佳结果添加为汇总条目（标记为"best"）
best_mismatch = best_epsilon if not outResults['is_noise'] else None
best_result = {
    'run': 'best',
    'fitness': float(outResults['bestFitness']),
    'r': float(10 ** outResults['allRunsOutput'][best_run_idx]['r']),
    'm_c': float(10 ** outResults['allRunsOutput'][best_run_idx]['m_c']),
    'tc': float(outResults['allRunsOutput'][best_run_idx]['tc']),
    'phi_c': float(outResults['allRunsOutput'][best_run_idx]['phi_c'] / np.pi),
    'A': float(outResults['allRunsOutput'][best_run_idx]['A']),
    'delta_t': float(outResults['allRunsOutput'][best_run_idx]['delta_t']),
    'SNR_pycbc': float(best_snr_pycbc),
    'is_noise': outResults['is_noise'],
    'mismatch': float(best_mismatch) if best_mismatch is not None else None,
    'mismatch_threshold': float(1 / best_snr_pycbc),
    'two_step_match_result': outResults['lensing_message'],
    'two_step_is_lensed': outResults['is_lensed'],
}
all_results.append(best_result)

# 定义CSV的列
columns = ['run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t',
           'SNR_pycbc', 'is_noise', 'mismatch', 'mismatch_threshold',
           'two_step_match_result', 'two_step_is_lensed']

# 使用pandas保存为CSV以便更好的格式化
df = pd.DataFrame(all_results, columns=columns)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\n本次飞行结束，飞行结果报告已保存在：{csv_filename}")