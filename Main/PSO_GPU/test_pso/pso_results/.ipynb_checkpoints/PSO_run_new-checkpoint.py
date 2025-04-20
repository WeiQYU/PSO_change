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

# 测试模式选择
print("选择功能:")
print("1、零检验 \t    2、信号搜寻")
test_mode = int(input("Input Num: "))

# 注入参数 - 用于计算误差
injection_params = {
    'm1': 36.0,  # 太阳质量
    'm2': 29.0,  # 太阳质量
    'mc': 28.10,  # 啁啾质量（太阳质量）
    'distance': 3100.0,  # 距离（pc）
    'lens_mass': 1.0e7,  # 透镜质量（太阳质量）
    'y': 0.5,  # y参数
    'flux_ratio': 0.3333,  # 通量比
    'time_delay': 1.0,  # 时间延迟（秒）
    'tc': 6.5,  # 合并时间（秒）
    'phi_c': 0.5 * np.pi,  # 相位
}

# 将注入参数转换为搜索参数格式
r_inj = np.log10(injection_params['distance'])
m_c_inj = np.log10(injection_params['mc'])
tc_inj = injection_params['tc']
phi_c_inj = injection_params['phi_c']
A_inj = injection_params['flux_ratio']
delta_t_inj = injection_params['time_delay']

# 根据测试模式调整参数
if test_mode == 1:
    print("运行零检验模式（模式1）：未透镜信号")
    analysisData = scio.loadmat('../../generate_ligo/data_without_lens.mat')
else:
    print("运行信号搜寻模式（模式2）：透镜化信号")
    analysisData = scio.loadmat('../../generate_ligo/data.mat')

print("加载数据...")
# Load data
TrainingData = scio.loadmat('../../generate_ligo/noise.mat')
print("加载完毕")

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])

# Search range parameters
#                r  mc tc phi A  Δtd
rmin = cp.array([-2, 0, 0, 0, 0, 0])  # parameter range lower bounds
rmax = cp.array([4, 3, 8, 2 * np.pi, 1, 4])  # parameter range upper bounds

# Time domain setup
dt = 1 / Fs  # sampling rate Hz
t = cp.arange(0, 8, dt)  # Using CuPy for t array
T = nSamples / Fs
df = 1 / T
Nyq = Fs / 2
# GPU-optimized PSD calculations
training_noise = cp.asarray(TrainingData['noise'][0])  # Move noise to GPU
psdHigh = cp.asarray(TrainingData['psd'][0])  # Get PSD data directly on GPU

# PSO input parameters - keep everything on GPU
inParams = {
    'dataX': t,
    'dataY': dataY,
    'sampFreq': Fs,
    'psdHigh': psdHigh,
    'rmin': rmin,
    'rmax': rmax,
}

# Number of PSO runs
nRuns = 8

# PSO parameters
pso_config = {
    'popsize': 80,  # population size
    'maxSteps': 3000,  # iterations
    'c1': 2,  # cognitive parameter
    'c2': 2,  # social parameter
    'w_start': 0.9,  # initial inertia weight
    'w_end': 0.4,  # final inertia weight
    'max_velocity': 0.5,  # maximum velocity limit
    'nbrhdSz': 5  # neighborhood size
}

print("PSO已部署完毕，请求起飞")
print("允许起飞！芜湖！！！！")
# Run PSO optimization
outResults, outStruct = crcbqcpsopsd(inParams, pso_config, nRuns)

# For plotting, we need to convert data back to CPU
# Only convert when needed for visualization
t_cpu = cp.asnumpy(t)

# Plot PSO convergence for each run
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

# New feature: show all runs' signals on one plot
fig = plt.figure(figsize=(15, 10), dpi=200)
ax = fig.add_subplot(111)

# Convert data to CPU for plotting
dataY_real_np = cp.asnumpy(cp.real(dataY))  # Use real part for plotting

# Plot observed data
ax.scatter(t_cpu, dataY_real_np, marker='.', s=1, color='gray', alpha=0.5, label='Observed Data')

# Plot all estimated signals with different colors
colors = plt.cm.tab10(np.linspace(0, 1, nRuns))
for lpruns in range(nRuns):
    # Get real part of estimated signal and transfer to CPU for plotting
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t_cpu, est_sig, color=colors[lpruns], lw=0.8, label=f'Run {lpruns + 1}')

# Highlight best signal
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
ax.plot(t_cpu, best_sig, 'red', lw=1.5, label='Best Fit (Run {})'.format(outResults['bestRun'] + 1))

# Set labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')
plt.title('All PSO Runs Comparison')

# Save plot
plt.savefig('all_pso_runs_comparison.png')
plt.close()

# Original single plot feature preserved
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

# Plot observed data
ax.scatter(t_cpu, dataY_real_np, marker='.', s=5, label='Observed Data')

# Plot all estimated signals
for lpruns in range(nRuns):
    # Get real part of estimated signal and transfer to CPU for plotting
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t_cpu, est_sig,
            color=[51 / 255, 255 / 255, 153 / 255],
            lw=0.8,
            alpha=0.5,
            label='Estimated Signal' if lpruns == 0 else "_nolegend_")

# Highlight best signal
ax.plot(t_cpu, best_sig, 'red', lw=0.4, label='Best Fit')

# Set labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')

# Save plot
plt.savefig('pso_optimization_results.png')
plt.close()


# Process best run first for visualization
best_run_idx = outResults['bestRun']
bestData_real = cp.real(outResults['bestSig'])

# Calculate SNRs
best_snr_optimal = outResults['bestFitness']
best_snr_pycbc = calculate_snr_pycbc(bestData_real, psdHigh, Fs)

# Calculate mismatch using PyCBC - 不再加噪声
best_epsilon = analyze_mismatch(bestData_real, dataY, Fs, psdHigh)

# Extract parameters from best run
best_total_mass = 10 ** outResults['allRunsOutput'][best_run_idx]['m_c'] * M_sun  # In kg
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # Using amplitude A as proxy for flux ratio
best_time_delay = outResults['allRunsOutput'][best_run_idx]['delta_t']  # In seconds

# Classify based on PyCBC SNR
best_classification, best_flux_threshold, best_timedelay_threshold, best_is_lensed = classify_signal(
    float(best_snr_pycbc), best_flux_ratio, best_time_delay, best_total_mass,best_epsilon)

# Print results
print('\n============= Final Results =============')
print(f"Best Fitness (negative SNR): {outResults['bestFitness']:.4f}")
print(f"Optimal SNR (from fitness): {best_snr_optimal:.2f}")
print(f"PyCBC SNR (calculated): {best_snr_pycbc:.2f}")
print(f"r : {10 ** outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}")
print(f"Mc: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}")
print(f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}")
print(f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c'] / np.pi:.4f}")
print(f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}")
print(f"delta_t: {outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")
print(f"Is Lensed: {best_is_lensed}")

print(f"\nBest Run Classification: {best_classification}")
print(f"Epsilon: {best_epsilon:.4f}")
print(f"Epsilon threshold (1/SNR^2): {1/(best_snr_pycbc**2):.6f}")
print(f"Mismatch criterion satisfied: {best_epsilon >= 1/(best_snr_pycbc**2)}")
print(f"Flux ratio: {best_flux_ratio:.4f}, Time delay: {best_time_delay:.4f} s")
print(f"Total mass: {best_total_mass / M_sun:.4f} M_sun")
print(f"Flux ratio threshold: {best_flux_threshold:.6f}")
print(f"Time delay threshold: {best_timedelay_threshold:.6f}")

# 添加注入参数比较
print("\n============= 注入参数与估计参数比较 =============")
print("参数            | 注入值      | 估计值      | 相对误差(%)")
print("-" * 60)
print(f"距离 (pc)       | {10**r_inj:.4f}    | {10**outResults['r']:.4f}    | {abs((10**outResults['r'] - 10**r_inj)/10**r_inj)*100:.2f}")
print(f"啁啾质量 (M☉)   | {10**m_c_inj:.4f}    | {10**outResults['m_c']:.4f}    | {abs((10**outResults['m_c'] - 10**m_c_inj)/10**m_c_inj)*100:.2f}")
print(f"合并时间 (s)    | {tc_inj:.4f}    | {outResults['tc']:.4f}    | {abs((outResults['tc'] - tc_inj)/tc_inj)*100:.2f}")
print(f"相位 (π)        | {phi_c_inj/np.pi:.4f}    | {outResults['phi_c']/np.pi:.4f}    | {abs((outResults['phi_c'] - phi_c_inj)/phi_c_inj)*100:.2f}")

# 避免除以零的情况
if A_inj != 0:
    print(f"通量比          | {A_inj:.4f}    | {outResults['A']:.4f}    | {abs((outResults['A'] - A_inj)/A_inj)*100:.2f}")
else:
    print(f"通量比          | {A_inj:.4f}    | {outResults['A']:.4f}    | N/A (注入值为0)")

if delta_t_inj != 0:
    print(f"时间延迟 (s)    | {delta_t_inj:.4f}    | {outResults['delta_t']:.4f}    | {abs((outResults['delta_t'] - delta_t_inj)/delta_t_inj)*100:.2f}")
else:
    print(f"时间延迟 (s)    | {delta_t_inj:.4f}    | {outResults['delta_t']:.4f}    | N/A (注入值为0)")

# Final comparison plots - convert to CPU only for plotting
bestSig_cpu = cp.asnumpy(cp.real(outResults['bestSig']))

fig = plt.figure(figsize=(20, 8))
plt.subplot(121)
plt.plot(t_cpu, dataY_real_np, label='data', alpha=0.75)
plt.plot(t_cpu, bestSig_cpu, label='bestSig')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Signal Comparison')

plt.subplot(122)
plt.plot(t_cpu, bestSig_cpu, label='bestSig')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Best Signal')

# Save the final comparison plot
plt.savefig('signal_comparison_plot.png')
plt.close()

# Initialize list to store all results
all_results = []

# Analyze all runs and add to results
for lpruns in range(nRuns):
    # Get parameters for this run
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])

    # Calculate SNRs
    run_snr_optimal = outStruct[lpruns]['bestFitness']  # Optimization-based SNR
    run_snr_pycbc = calculate_snr_pycbc(run_sig, psdHigh, Fs)  # PyCBC method SNR

    run_mass = 10 ** outResults['allRunsOutput'][lpruns]['m_c'] * M_sun
    run_flux_ratio = outResults['allRunsOutput'][lpruns]['A']
    run_time_delay = outResults['allRunsOutput'][lpruns]['delta_t']

    # Calculate mismatch for this run - 修正为不加噪声
    run_epsilon = analyze_mismatch(run_sig, dataY, Fs, psdHigh)

    # Classify this run with PyCBC SNR
    run_classification, run_flux_threshold, run_timedelay_threshold, run_is_lensed = classify_signal(
        float(run_snr_pycbc), run_flux_ratio, run_time_delay, run_mass,run_epsilon)

    # Print classification for each run
    print(f"\nRun {lpruns + 1} Classification: {run_classification}")
    print(f"  Optimal SNR: {run_snr_optimal:.2f}, PyCBC SNR: {run_snr_pycbc:.2f}")
    print(f"  Epsilon: {run_epsilon:.6f}, epsilon threshold: {1/(run_snr_pycbc**2):.6f}")
    print(f"  Flux ratio: {run_flux_ratio:.4f}, Time delay: {run_time_delay:.4f}")
    print(f"  Flux threshold: {run_flux_threshold:.6f}, time threshold: {run_timedelay_threshold:.6f}")
    print(f"  Is Lensed: {run_is_lensed}")

    # Add to results - ensure all values are Python types, not CuPy arrays
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
        'mismatch': float(run_epsilon),
        'flux_ratio_threshold': float(run_flux_threshold),
        'time_delay_threshold': float(run_timedelay_threshold),
        'classification': run_classification,
        'is_lensed': run_is_lensed
    }
    all_results.append(run_result)

# Add the best result as a summary entry (labeled as 'best')
best_result = {
    'run': 'best',
    'fitness': float(outResults['bestFitness']),
    'r': float(10 ** outResults['allRunsOutput'][best_run_idx]['r']),
    'm_c': float(10 ** outResults['allRunsOutput'][best_run_idx]['m_c']),
    'tc': float(outResults['allRunsOutput'][best_run_idx]['tc']),
    'phi_c': float(outResults['allRunsOutput'][best_run_idx]['phi_c'] / np.pi),
    'A': float(outResults['allRunsOutput'][best_run_idx]['A']),
    'delta_t': float(outResults['allRunsOutput'][best_run_idx]['delta_t']),
    'SNR_optimal': float(best_snr_optimal),
    'SNR_pycbc': float(best_snr_pycbc),
    'mismatch': float(best_epsilon),
    'flux_ratio_threshold': float(best_flux_threshold),
    'time_delay_threshold': float(best_timedelay_threshold),
    'classification': best_classification,
    'is_lensed': best_is_lensed
}
all_results.append(best_result)

# 添加注入参数作为参考值
injection_result = {
    'run': 'injection',
    'fitness': None,
    'r': float(10 ** r_inj),
    'm_c': float(10 ** m_c_inj),
    'tc': float(tc_inj),
    'phi_c': float(phi_c_inj / np.pi),
    'A': float(A_inj),
    'delta_t': float(delta_t_inj),
    'SNR_optimal': None,
    'SNR_pycbc': None,
    'mismatch': None,
    'flux_ratio_threshold': None,
    'time_delay_threshold': None,
    'classification': 'Injection Parameters',
    'is_lensed': True if A_inj > 0 and delta_t_inj > 0 else False
}
all_results.append(injection_result)

# Define the columns for our CSV
columns = ['run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t',
           'SNR_optimal', 'SNR_pycbc', 'mismatch', 'flux_ratio_threshold',
           'time_delay_threshold', 'classification', 'is_lensed']

# Save to CSV using pandas for better formatting
df = pd.DataFrame(all_results, columns=columns)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\n本次飞行结束，飞行结果报告已保存在：{csv_filename}")