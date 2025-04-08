import cupy as cp
import scipy.io as scio
from scipy.signal import welch
from PSO_main_new import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import pandas as pd


# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30
pc = 3.086e16

print("Loading data...")
# Load data
TrainingData = scio.loadmat('../../generate_ligo/noise.mat')
analysisData = scio.loadmat('../../generate_ligo/data')
print("Data loaded successfully.")

t0 = int(analysisData['t0'][0][0])  # Extract the scalar value properly
t1 = int(analysisData['t1'][0][0]) + 1  # Extract the scalar value properly
# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])
# Search range parameters
#                r  mc tc phi A  Δtd
rmin = cp.array([-2, 0, t0, 0, 0, -3])  # 对应参数范围下限
rmax = cp.array([4, 3, t1, 2 * np.pi, 1, 1])  # 对应参数范围上限

# Time domain setup
dt = 1 / Fs  # 采样率Hz
t = np.arange(0, 32, dt)
T = nSamples / Fs
df = 1 / T
Nyq = Fs / 2

print("Calculating PSD...")
# PSD estimation (CPU operation)
training_noise = TrainingData['noise'][0]  # Get noise data
psdHigh = cp.asarray(TrainingData['psd'][0])  # Get PSD data directly

# Calculate Welch periodogram for training noise
[f_noise, pxx_noise] = welch(training_noise, fs=Fs,
                             window='hamming', nperseg=int(Fs / 2),
                             noverlap=None, nfft=None,
                             detrend=False)

# Convert data to CPU for Welch on signal+noise
# Explicitly get real part to avoid complex warnings
dataY_real = cp.real(dataY)
dataY_cpu = cp.asnumpy(dataY_real)
[f_signal, pxx_signal] = welch(dataY_cpu, fs=Fs,
                               window='hamming', nperseg=int(Fs / 2),  # Using the same nperseg for consistency
                               noverlap=None, nfft=None,
                               detrend=False)

# Plot PSDs
plt.figure(dpi=200)
plt.plot(f_noise, pxx_noise, label='Noise (raw)')
plt.plot(f_signal, pxx_signal, label='Signal + Noise', alpha=0.7)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('PSD Comparison')
plt.savefig('psd_comparison_plot.png')
plt.close()

# PSO input parameters
inParams = {
    'dataX': t,
    'dataY': dataY,
    'sampFreq': Fs,
    'psdHigh': psdHigh,
    'rmin': rmin,
    'rmax': rmax,
}

# Number of PSO runs
nRuns = 4

# 修改后的PSO参数
pso_config = {
    'popsize': 50,  # 种群大小
    'maxSteps': 1500,  # 迭代次数
    'c1': 2,  # 认知参数
    'c2': 2,  # 社会参数
    'w_start': 0.9,  # 初始惯性权重
    'w_end': 0.4,  # 最终惯性权重
    'max_velocity': 0.5,  # 限制最大速度
    'nbrhdSz': 5  # 邻域大小
}

print("Running PSO optimization...")
# Run PSO optimization
outResults, outStruct = crcbqcpsopsd(inParams, pso_config, nRuns)
print("PSO optimization completed.")

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

# 新增功能：在一张画布上展示所有运行的信号时域图
# Plotting results - all runs in one figure
fig = plt.figure(figsize=(15, 10), dpi=200)
ax = fig.add_subplot(111)

# Convert and get real parts of data
t = cp.asnumpy(t)
dataY_real_np = cp.asnumpy(dataY_real)  # Use the real part for plotting

# Plot observed data
ax.scatter(t, dataY_real_np, marker='.', s=1, color='gray', alpha=0.5, label='Observed Data')

# Plot all estimated signals with different colors
colors = plt.cm.tab10(np.linspace(0, 1, nRuns))
for lpruns in range(nRuns):
    # Explicitly get real part of the estimated signal
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t, est_sig, color=colors[lpruns], lw=0.8, label=f'Run {lpruns + 1}')

# Highlight best signal - explicitly get real part
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
ax.plot(t, best_sig, 'red', lw=1.5, label='Best Fit (Run {})'.format(outResults['bestRun'] + 1))

# Set labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')
plt.title('All PSO Runs Comparison')

# Save plot
plt.savefig('all_pso_runs_comparison.png')
plt.close()

# 原来的单独绘图功能保留
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

# Plot observed data
ax.scatter(t, dataY_real_np, marker='.', s=5, label='Observed Data')

# Plot all estimated signals
for lpruns in range(nRuns):
    # Explicitly get real part of the estimated signal
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t, est_sig,
            color=[51 / 255, 255 / 255, 153 / 255],
            lw=0.8,
            alpha=0.5,
            label='Estimated Signal' if lpruns == 0 else "_nolegend_")

# Highlight best signal - explicitly get real part
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
ax.plot(t, best_sig, 'red', lw=0.4, label='Best Fit')

# Set labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')

# Save plot
plt.savefig('pso_optimization_results.png')
plt.close()

# Ensure dataY is using the real part for consistency
dataY = dataY_real

# Process best run first for visualization
best_run_idx = outResults['bestRun']
bestData_real = cp.real(outResults['bestSig'])
bestData = bestData_real + cp.asarray(training_noise)

# Using negative of fitness value as SNR (since fitness is negative SNR)
best_snr_optimal = -outResults['bestFitness']

# Also calculate SNR using PyCBC for verification
best_snr_pycbc = calculate_snr_pycbc(bestData_real, psdHigh, Fs)

# Calculate mismatch using PyCBC
best_epsilon = analyze_mismatch(bestData, dataY, Fs, psdHigh)

best_total_mass = 10 ** outResults['allRunsOutput'][best_run_idx]['m_c'] * M_sun  # In kg
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # Using amplitude A as proxy for flux ratio
best_time_delay = 10 ** outResults['allRunsOutput'][best_run_idx]['delta_t']  # In seconds

# 根据论文使用PyCBC SNR进行分类
best_classification, best_flux_threshold, best_inverse_mass, best_is_lensed = classify_signal(
    float(best_snr_pycbc), best_flux_ratio, best_time_delay, best_total_mass)

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
print(f"delta_t: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")
print(f"Is Lensed: {best_is_lensed}")

print(f"\nBest Run Classification: {best_classification}")
print(f"Epsilon: {best_epsilon:.4f}")
print(f"Flux ratio: {best_flux_ratio:.4f}, Time delay: {best_time_delay:.4f} s")
print(f"Total mass: {best_total_mass / M_sun:.4f} M_sun")
print(f"Flux ratio threshold: {best_flux_threshold:.6f}")
print(f"Inverse mass: {best_inverse_mass:.6f}")

# Final comparison plots
fig = plt.figure(figsize=(20, 8))
plt.subplot(121)
# Ensure all data is real before plotting
plt.plot(t, cp.asnumpy(cp.real(bestData)), label='bestSig + noise')
plt.plot(t, dataY_real_np, label='data', alpha=0.75)
plt.plot(t, cp.asnumpy(cp.real(outResults['bestSig'])), label='bestSig')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Signal Comparison')

plt.subplot(122)
plt.plot(t, cp.asnumpy(cp.real(outResults['bestSig'])), label='bestSig')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Best Signal')

# Save the final comparison plot
plt.savefig('signal_comparison_plot.png')
plt.close()

# Initialize list to store all results
all_results = []

# Now, analyze all runs and add to results
for lpruns in range(nRuns):
    # 获取这次运行的参数
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])

    # 计算SNR
    run_snr_optimal = outStruct[lpruns]['bestFitness']  # 基于优化的SNR
    run_snr_pycbc = calculate_snr_pycbc(run_sig, psdHigh, Fs)  # PyCBC方法的SNR

    run_mass = 10 ** outResults['allRunsOutput'][lpruns]['m_c'] * M_sun
    run_flux_ratio = outResults['allRunsOutput'][lpruns]['A']
    run_time_delay = 10 ** outResults['allRunsOutput'][lpruns]['delta_t']

    # Generate signal for this run
    run_sig_with_noise = run_sig + cp.asarray(training_noise)

    # Calculate mismatch for this run
    run_epsilon = analyze_mismatch(run_sig_with_noise, dataY, Fs, psdHigh)

    # Classify this run with PyCBC SNR
    run_classification, run_flux_threshold, run_inverse_mass, run_is_lensed = classify_signal(
        float(run_snr_pycbc), run_flux_ratio, run_time_delay, run_mass)

    # Print classification for each run
    print(f"\nRun {lpruns + 1} Classification: {run_classification}")
    print(f"  Optimal SNR: {run_snr_optimal:.2f}, PyCBC SNR: {run_snr_pycbc:.2f}")
    print(f"  Flux ratio: {run_flux_ratio:.4f}, Time delay: {run_time_delay:.4f}")
    print(f"  Flux threshold: {run_flux_threshold:.6f}, Inverse mass: {run_inverse_mass:.6f}")
    print(f"  Is Lensed: {run_is_lensed}")

    # Add to results
    run_result = {
        'run': lpruns + 1,
        'fitness': float(outStruct[lpruns]['bestFitness']),
        'r': float(10 ** outResults['allRunsOutput'][lpruns]['r']),
        'm_c': float(10 ** outResults['allRunsOutput'][lpruns]['m_c']),
        'tc': float(outResults['allRunsOutput'][lpruns]['tc']),
        'phi_c': float(outResults['allRunsOutput'][lpruns]['phi_c']) / np.pi,
        'A': float(outResults['allRunsOutput'][lpruns]['A']),
        'delta_t': float(10 ** outResults['allRunsOutput'][lpruns]['delta_t']),
        'SNR_optimal': float(run_snr_optimal),
        'SNR_pycbc': float(run_snr_pycbc),
        'mismatch': float(run_epsilon),
        'flux_ratio_threshold': float(run_flux_threshold),
        'inverse_mass': float(run_inverse_mass),
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
    'delta_t': float(10 ** outResults['allRunsOutput'][best_run_idx]['delta_t']),
    'SNR_optimal': float(best_snr_optimal),
    'SNR_pycbc': float(best_snr_pycbc),
    'mismatch': float(best_epsilon),
    'flux_ratio_threshold': float(best_flux_threshold),
    'inverse_mass': float(best_inverse_mass),
    'classification': best_classification,
    'is_lensed': best_is_lensed
}
all_results.append(best_result)

# Define the columns for our CSV
columns = ['run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t',
           'SNR_optimal', 'SNR_pycbc', 'mismatch', 'flux_ratio_threshold',
           'inverse_mass', 'classification', 'is_lensed']

# Save to CSV using pandas for better formatting
df = pd.DataFrame(all_results, columns=columns)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\nAll run results saved to {csv_filename}")