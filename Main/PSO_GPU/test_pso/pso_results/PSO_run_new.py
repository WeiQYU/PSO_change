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
analysisData = scio.loadmat('../../generate_ligo/data_without_lens.mat')
print("加载完毕")

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
training_noise = cp.asarray(TrainingData['noise'][0])  # Move to GPU
dataY_only_signal = dataY - training_noise  # Extract signal part (for comparison)

nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])
# Search range parameters
#                r  mc tc phi A  Δtd
rmin = cp.array([-2, 0, 0, 0, 0, 0])  # parameter range lower bounds
rmax = cp.array([4, 3, 8, 2 * np.pi, 2, 7])  # parameter range upper bounds
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
    'dataY_only_signal': dataY_only_signal,  # Add signal-only data for mismatch calculation
    'sampFreq': Fs,
    'psdHigh': psdHigh,
    'rmin': rmin,
    'rmax': rmax,
}

# Number of PSO runs
nRuns = 8

# PSO configuration parameters
pso_config = {
    'popsize': 50,  # Population size
    'maxSteps': 3000,  # Number of iterations
    'c1': 2,  # Individual learning factor
    'c2': 2,  # Social learning factor
    'w_start': 0.9,  # Initial inertia weight
    'w_end': 0.4,  # Final inertia weight
    'max_velocity': 0.5,  # Maximum velocity limit
    'nbrhdSz': 5  # Neighborhood size
}

print("PSO已部署完毕,芜湖！！！！")
# Run PSO optimization with two-step matching process enabled
outResults, outStruct = crcbqcpsopsd(inParams, pso_config, nRuns, use_two_step=True)

# For plotting, data must be moved back to CPU
# Only convert when visualization is needed
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

# Display all runs' signals in one plot
fig = plt.figure(figsize=(15, 10), dpi=200)
ax = fig.add_subplot(111)

# Convert data to CPU for plotting
dataY_real_np = cp.asnumpy(cp.real(dataY))  # Use real part for plotting

# Plot observed data
ax.scatter(t_cpu, dataY_real_np, marker='.', s=1, color='gray', alpha=0.5, label='Observed Data')

# Plot all estimated signals with different colors
colors = plt.cm.tab10(np.linspace(0, 1, nRuns))
for lpruns in range(nRuns):
    # Get real part of estimated signal and move to CPU for plotting
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))

    # Add label to show classification result
    classification = outResults['allRunsOutput'][lpruns]['classification']
    ax.plot(t_cpu, est_sig, color=colors[lpruns], lw=0.8,
            label=f'Run {lpruns + 1} ({classification})')

# Highlight best signal
best_sig = cp.asnumpy(cp.real(outResults['bestSig']))
best_classification = outResults['classification']
ax.plot(t_cpu, best_sig, 'red', lw=1.5,
        label=f'Best Fit (Run {outResults["bestRun"] + 1}, {best_classification})')

# Set labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend(loc='upper right')
plt.title('All PSO Runs Comparison')

# Save chart
plt.savefig('all_pso_runs_comparison.png')
plt.close()

# Process best run for visualization
best_run_idx = outResults['bestRun']
bestSig_real = cp.real(outResults['bestSig'])

# Calculate SNR
best_snr_pycbc = calculate_snr_pycbc(bestSig_real, psdHigh, Fs)

# Calculate mismatch using PyCBC
best_epsilon = analyze_mismatch(bestSig_real, dataY_only_signal, Fs, psdHigh)

# Extract parameters from best run
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # Amplitude A as lens flux ratio
best_time_delay = outResults['allRunsOutput'][best_run_idx]['delta_t']  # In seconds

# Print results
print('\n============= Final Results =============')
print(f"Best Fitness (Inner Product): {outResults['bestFitness']:.4f}")
print(f"PyCBC SNR (Independently Calculated): {best_snr_pycbc:.2f}")
print(f"r : {10 ** outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}")
print(f"Mc: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}")
print(f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}")
print(f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c'] / np.pi:.4f}")
print(f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}")
print(f"delta_t: {outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")

# Print two-step matching results
print(f"\n============= Lensing Analysis =============")
print(f"Two-step Matching Result: {outResults['lensing_message']}")
print(f"Classification: {outResults['classification']}")
print(f"Is Lensed: {outResults['is_lensed']}")
print(f"Mismatch: {best_epsilon:.6f}")
print(f"Mismatch Threshold (1/SNR): {1 / (best_snr_pycbc ** 2):.6f}")

# Final comparison plot - convert to CPU just for plotting
bestData_cpu = cp.asnumpy(cp.real(dataY))
bestSig_cpu = cp.asnumpy(cp.real(outResults['bestSig']))

# Plot best signal vs data comparison
fig = plt.figure(figsize=(20, 8))
plt.plot(t_cpu, bestData_cpu, 'gray', alpha=0.5, label='Observed Data')
plt.plot(t_cpu, bestSig_cpu, 'r', label='Best Signal')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
classification = outResults['classification']
plt.title(f'Best Signal Comparison ({classification}): {outResults["lensing_message"]}')

# Save final comparison plot
plt.savefig('signal_comparison_plot.png')
plt.close()

# Initialize list to store all results
all_results = []

# Analyze all runs and add to results
for lpruns in range(nRuns):
    # Get parameters for this run
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])

    # Calculate SNR
    run_snr_optimal = -float(outStruct[lpruns]['bestFitness'])  # SNR based on optimization (negative value)
    run_snr_pycbc = calculate_snr_pycbc(run_sig, psdHigh, Fs)  # PyCBC method SNR

    run_mass = 10 ** outResults['allRunsOutput'][lpruns]['m_c'] * M_sun
    run_flux_ratio = outResults['allRunsOutput'][lpruns]['A']
    run_time_delay = outResults['allRunsOutput'][lpruns]['delta_t']

    # Calculate mismatch for this run
    run_epsilon = analyze_mismatch(run_sig, dataY_only_signal, Fs, psdHigh)

    # Mismatch threshold
    mismatch_threshold = 1.0 / run_snr_pycbc

    # Print classification for each run
    print(f"\nRun {lpruns + 1} Results:")
    print(f"  Classification: {outResults['allRunsOutput'][lpruns]['classification']}")
    print(f"  Two-step Matching Result: {outResults['allRunsOutput'][lpruns]['lensing_message']}")
    print(f"  Is Lensed: {outResults['allRunsOutput'][lpruns]['is_lensed']}")
    print(f"  SNR: {run_snr_pycbc:.2f}, Mismatch: {run_epsilon:.6f}, Threshold: {mismatch_threshold:.6f}")

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
        'mismatch_threshold': float(mismatch_threshold),
        'two_step_match_result': outResults['allRunsOutput'][lpruns]['lensing_message'],
        'two_step_is_lensed': outResults['allRunsOutput'][lpruns]['is_lensed'],
        'classification': outResults['allRunsOutput'][lpruns]['classification']
    }
    all_results.append(run_result)

# Add best result as summary entry (marked as "best")
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
    'mismatch': float(best_epsilon),
    'mismatch_threshold': float(1 / (best_snr_pycbc)),
    'two_step_match_result': outResults['lensing_message'],
    'two_step_is_lensed': outResults['is_lensed'],
    'classification': outResults['classification']
}
all_results.append(best_result)

# Define CSV columns
columns = ['run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t',
           'SNR_pycbc', 'mismatch', 'mismatch_threshold',
           'two_step_match_result', 'two_step_is_lensed', 'classification']

# Use pandas to save as CSV for better formatting
df = pd.DataFrame(all_results, columns=columns)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\n本次飞行结束，飞行结果报告已保存在：{csv_filename}")