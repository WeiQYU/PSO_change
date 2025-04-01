import cupy as cp
import scipy.io as scio
from scipy.signal import welch, filtfilt
from scipy.interpolate import interp1d
from PSO_GPU_main import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import csv
import os
import pandas as pd

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30
pc = 3.086e16

# Load data
TrainingData = scio.loadmat('../generate_ligo/noise.mat')
analysisData = scio.loadmat('../generate_ligo/data_without_lens.mat')
t0 = int(analysisData['t0'][0][0])  # Extract the scalar value properly
t1 = int(analysisData['t1'][0][0]) + 1  # Extract the scalar value properly
# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
# dataY = cp.asarray(analysisData['noise'][0])
nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])
# Fs = 2000
# Search range parameters
#                r  mc tc phi A  Δtd
rmin = cp.array([-2, 0, t0, 0, 0, -3])  # 对应参数范围下限`
rmax = cp.array([4, 3, t1, 2 * np.pi, 1, 1])  # 对应参数范围上限

# Time domain setup
dt = 1 / Fs  # 采样率Hz
t = np.arange(0, 32, dt)
T = nSamples / Fs
df = 1 / T
Nyq = Fs / 2

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
nRuns = 8
# Run PSO optimization
outResults, outStruct = crcbqcpsopsd(inParams, {'maxSteps': 2000}, nRuns)

# Plotting results
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

# Convert and get real parts of data
t = cp.asnumpy(t)
dataY_real_np = cp.asnumpy(dataY_real)  # Use the real part for plotting

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

print('\n============= Final Results =============')
print(f"Best Fitness (GLRT): {outResults['bestFitness']:.4f}")
print(f"r : {10 ** outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}")
print(f"Mc: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}")
print(f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}")
print(f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c'] / np.pi:.4f}")
print(f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}")
print(f"delta_t: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")
print(f"SNR: {cp.sqrt(-outResults['bestFitness']):.2f}")

for lpruns in range(nRuns):
    print(f"\nRun No.{lpruns + 1}:")
    print(f"bestFitness={float(outStruct[lpruns]['bestFitness']):.4f}")
    print(f"r = {float(10 ** outResults['allRunsOutput'][lpruns]['r']):.4f}")
    print(f"m_c = {float(10 ** outResults['allRunsOutput'][lpruns]['m_c']):.4f}")
    print(f"tc = {float(outResults['allRunsOutput'][lpruns]['tc']):.4f}")
    print(f"phi_c = {float(outResults['allRunsOutput'][lpruns]['phi_c']) / np.pi:.4f}")
    print(f"A = {float(outResults['allRunsOutput'][lpruns]['A']):.4f}")
    print(f"delta_t = {10 ** float(outResults['allRunsOutput'][lpruns]['delta_t']):.4f}")
    print(f"SNR = {float(cp.sqrt(-outStruct[lpruns]['bestFitness'])):.2f}")


def analyze_mismatch(data, h_lens, samples, psdHigh):
    # Convert inputs to CuPy arrays
    data_cupy = cp.asarray(data)
    h_lens_cupy = cp.asarray(h_lens)
    # Calculate match value
    match_value = (innerprodpsd(h_lens_cupy, data_cupy, samples, psdHigh) /
                   cp.sqrt(innerprodpsd(h_lens_cupy, h_lens_cupy, samples, psdHigh) *
                           innerprodpsd(data_cupy, data_cupy, samples, psdHigh)))

    # Calculate mismatch
    # print(f"match_value:{match_value:.4f}")
    epsilon = 1 - match_value
    return epsilon


def classify_signal(snr, flux_ratio, time_delay, total_mass):
    # Critical threshold based on paper
    flux_threshold = 2 * (snr ** (-2))
    inverse_mass = M_sun / total_mass  # Inverse mass in solar mass units

    # Classification based on paper criteria
    if snr < 8:
        classification = "Pure Noise (SNR too low)"
    else:
        if flux_ratio >= flux_threshold and time_delay >= inverse_mass:
            classification = "Lensed Signal (matches both criteria)"
        elif flux_ratio >= flux_threshold:
            classification = "Potential Lensed Signal (matches flux ratio criterion only,I)"
        elif time_delay >= inverse_mass:
            classification = "Potential Lensed Signal (matches time delay criterion only,Δtd)"
        else:
            classification = "Unlensed Signal (doesn't meet lensing criteria)"

    return classification, flux_threshold, inverse_mass


# Ensure dataY is using the real part for consistency
dataY = dataY_real
psdHigh = cp.asarray(psdHigh)

# Process best run first for visualization
best_run_idx = outResults['bestRun']
bestData_real = cp.real(outResults['bestSig'])
bestData = bestData_real + cp.asarray(training_noise)
best_epsilon = analyze_mismatch(bestData, dataY, Fs, psdHigh)
best_snr = cp.sqrt(-outResults['bestFitness'])
best_total_mass = 10 ** outResults['allRunsOutput'][best_run_idx]['m_c'] * M_sun  # In kg
best_flux_ratio = outResults['allRunsOutput'][best_run_idx]['A']  # Using amplitude A as proxy for flux ratio
best_time_delay = 10 ** outResults['allRunsOutput'][best_run_idx]['delta_t']  # In seconds

best_classification, best_flux_threshold, best_inverse_mass = classify_signal(
    best_snr, best_flux_ratio, best_time_delay, best_total_mass)

print(f"\nBest Run Classification: {best_classification}")
print(f"Epsilon: {best_epsilon:.4f}, SNR: {best_snr:.2f}")
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

# Create a summary figure with plots on left, text on right
plt.figure(figsize=(15, 10), dpi=200)

# Top left: PSD Comparison
plt.subplot(221)
plt.plot(f_noise, pxx_noise, label='Noise (raw)')
plt.plot(f_signal, pxx_signal, label='Signal + Noise', alpha=0.7)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.title('PSD Comparison')

# Bottom left: Data vs Best Signal
plt.subplot(223)
plt.plot(t, dataY_real_np, 'gray', alpha=0.5, label='Observed Data')
plt.plot(t, cp.asnumpy(cp.real(outResults['bestSig'])), 'r', lw=1, label='Best Fit')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Data vs Best Signal')

# Right side: Parameter summary
plt.subplot(122)
plt.text(0.1, 0.95, "Results Summary", fontsize=14, fontweight='bold')
plt.text(0.1, 0.88, f"Best Fitness: {outResults['bestFitness']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.82, f"SNR: {cp.sqrt(-outResults['bestFitness']):.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.76, f"r: {10 ** outResults['allRunsOutput'][best_run_idx]['r']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.70, f"Mc: {10 ** outResults['allRunsOutput'][best_run_idx]['m_c']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.64, f"tc: {outResults['allRunsOutput'][best_run_idx]['tc']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.58, f"phi_c: {outResults['allRunsOutput'][best_run_idx]['phi_c'] / np.pi:.4f}π",
         transform=plt.gca().transAxes)
plt.text(0.1, 0.52, f"A: {outResults['allRunsOutput'][best_run_idx]['A']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.46, f"delta_t: {outResults['allRunsOutput'][best_run_idx]['delta_t']:.4f}",
         transform=plt.gca().transAxes)
plt.text(0.1, 0.40, f"Mismatch: {best_epsilon:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.34, f"Flux ratio threshold: {best_flux_threshold:.6f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.28, f"Inverse mass: {best_inverse_mass:.6f}", transform=plt.gca().transAxes)

# Add signal classification result
plt.text(0.1, 0.18, f"Classification: {best_classification}", transform=plt.gca().transAxes, fontsize=12, color='blue',
         fontweight='bold')

plt.axis('off')
plt.tight_layout()
plt.savefig('analysis_summary.png')
plt.close()

# Initialize list to store all results
all_results = []

# Now, analyze all runs and add to results
for lpruns in range(nRuns):
    # Get parameters for this run
    run_snr = cp.sqrt(-outStruct[lpruns]['bestFitness'])
    run_mass = 10 ** outResults['allRunsOutput'][lpruns]['m_c'] * M_sun
    run_flux_ratio = outResults['allRunsOutput'][lpruns]['A']
    run_time_delay = 10 ** outResults['allRunsOutput'][lpruns]['delta_t']

    # Generate signal for this run
    run_sig = cp.real(outResults['allRunsOutput'][lpruns]['estSig'])
    run_sig_with_noise = run_sig + cp.asarray(training_noise)

    # Calculate mismatch for this run
    run_epsilon = analyze_mismatch(run_sig_with_noise, dataY, Fs, psdHigh)

    # Classify this run
    run_classification, run_flux_threshold, run_inverse_mass = classify_signal(
        run_snr, run_flux_ratio, run_time_delay, run_mass)

    # Print classification for each run
    print(f"\nRun {lpruns + 1} Classification: {run_classification}")
    print(f"  SNR: {run_snr:.2f}, Flux ratio: {run_flux_ratio:.4f}, Time delay: {run_time_delay:.4f}")
    print(f"  Flux threshold: {run_flux_threshold:.6f}, Inverse mass: {run_inverse_mass:.6f}")

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
        'SNR': float(run_snr),
        'mismatch': float(run_epsilon),
        'flux_ratio_threshold': float(run_flux_threshold),
        'inverse_mass': float(run_inverse_mass),
        'classification': run_classification
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
    'SNR': float(best_snr),
    'mismatch': float(best_epsilon),
    'flux_ratio_threshold': float(best_flux_threshold),
    'inverse_mass': float(best_inverse_mass),
    'classification': best_classification
}
all_results.append(best_result)

# Define the columns for our CSV
columns = ['run', 'fitness', 'r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t', 'SNR',
           'mismatch', 'flux_ratio_threshold', 'inverse_mass', 'classification']

# Save to CSV using pandas for better formatting
df = pd.DataFrame(all_results, columns=columns)
csv_filename = 'pso_results.csv'
df.to_csv(csv_filename, index=False)
print(f"\nAll run results saved to {csv_filename}")