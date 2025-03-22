import cupy as cp
import scipy.io as scio
from scipy.signal import welch, filtfilt
from scipy.interpolate import interp1d
from PSO_GPU_main import *
import matplotlib.pyplot as plt
import numpy as np
import pycbc.types
from pycbc.filter import match

# Constants
G = 6.67430e-11
c = 2.998e8
M_sun = 1.989e30
pc = 3.086e16

# Load data
TrainingData = scio.loadmat('generate_ligo/noise.mat')
analysisData = scio.loadmat('generate_ligo/data_No_lens.mat')

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
# dataY = cp.asarray(analysisData['noise'][0])
nSamples = dataY.size
Fs = float(analysisData['samples'][0][0])  
# Fs = 2000
# Search range parameters
#                r  mc tc phi A  Δtd
rmin = cp.array([-2, 0, 0, 0, 0, -2])   # 对应参数范围下限`
rmax = cp.array([4, 3, 10, 2*np.pi, 1, 2])  # 对应参数范围上限

# Time domain setup
dt = 1/Fs  # 采样率Hz
t = np.arange(-90, 10, dt) 
T = nSamples/Fs
df = 1/T
Nyq = Fs/2

# PSD estimation (CPU operation)
training_noise = TrainingData['noise'][0]  # Get noise data
psdHigh = TrainingData['psd'][0]  # Get PSD data directly

# Calculate Welch periodogram for training noise
[f_noise, pxx_noise] = welch(training_noise, fs=Fs,
                 window='hamming', nperseg=int(Fs/2),
                 noverlap=None, nfft=None,
                 detrend=False)

# Convert data to CPU for Welch on signal+noise
dataY_cpu = cp.asnumpy(dataY)
[f_signal, pxx_signal] = welch(dataY_cpu, fs=Fs,
                  window='hamming', nperseg=int(Fs/2),  # Using the same nperseg for consistency
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
dataY = cp.asnumpy(dataY)
# Plot observed data
ax.scatter(t, dataY, marker='.', s=5, label='Observed Data')

# Plot all estimated signals
for lpruns in range(nRuns):
    est_sig = cp.asnumpy(cp.real(outResults['allRunsOutput'][lpruns]['estSig']))
    ax.plot(t, est_sig,
            color=[51/255, 255/255, 153/255],
            lw=0.8,
            alpha=0.5,
            label='Estimated Signal' if lpruns == 0 else "_nolegend_")

# Highlight best signal
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
print(f"r : {10**outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}")
print(f"Mc: {10**outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}")
print(f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}")
print(f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c']/np.pi:.4f}")
print(f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}")
print(f"delta_t: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}")
print(f"SNR: {cp.sqrt(-outResults['bestFitness']):.2f}")

for lpruns in range(nRuns):
    print(f"\nRun No.{lpruns+1}:")
    print(f"bestFitness={float(outStruct[lpruns]['bestFitness']):.4f}")
    print(f"r = {float(10 ** outResults['allRunsOutput'][lpruns]['r']):.4f}")
    print(f"m_c = {float(10 ** outResults['allRunsOutput'][lpruns]['m_c']):.4f}")
    print(f"tc = {float(outResults['allRunsOutput'][lpruns]['tc']):.4f}")
    print(f"phi_c = {float(outResults['allRunsOutput'][lpruns]['phi_c'])/np.pi:.4f}")
    print(f"A = {float(outResults['allRunsOutput'][lpruns]['A']):.4f}")
    print(f"delta_t = {float(10 ** outResults['allRunsOutput'][lpruns]['delta_t']):.4f}")
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
    print(f"match_value:{match_value:.4f}")
    epsilon = 1 - match_value
    return epsilon

# Get parameters for classification
bestData = outResults['bestSig'] + training_noise
bestData = cp.asarray(bestData)
dataY = cp.asarray(dataY)
psdHigh = cp.asarray(psdHigh)

# Calculate mismatch and SNR
epsilon = analyze_mismatch(bestData, dataY, Fs, psdHigh)
snr = cp.sqrt(-outResults['bestFitness'])
total_mass = 10**outResults['allRunsOutput'][outResults['bestRun']]['m_c'] * M_sun  # In kg
flux_ratio = outResults['allRunsOutput'][outResults['bestRun']]['A']  # Using amplitude A as proxy for flux ratio
time_delay = 10**outResults['allRunsOutput'][outResults['bestRun']]['delta_t']  # In seconds

# Apply criteria from the paper
print(f"Epsilon: {epsilon:.4f}, SNR: {snr:.2f}")
print(f"Flux ratio: {flux_ratio:.4f}, Time delay: {time_delay:.4f} s")
print(f"Total mass: {total_mass/M_sun:.4f} M_sun")

# Critical threshold based on paper
flux_threshold = 2 * (snr ** (-2))
inverse_mass = M_sun / total_mass  # Inverse mass in solar mass units

print(f"Flux ratio threshold: {flux_threshold:.6f}")
print(f"Inverse mass: {inverse_mass:.6f}")

# Classification based on paper criteria
if snr < 8:
    classification = "Pure Noise (SNR too low)"
else:
    if flux_ratio >= flux_threshold and time_delay >= inverse_mass:
        classification = "Lensed Signal (matches both criteria)"
    elif flux_ratio >= flux_threshold:
        classification = "Potential Lensed Signal (matches flux ratio criterion only)"
    elif time_delay >= inverse_mass:
        classification = "Potential Lensed Signal (matches time delay criterion only)"
    else:
        classification = "Unlensed Signal (doesn't meet lensing criteria)"

print(f"\nClassification: {classification}")

# Final comparison plots
fig = plt.figure(figsize=(20,8))
plt.subplot(121)
plt.plot(t, cp.asnumpy(bestData), label='bestSig + noise')
plt.plot(t, cp.asnumpy(dataY), label='data', alpha=0.75)
plt.plot(t, cp.asnumpy(outResults['bestSig']), label='bestSig')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Signal Comparison')

plt.subplot(122)
plt.plot(t, cp.asnumpy(outResults['bestSig']), label='bestSig')
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
plt.plot(t, cp.asnumpy(dataY), 'gray', alpha=0.5, label='Observed Data')
plt.plot(t, cp.asnumpy(outResults['bestSig']), 'r', lw=1, label='Best Fit')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Data vs Best Signal')

# Right side: Parameter summary
plt.subplot(122)
plt.text(0.1, 0.95, "Results Summary", fontsize=14, fontweight='bold')
plt.text(0.1, 0.88, f"Best Fitness: {outResults['bestFitness']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.82, f"SNR: {cp.sqrt(-outResults['bestFitness']):.2f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.76, f"r: {10**outResults['allRunsOutput'][outResults['bestRun']]['r']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.70, f"Mc: {10**outResults['allRunsOutput'][outResults['bestRun']]['m_c']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.64, f"tc: {outResults['allRunsOutput'][outResults['bestRun']]['tc']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.58, f"phi_c: {outResults['allRunsOutput'][outResults['bestRun']]['phi_c']/np.pi:.4f}π", transform=plt.gca().transAxes)
plt.text(0.1, 0.52, f"A: {outResults['allRunsOutput'][outResults['bestRun']]['A']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.46, f"delta_t: {10 ** outResults['allRunsOutput'][outResults['bestRun']]['delta_t']:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.40, f"Mismatch: {epsilon:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.34, f"Flux ratio threshold: {flux_threshold:.6f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.28, f"Inverse mass: {inverse_mass:.6f}", transform=plt.gca().transAxes)

# Add signal classification result
plt.text(0.1, 0.18, f"Classification: {classification}", transform=plt.gca().transAxes, fontsize=12, color='blue', fontweight='bold')

plt.axis('off')
plt.tight_layout()
plt.savefig('analysis_summary.png')
plt.close()