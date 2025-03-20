import numpy as np
import cupy as cp
import scipy.io as scio
from scipy.signal import welch, filtfilt
from scipy.interpolate import interp1d
from PSO_GPU_main import *
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# Constants
G = 6.67430e-11
c = 2.998e8
M_sun = 1.989e30
pc = 3.086e16

# Load data (adjust paths if needed)
data_dir = os.path.join(os.path.dirname(__file__), '../generate_ligo')
try:
    TrainingData = scio.loadmat(os.path.join(data_dir, 'noise.mat'))
    analysisData = scio.loadmat(os.path.join(data_dir, 'data.mat'))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required .mat files not found in {data_dir}. Please check the paths.") from e

# Convert data to CuPy arrays
dataY = cp.asarray(analysisData['data'][0])
# dataY = cp.asarray(analysisData['noise'][0])
nSamples = dataY.size
Fs = float(analysisData['samples'][0])
# Fs = 2000
# Search range parameters (r, m_c, tc, phi_c, mlz, y)
rmin = cp.array([-2, 0, 0, 0, 4, -5])   # 对应参数范围下限
rmax = cp.array([4, 3, 10, 2*np.pi, 14, 1])  # 对应参数范围上限

# Time domain setup
dt = 1/Fs
t = cp.linspace(-90, 10, nSamples)
T = nSamples/Fs
df = 1/T
Nyq = Fs/2

# PSD estimation (CPU operation)
training_noise = TrainingData['noise'][0]
[f, pxx] = welch(training_noise, fs=Fs,
                 window='hamming', nperseg=int(Fs/2),
                 noverlap=None, nfft=None,
                 detrend=False)

# Smooth PSD
smthOrdr = 10
b = np.ones(smthOrdr)/smthOrdr
pxxSmth = filtfilt(b, 1, pxx)

# Interpolate PSD
kNyq = int(cp.floor(nSamples/2)) + 1
posFreq = cp.arange(0, kNyq)*Fs/nSamples
psdPosFreq = cp.asarray(interp1d(f, pxxSmth)(cp.asnumpy(posFreq)))

# Plot PSDs
plt.figure(dpi=200)
plt.plot(f, pxx, label='Noise (raw)')
plt.plot(f, pxxSmth, label='Noise (smoothed)', linestyle='--')

# Convert data to CPU for Welch
dataY_cpu = cp.asnumpy(dataY)
[f, pxxY] = welch(dataY_cpu, fs=Fs,
                  window='hamming', nperseg=256,
                  noverlap=None, nfft=None,
                  detrend=False)

plt.plot(np.abs(f), pxxY, label='Signal + Noise', alpha=0.7)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.yscale('log')
plt.xlim(0, 250)
plt.legend()
plt.savefig('output_psd.png', dpi=200, bbox_inches='tight')
plt.close()

# PSO input parameters
inParams = {
    'dataX': t,
    'dataY': dataY,
    'psdPosFreq': psdPosFreq,
    'sampFreq': Fs,
    'rmin': rmin,
    'rmax': rmax,
}

# Number of PSO runs
nRuns = 15

# Run PSO optimization
outResults, outStruct = crcbqcpsopsd(inParams, {'maxSteps': 2000}, nRuns)

# Plotting results
fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.scatter(cp.asnumpy(t), cp.asnumpy(dataY), marker='.', s=5, label='Observed Data')

# Plot all estimated signals
for lpruns in range(nRuns):
    est_sig = outResults['allRunsOutput'][lpruns]['estSig']
    ax.plot(cp.asnumpy(t), est_sig,
            color=[51/255, 255/255, 153/255], lw=0.8, alpha=0.5, label='Estimated Signal' if lpruns == 0 else "")

# Highlight best signal
ax.plot(cp.asnumpy(t), outResults['bestSig'], 'red', lw=0.4, label='Best Fit')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.savefig('output_signal.png', dpi=200, bbox_inches='tight')
plt.close()

# Print results
print('\n============= Final Results =============')
print(f"Best Fitness (GLRT): {outResults['bestFitness']:.4f}")
print(f"r : {10**outResults['r']:.4f}")
print(f"Mc: {10**outResults['m_c']:.4f}")
print(f"tc: {outResults['tc']:.4f}")
print(f"phi_c: {outResults['phi_c']/np.pi:.4f}")
print(f"mlz: {10**outResults['mlz']:.4e}")
print(f"y: {10 ** outResults['y']:.4f}")
print(f"SNR: {cp.sqrt(-outResults['bestFitness']):.2f}")

for lpruns in range(nRuns):
    print(f"\nRun No.{lpruns+1}:")
    print(f"bestFitness={float(outStruct[lpruns]['bestFitness']):.4f}")
    print(f"r = {float(10 ** outResults['allRunsOutput'][lpruns]['r']):.4f}")
    print(f"m_c = {float(10 ** outResults['allRunsOutput'][lpruns]['m_c']):.4f}")
    print(f"tc = {float(outResults['allRunsOutput'][lpruns]['tc']):.4f}")
    print(f"phi_c = {float(outResults['allRunsOutput'][lpruns]['phi_c'])/np.pi:.4f}")
    print(f"mlz = {float(10 ** outResults['allRunsOutput'][lpruns]['mlz']):.4e}")
    print(f"y = {float(10 ** outResults['allRunsOutput'][lpruns]['y']):.4f}")
    print(f"SNR = {float(cp.sqrt(-outStruct[lpruns]['bestFitness'])):.2f}")

fig = plt.figure(dpi = 200)
plt.plot(cp.asnumpy(t),outResults['bestSig'],label = 'Best signal')
plt.savefig('BestSignal.png',dpi = 200,bbox_inches ='tight')
# Save results
np.save('output_results.npy', outResults)
np.save('output_struct.npy', outStruct)
