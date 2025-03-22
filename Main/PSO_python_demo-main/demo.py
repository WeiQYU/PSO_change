import numpy as np
import scipy.io as scio # load mat file
from scipy.signal import welch, filtfilt
from scipy.interpolate import interp1d

from PSO import *  # demo PSO codes!

import matplotlib
matplotlib.use('agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

G = 6.67430e-11  # 万有引力常数, m^3 kg^-1 s^-2
c = 2.998e8  # 光速, m/s
M_sun = 1.989e30  # 太阳质量, kg
pc = 3.086e16  # pc到m的转换
if __name__ == '__main__':

    # load data
    TrainingData = scio.loadmat('../PSO_GPU/generate_ligo/noise.mat')  # only noise
    analysisData = scio.loadmat('../PSO_GPU/generate_ligo/data.mat')  # noise + signal
    # analysisData = scio.loadmat('TrainingData.mat') # only noise to check

    ## Preparing
    dataY = analysisData['data'][0]  # 有信号用
    # dataY = analysisData['noise'][0] # 纯噪声用
    # Data length
    nSamples = dataY.size
    # Sampling frequency
    Fs = analysisData['samples'][0]
    # Fs = 2000
    # Search range of phase coefficients
    """
            r(Mpc),m_c(Msun), tc(s),phi_c,mlz(Msun),y
            use log:r,m_c,mlz
    """
    rmin = [-2, 0, 0, 0, 4, 0]
    rmax = [4, 3, 10, 2 * np.pi, 14, 10]

    # Noise realization: PSD estimated from TrainingData
    dt = 1 / Fs
    # t = np.arange(0, nSamples*dt, dt) # (2048,)

    t = np.linspace(-30, 10, nSamples)
    T = nSamples / Fs
    df = 1 / T
    Nyq = Fs / 2  # Nyquist frequency
    [f, pxx] = welch(TrainingData['noise'][0], fs=Fs,
                     window='hamming', nperseg=Fs / 2,
                     noverlap=None, nfft=None,
                     detrend=False)
    # Why 'detrend=False'? 
    # See https://github.com/scipy/scipy/issues/8045#issuecomment-337319294
    # or https://iphysresearch.github.io/blog/post/signal_processing/spectral_analysis_scipy/

    # Smooth the PSD estimate
    smthOrdr = 10
    b = np.ones(smthOrdr) / smthOrdr
    pxxSmth = filtfilt(b, 1, pxx)
    # PSD must be supplied at DFT frequencies.
    kNyq = np.floor(nSamples / 2) + 1
    posFreq = np.arange(0, kNyq) * Fs / nSamples
    psdPosFreq = interp1d(f, pxxSmth)(posFreq)

    # Plot PSDs for the noise and noise + signal.
    plt.figure(dpi=200)
    plt.plot(f, pxx, label='noise')
    plt.plot(f, pxxSmth, label='noise (smooth)')
    [f, pxxY] = welch(dataY, fs=Fs,
                      window='hamming', nperseg=256,
                      noverlap=None, nfft=None,
                      detrend=False)
    plt.plot(np.abs(f), pxxY, label='noise + signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.yscale('log')
    plt.xlim(0, 250)
    plt.legend()
    plt.savefig('output_psd.png', dpi=200)
    plt.show()


    # Number of independent PSO runs
    nRuns = 8

    ## PSO
    # Input parameters for CRCBQCHRPPSO
    inParams = {
        'dataX': t,
        'dataY': dataY,
        # 'dataXSq': t**2,
        # 'dataXCb': t**3,
        'psdPosFreq': psdPosFreq,
        'sampFreq': Fs,
        'rmin': rmin,
        'rmax': rmax,
    }
    # CRCBQCHRPPSOPSD runs PSO on the CRCBQCHRPFITFUNC fitness function. As an
    # illustration of usage, we change one of the PSO parameters from itsw
    # default value.
    outResults, outStruct = crcbqcpsopsd(inParams, {'maxSteps': 2000}, nRuns)

    ## Plots
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    a = ax.scatter(t, dataY, marker='.', s=5,  # label='analysisData'
                   )
    a.set_label('analysisData')
    for lpruns in range(nRuns):
        b, = ax.plot(t, outResults['allRunsOutput'][lpruns]['estSig'],
                     color=[51 / 255, 255 / 255, 153 / 255], lw=.4 * 2)
    b.set_label('estSig')
    c, = ax.plot(t, outResults['bestSig'],  # label='BestSig',
                 'red', lw=.2 * 2)
    c.set_label('BestSig')
    plt.legend()
    plt.savefig('output_sig.png', dpi=200)
    plt.show()

    # Print estimated parameters
    print('Estimated parameters:')
    print('bestFitness = {:.4f}'.format(outResults['bestFitness']))
    print('r = {:.4f}'.format(10 ** outResults['r']))
    print('m_c = {:.4f}'.format(10 ** outResults['m_c']))
    print('tc = {:.4f}'.format(outResults['tc']))
    print('phi_c = {:.4f}'.format(outResults['phi_c'] / np.pi))
    print('mlz = {:.4e}'.format(10 ** outResults['mlz']))
    print('y = {:.4f}'.format(outResults['y']))
    print('SNR = {:.2f}'.format(np.sqrt(-outResults['bestFitness'])))

    for lpruns in range(nRuns):
        print('\nRun No.{}:'.format(lpruns + 1))
        print('bestFitness={:.4f}'.format(outStruct[lpruns]['bestFitness']))
        print('r = {:.4f}'.format(10 ** outResults['allRunsOutput'][lpruns]['r']))
        print('m_c = {:.4f}'.format(10 ** outResults['allRunsOutput'][lpruns]['m_c']))
        print('tc = {:.4f}'.format(outResults['allRunsOutput'][lpruns]['tc']))
        print('phi_c = {:.4f}'.format(outResults['allRunsOutput'][lpruns]['phi_c'] / np.pi))
        print('mlz = {:.4e}'.format(10 ** outResults['allRunsOutput'][lpruns]['mlz']))
        print('y = {:.4f}'.format(outResults['allRunsOutput'][lpruns]['y']))
        print('SNR = {:.2f}'.format(np.sqrt(-outStruct[lpruns]['bestFitness'])))
    # Save
    np.save('output_results', outResults)
    np.save('output_struct', outStruct)
