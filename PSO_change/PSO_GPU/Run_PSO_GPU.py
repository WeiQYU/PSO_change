import cupy as cp
from cupyx.scipy.fft import fft, fftfreq
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
import scipy.io as scio
import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

# 配置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cp.cuda.Device(0).use()

# 物理常数
G = 6.67430e-11
c = 2.998e8
M_sun = 1.989e30
pc = 3.086e16


def gpu_welch(x, fs, window='hamming', nperseg=256):
    """GPU加速的PSD估算实现"""
    n = len(x)
    nseg = n // nperseg
    window = cp.hamming(nperseg)

    # 生成频率数组（关键修改）
    freqs = cp.fft.fftfreq(nperseg, 1 / fs)[:nperseg // 2 + 1]

    pxx = cp.zeros(nperseg // 2 + 1, dtype=cp.float64)
    for i in range(nseg):
        seg = x[i * nperseg: (i + 1) * nperseg]
        seg = (seg - cp.mean(seg)) * window
        fft_vals = fft(seg)[:nperseg // 2 + 1]
        pxx += cp.abs(fft_vals) ** 2 / (fs * cp.sum(window ** 2))

    return freqs, pxx / nseg  # 返回频率和PSD数组


if __name__ == '__main__':
    # 加载数据并转换到GPU
    TrainingData = scio.loadmat('noise.mat')
    analysisData = scio.loadmat('data.mat')

    dataY = cp.asarray(analysisData['data'][0])
    Fs = analysisData['samples'][0].item()
    nSamples = dataY.size

    # 生成时间序列
    t = cp.linspace(-30, 10, nSamples)
    T = nSamples / Fs

    # PSD估计
    noise_gpu = cp.asarray(TrainingData['noise'][0])
    f, pxx = gpu_welch(noise_gpu, Fs, nperseg=int(Fs // 2))

    # 平滑处理（关键修复部分）
    smthOrdr = 10
    b_gpu = cp.ones(smthOrdr) / smthOrdr
    b_cpu = cp.asnumpy(b_gpu)  # 显式转换为 NumPy
    pxx_cpu = cp.asnumpy(pxx)  # 转换为 NumPy
    pxxSmth = filtfilt(b_cpu, [1.0], pxx_cpu)  # a参数改为数组 [1.0]

    # 频域插值（关键修复部分）
    kNyq = int(cp.floor(nSamples / 2).get()) + 1
    posFreq = cp.arange(0, kNyq) * Fs / nSamples
    posFreq_cpu = cp.asnumpy(posFreq)

    # 获取原始频率的最大值并限制新频率范围
    f_cpu = cp.asnumpy(f)
    f_max = f_cpu[-1]
    posFreq_cpu = np.clip(posFreq_cpu, None, f_max)

    # 创建插值函数并应用
    interp_func = interp1d(f_cpu, pxxSmth, bounds_error=False, fill_value=pxxSmth[-1])
    psdPosFreq = interp_func(posFreq_cpu)
    psdPosFreq_gpu = cp.asarray(psdPosFreq)

    # PSO参数设置（根据下游代码需求选择是否转CPU）
    inParams = {
        'dataX': cp.asnumpy(t),  # 如果 PSO 模块需要 CPU 数据
        'dataY': cp.asnumpy(dataY),
        'psdPosFreq': psdPosFreq_gpu,
        'sampFreq': Fs,
        'rmin': [-2, 0, 0, 0, 4, 0],
        'rmax': [4, 3, 10, 2 * np.pi, 14, 10],  # 使用 np.pi 而非 cp.pi
    }

    # 运行PSO（假设PSO模块已GPU加速）
    from PSO_GPU_main import crcbqcpsopsd

    nRuns = 8
    outResults, outStruct = crcbqcpsopsd(inParams, {'maxSteps': 10000}, nRuns)

    # 转换为CPU数据用于绘图
    t_cpu = cp.asnumpy(t)
    dataY_cpu = cp.asnumpy(dataY)
    bestSig_cpu = cp.asnumpy(outResults['bestSig'])
    estSigs_cpu = [cp.asnumpy(outResults['allRunsOutput'][i]['estSig']) for i in range(nRuns)]

    # 绘制信号图
    plt.figure(dpi=200, figsize=(12, 6))
    plt.scatter(t_cpu, dataY_cpu, s=5, label='Analysis Data', alpha=0.5)

    for i in range(nRuns):
        plt.plot(t_cpu, estSigs_cpu[i],
                 color=(51 / 255, 255 / 255, 153 / 255, 0.3),
                 linewidth=0.8,
                 label='Estimated Signal' if i == 0 else None)

    plt.plot(t_cpu, bestSig_cpu, 'r', linewidth=1.5, label='Best Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig('gpu_signal_comparison.png', bbox_inches='tight')
    plt.close()


    # 打印估计参数（转换所有GPU数据到CPU）
    def convert_to_cpu(data):
        if isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data


    print('\nEstimated parameters:')
    params = {
        'bestFitness': convert_to_cpu(outResults['bestFitness']),
        'r': 10 ** convert_to_cpu(outResults['r']),
        'm_c': 10 ** convert_to_cpu(outResults['m_c']),
        'tc': convert_to_cpu(outResults['tc']),
        'phi_c': convert_to_cpu(outResults['phi_c']) / cp.pi,
        'mlz': 10 ** convert_to_cpu(outResults['mlz']),
        'y': convert_to_cpu(outResults['y']),
        'SNR': cp.sqrt(-outResults['bestFitness']).get()
    }

    print(f"bestFitness = {params['bestFitness']:.4f}")
    print(f"r = {params['r']:.4f} Mpc")
    print(f"m_c = {params['m_c']:.4f} M_sun")
    print(f"tc = {params['tc']:.4f} s")
    print(f"phi_c = {params['phi_c']:.4f} π")
    print(f"mlz = {params['mlz']:.4e} M_sun")
    print(f"y = {params['y']:.4f}")
    print(f"SNR = {params['SNR']:.2f}")

    # 各次运行结果
    print('\nIndividual runs results:')
    for lpruns in range(nRuns):
        run = outResults['allRunsOutput'][lpruns]
        print(f"\nRun #{lpruns + 1}:")
        print(f"bestFitness = {convert_to_cpu(outStruct[lpruns]['bestFitness']):.4f}")
        print(f"r = {10 ** convert_to_cpu(run['r']):.4f} Mpc")
        print(f"m_c = {10 ** convert_to_cpu(run['m_c']):.4f} M_sun")
        print(f"tc = {convert_to_cpu(run['tc']):.4f} s")
        print(f"phi_c = {convert_to_cpu(run['phi_c']) / cp.pi:.4f} π")
        print(f"mlz = {10 ** convert_to_cpu(run['mlz']):.4e} M_sun")
        print(f"y = {convert_to_cpu(run['y']):.4f}")
        print(f"SNR = {cp.sqrt(-outStruct[lpruns]['bestFitness']).get():.2f}")

    # 保存结果
    cp.save('gpu_results.npy', outResults)
    cp.save('gpu_struct.npy', outStruct)