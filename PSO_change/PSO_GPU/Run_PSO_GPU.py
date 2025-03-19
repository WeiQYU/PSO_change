import cupy as cp
from cupyx.scipy.fft import fft, fftfreq
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
import scipy.io as scio
import matplotlib
import numpy as np
# 运行PSO（假设PSO模块已GPU加速）
from PSO_GPU_main import crcbqcpsopsd

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
    # 生成频率数组
    freqs = cp.fft.fftfreq(nperseg, 1 / fs)[:nperseg // 2 + 1]
    pxx = cp.zeros(nperseg // 2 + 1, dtype=cp.float64)
    for i in range(nseg):
        seg = x[i * nperseg: (i + 1) * nperseg]
        seg = (seg - cp.mean(seg)) * window
        fft_vals = fft(seg)[:nperseg // 2 + 1]
        # 确保使用实部的平方加虚部的平方计算功率
        pxx += (cp.abs(fft_vals) ** 2) / (fs * cp.sum(window ** 2))
    return freqs, pxx / nseg  # 返回频率和PSD数组


if __name__ == '__main__':
    # 加载数据并转换到GPU
    TrainingData = scio.loadmat('../generate_ligo/noise.mat')
    analysisData = scio.loadmat('../generate_ligo/data.mat')
    dataY = cp.asarray(analysisData['data'][0])
    Fs = analysisData['samples'][0].item()
    nSamples = dataY.size

    # 生成时间序列
    t = cp.linspace(-40, 10, nSamples)
    T = nSamples / Fs

    # PSD估计
    noise_gpu = cp.asarray(TrainingData['noise'][0])
    f, pxx = gpu_welch(noise_gpu, Fs, nperseg=int(Fs // 2))

    # 平滑处理
    smthOrdr = 10
    b_gpu = cp.ones(smthOrdr) / smthOrdr
    b_cpu = cp.asnumpy(b_gpu)  # 显式转换为 NumPy
    pxx_cpu = cp.asnumpy(pxx)  # 转换为 NumPy

    # 确保pxx_cpu是实数
    if np.iscomplexobj(pxx_cpu):
        pxx_cpu = np.abs(pxx_cpu)  # 使用绝对值确保是实数

    pxxSmth = filtfilt(b_cpu, [1.0], pxx_cpu)  # a参数改为数组 [1.0]

    # 频域插值
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

    # PSO参数设置
    inParams = {
        'dataX': cp.asnumpy(t),  # 如果 PSO 模块需要 CPU 数据
        'dataY': cp.asnumpy(dataY),
        'psdPosFreq': cp.asnumpy(psdPosFreq_gpu),  # 确保传递给PSO的是CPU数据
        'sampFreq': Fs,
        'rmin': [-2, 0, 0, 0, 4, 0],
        'rmax': [4, 3, 10, 2 * np.pi, 14, 10],  # 使用 np.pi 而非 cp.pi
    }

    nRuns = 8
    outResults, outStruct = crcbqcpsopsd(inParams, {'maxSteps': 4000}, nRuns)

    # 转换为CPU数据用于绘图
    t_cpu = cp.asnumpy(t)
    dataY_cpu = cp.asnumpy(dataY)

    # 确保bestSig是CPU数据，如果已经是则不需要转换
    if hasattr(outResults['bestSig'], 'get'):
        bestSig_cpu = cp.asnumpy(outResults['bestSig'])
    else:
        bestSig_cpu = outResults['bestSig']

    # 处理所有运行的估计信号
    estSigs_cpu = []
    for i in range(nRuns):
        estSig = outResults['allRunsOutput'][i]['estSig']
        # 检查是否需要从GPU转换
        if hasattr(estSig, 'get'):
            estSigs_cpu.append(cp.asnumpy(estSig))
        else:
            estSigs_cpu.append(estSig)

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

    plt.figure(dpi=200, figsize=(12, 6))
    plt.plot(t_cpu, bestSig_cpu, 'r', linewidth=1.5, label='Best Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('best_signal.png', bbox_inches='tight')
    plt.close()


    def convert_to_cpu(data):
        """将数据从GPU转移到CPU，并确保是实数"""
        if isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)

        # 确保数据是实数
        if np.iscomplexobj(data):
            data = np.abs(data)  # 使用绝对值

        return data


    print('\nEstimated parameters:')
    # 先获取并转换所有参数
    r_value = convert_to_cpu(outResults['r'])
    m_c_value = convert_to_cpu(outResults['m_c'])
    tc_value = convert_to_cpu(outResults['tc'])
    phi_c_value = convert_to_cpu(outResults['phi_c'])
    mlz_value = convert_to_cpu(outResults['mlz'])
    y_value = convert_to_cpu(outResults['y'])
    bestFitness_value = convert_to_cpu(outResults['bestFitness'])

    # 计算SNR，确保是实数
    if np.isscalar(bestFitness_value) and bestFitness_value < 0:
        snr_value = np.sqrt(-bestFitness_value)
    else:
        # 如果bestFitness不是负数，使用绝对值
        snr_value = np.sqrt(np.abs(bestFitness_value))

    params = {
        'bestFitness': bestFitness_value,
        'r': 10 ** r_value,
        'm_c': 10 ** m_c_value,
        'tc': tc_value,
        'phi_c': phi_c_value / np.pi,  # 使用np.pi而非cp.pi
        'mlz': 10 ** mlz_value,
        'y': y_value,
        'SNR': snr_value
    }

    print(f"bestFitness = {params['bestFitness']:.4f}")
    print(f"r = {params['r']:.4f}")
    print(f"m_c = {params['m_c']:.4f}")
    print(f"tc = {params['tc']:.4f}")
    print(f"phi_c = {params['phi_c']:.4f}")
    print(f"mlz = {params['mlz']:.4e}")
    print(f"y = {params['y']:.4f}")
    print(f"SNR = {params['SNR']:.2f}")

    # 各次运行结果
    print('\nIndividual runs results:')
    for lpruns in range(nRuns):
        run = outResults['allRunsOutput'][lpruns]

        # 获取并转换当前运行的参数
        run_r = convert_to_cpu(run['r'])
        run_m_c = convert_to_cpu(run['m_c'])
        run_tc = convert_to_cpu(run['tc'])
        run_phi_c = convert_to_cpu(run['phi_c'])
        run_mlz = convert_to_cpu(run['mlz'])
        run_y = convert_to_cpu(run['y'])
        run_bestFitness = convert_to_cpu(outStruct[lpruns]['bestFitness'])

        # 计算SNR，确保是实数
        if np.isscalar(run_bestFitness) and run_bestFitness < 0:
            run_snr = np.sqrt(-run_bestFitness)
        else:
            run_snr = np.sqrt(np.abs(run_bestFitness))

        print(f"\nRun #{lpruns + 1}:")
        print(f"bestFitness = {run_bestFitness:.4f}")
        print(f"r = {10 ** run_r:.4f}")
        print(f"m_c = {10 ** run_m_c:.4f}")
        print(f"tc = {run_tc:.4f}")
        print(f"phi_c = {run_phi_c / np.pi:.4f}")  # 使用np.pi
        print(f"mlz = {10 ** run_mlz:.4e}")
        print(f"y = {run_y:.4f}")
        print(f"SNR = {run_snr:.2f}")

    # 保存结果
    # 确保保存前转换为CPU数据
    outResults_cpu = {}
    for key, value in outResults.items():
        if key == 'allRunsOutput':
            outResults_cpu[key] = []
            for run in value:
                run_cpu = {k: convert_to_cpu(v) for k, v in run.items()}
                outResults_cpu[key].append(run_cpu)
        else:
            outResults_cpu[key] = convert_to_cpu(value)

    outStruct_cpu = [convert_to_cpu(item) for item in outStruct]

    np.save('gpu_results.npy', outResults_cpu)
    np.save('gpu_struct.npy', outStruct_cpu)