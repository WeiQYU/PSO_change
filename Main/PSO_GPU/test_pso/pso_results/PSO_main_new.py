import cupy as cp
from cupyx.scipy.fftpack import fft
from lalburst.power import psds_from_job_length
from tqdm import tqdm
import numpy as np
import scipy.constants as const
import time
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries, FrequencySeries

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30  # Solar mass, kg
pc = 3.086e16  # Parsec to meters

__all__ = [
    'crcbqcpsopsd',
    'crcbpso',
    'crcbgenqcsig',
    'glrtqcsig4pso',
    'calculate_snr_pycbc',
    'normsig4psd',
    'innerprodpsd',
    's2rv',
    'crcbchkstdsrchrng',
    'classify_signal',
    'analyze_mismatch',
    'classify_signal'
]


def crcbqcpsopsd(inParams, psoParams, nRuns):
    """
    对多个PSO运行进行管理的主函数
    """
    # Transfer data to GPU
    inParams['dataX'] = cp.asarray(inParams['dataX'])
    inParams['dataY'] = cp.asarray(inParams['dataY'])
    inParams['psdHigh'] = cp.asarray(inParams['psdHigh'])
    inParams['rmax'] = cp.asarray(inParams['rmax'])
    inParams['rmin'] = cp.asarray(inParams['rmin'])

    nSamples = len(inParams['dataX'])
    nDim = 6
    # 创建适应度函数句柄
    fHandle = lambda x, returnxVec: glrtqcsig4pso(x, inParams, returnxVec)

    outStruct = [{} for _ in range(nRuns)]
    outResults = {
        'allRunsOutput': [],
        'bestRun': None,
        'bestFitness': None,
        'bestSig': cp.zeros(nSamples),
        'r': None,
        'm_c': None,
        'tc': None,
        'phi_c': None,
        'A': None,
        'delta_t': None,
    }

    # 运行多次PSO优化，每次使用不同的随机初始种子
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1
        # 设置不同的随机种子，确保多次运行结果不同
        cp.random.seed(int(time.time()) + lpruns * 1000)
        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)
        # 打印每次运行的最佳适应度
        print(f"Run {lpruns + 1} completed with best fitness: {outStruct[lpruns]['bestFitness']}")

    # 处理所有运行的结果
    fitVal = cp.zeros(nRuns)
    for lpruns in range(nRuns):
        allRunsOutput = {
            'fitVal': 0,
            'r': 0,
            'm_c': 0,
            'tc': 0,
            'phi_c': 0,
            'A': 0,
            'delta_t': 0,
            'estSig': cp.zeros(nSamples),
            'totalFuncEvals': [],
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']

        # 确保维度处理正确
        bestLocation = cp.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)  # 确保2D形状 (1, nDim)

        # 获取最佳位置的参数
        _, params = fHandle(bestLocation, returnxVec=1)

        # 处理参数维度
        if isinstance(params, list) and len(params) > 0:
            params = params[0]
        elif isinstance(params, cp.ndarray) and params.ndim > 1 and params.shape[0] == 1:
            params = params[0]

        # 如果需要，转换为numpy
        if isinstance(params, cp.ndarray):
            params = cp.asnumpy(params)

        r, m_c, tc, phi_c, A, delta_t = params

        # 生成最佳信号
        estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t)

        # 对信号进行归一化处理
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdHigh'])
        estSig = estAmp * estSig

        # 更新输出结果
        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns].get()) if hasattr(fitVal[lpruns], 'get') else float(fitVal[lpruns]),
            'r': r,
            'm_c': m_c,
            'tc': tc,
            'phi_c': phi_c,
            'A': A,
            'delta_t': delta_t,
            'estSig': cp.asarray(estSig),
            'totalFuncEvals': outStruct[lpruns]['totalFuncEvals']
        })
        outResults['allRunsOutput'].append(allRunsOutput)

    # 找出最佳运行
    if hasattr(fitVal, 'get'):
        fitVal_np = cp.asnumpy(fitVal)
    else:
        fitVal_np = fitVal

    bestRun = np.argmax(fitVal_np)
    outResults.update({
        'bestRun': int(bestRun),
        'bestFitness': outResults['allRunsOutput'][bestRun]['fitVal'],
        'bestSig': outResults['allRunsOutput'][bestRun]['estSig'],
        'r': outResults['allRunsOutput'][bestRun]['r'],
        'm_c': outResults['allRunsOutput'][bestRun]['m_c'],
        'tc': outResults['allRunsOutput'][bestRun]['tc'],
        'phi_c': outResults['allRunsOutput'][bestRun]['phi_c'],
        'A': outResults['allRunsOutput'][bestRun]['A'],
        'delta_t': outResults['allRunsOutput'][bestRun]['delta_t'],
    })
    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """
    完全重写的PSO核心算法实现
    """
    # 默认PSO参数
    psoParams = {
        'popsize': 200,
        'maxSteps': 2000,
        'c1': 2.0,  # 个体学习因子
        'c2': 2.0,  # 社会学习因子
        'max_velocity': 0.5,  # 最大速度限制
        'w_start': 0.9,  # 初始惯性权重
        'w_end': 0.4,  # 最终惯性权重
        'run': 1,  # 运行序号
        'nbrhdSz': 4  # 邻域大小
    }
    # 更新参数
    psoParams.update(kwargs)

    # 确保随机数重现性
    if 'seed' in psoParams:
        cp.random.seed(psoParams['seed'])

    # 返回数据结构初始化
    returnData = {
        'totalFuncEvals': 0,
        'bestLocation': cp.zeros((1, nDim)),
        'bestFitness': -cp.inf,  # 初始化为负无穷，因为我们要最大化SNR
        'fitnessHistory': []  # 记录历史适应度
    }

    # 初始化粒子群
    particles = cp.random.rand(psoParams['popsize'], nDim)  # 位置：均匀分布在[0,1]之间
    velocities = cp.random.uniform(-0.1, 0.1, (psoParams['popsize'], nDim))  # 速度：小的随机值

    # 评估初始适应度
    fitness = cp.zeros(psoParams['popsize'])
    for i in range(psoParams['popsize']):
        fitness[i] = fitfuncHandle(particles[i:i + 1], returnxVec=0)

    # 初始化个体最佳和全局最佳
    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    # 找出全局最佳
    gbest_idx = cp.argmax(pbest_fitness)  # 使用argmax因为我们要最大化SNR
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx].copy()

    # 记录初始适应度
    returnData['fitnessHistory'].append(float(gbest_fitness))

    total_evals = psoParams['popsize']  # 计数器：已评估的适应度次数

    # 创建进度条
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # 更新惯性权重 - 线性递减
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

            # 更新每个粒子
            for i in range(psoParams['popsize']):
                # 获取局部最佳（环形拓扑）
                neighbors = []
                for j in range(psoParams['nbrhdSz']):
                    idx = (i + j) % psoParams['popsize']
                    neighbors.append(idx)

                # 使用numpy的argmax而不是cupy的argmax，因为neighbors是Python列表
                neighbor_fitness = [float(pbest_fitness[n]) for n in neighbors]
                best_neighbor_idx = np.argmax(neighbor_fitness)  # 使用argmax因为我们要最大化SNR
                lbest_idx = neighbors[best_neighbor_idx]
                lbest = pbest[lbest_idx].copy()

                # 生成随机系数
                r1 = cp.random.rand(nDim)
                r2 = cp.random.rand(nDim)

                # 更新速度
                velocities[i] = (w * velocities[i] +
                                 psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                 psoParams['c2'] * r2 * (lbest - particles[i]))

                # 限制速度
                velocities[i] = cp.clip(velocities[i], -psoParams['max_velocity'], psoParams['max_velocity'])

                # 更新位置
                particles[i] += velocities[i]

                # 处理边界约束 - 反射边界
                # 如果位置超出边界，反弹回来并反转速度方向
                out_low = particles[i] < 0
                out_high = particles[i] > 1

                particles[i] = cp.where(out_low, -particles[i], particles[i])
                particles[i] = cp.where(out_high, 2 - particles[i], particles[i])

                # 确保位置在[0,1]范围内（防止数值误差）
                particles[i] = cp.clip(particles[i], 0, 1)

                # 在边界处反转速度
                velocities[i] = cp.where(out_low | out_high, -velocities[i], velocities[i])

                # 评估新位置
                new_fitness = fitfuncHandle(particles[i:i + 1], returnxVec=0)
                fitness[i] = new_fitness
                total_evals += 1

                # 更新个体最佳 - 最大化SNR
                if new_fitness > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = new_fitness

            # 更新全局最佳 - 最大化SNR
            current_best_idx = cp.argmax(pbest_fitness)
            if pbest_fitness[current_best_idx] > gbest_fitness:
                gbest = pbest[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx].copy()

                # 更新进度条信息
                pbar.set_postfix({'fitness': float(gbest_fitness)})

            # 记录每一步的最佳适应度
            returnData['fitnessHistory'].append(float(gbest_fitness))

    # 完成后更新返回数据
    returnData.update({
        'totalFuncEvals': total_evals,
        'bestLocation': cp.asnumpy(gbest.reshape(1, -1)),
        'bestFitness': float(gbest_fitness)
    })

    return returnData


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t):
    """
    生成引力波信号，包括引力透镜效应
    """
    # 转换参数单位
    r = (10 ** r) * 1e6 * pc  # 距离（米）
    m_c = (10 ** m_c) * M_sun  # 组合质量（kg）
    delta_t = 10 ** delta_t  # 时间延迟（秒）

    # 确保输入是CuPy数组
    if not isinstance(dataX, cp.ndarray):
        dataX_gpu = cp.asarray(dataX)
    else:
        dataX_gpu = dataX

    # 生成引力波信号
    t = dataX_gpu  # 时间序列

    # 在合并前的有效区域计算信号
    valid_idx = t < tc
    t_valid = t[valid_idx]

    # 初始化波形
    h = cp.zeros_like(t)

    if cp.sum(valid_idx) > 0:  # 确保有有效区域
        # 计算频率演化参数 Theta
        Theta = c ** 3 * (tc - t_valid) / (5 * G * m_c)

        # 计算振幅
        A_gw = (G * m_c / (c ** 2 * r)) * Theta ** (-1 / 4)

        # 计算相位
        phase = 2 * phi_c - 2 * Theta ** (5 / 8)

        # 生成波形
        h[valid_idx] = A_gw * cp.cos(phase)

    # 转换到频域应用引力透镜效应
    h_f = cp.fft.rfft(h)
    freqs = cp.fft.rfftfreq(len(h), t[1] - t[0])

    # 应用引力透镜效应
    phi = 2 * cp.pi * freqs * delta_t
    F_f = 1 + A * cp.exp(1j * phi)
    h_lens_f = h_f * F_f

    # 转回时域
    h_lens = cp.fft.irfft(h_lens_f)

    return h_lens


def glrtqcsig4pso(xVec, params, returnxVec=0):
    """
    改进的适应度函数计算，使用PyCBC的SNR计算方法
    """
    # 确保输入是CuPy数组
    if isinstance(xVec, np.ndarray):
        xVec = cp.asarray(xVec)

    # 确保输入维度正确
    if xVec.ndim == 1:
        xVec = xVec.reshape(1, -1)

    # 检查参数是否在有效范围内
    validPts = crcbchkstdsrchrng(xVec)
    nPoints = xVec.shape[0]

    # 初始化适应度数组
    fitVal = cp.full(nPoints, -cp.inf)  # 使用负无穷作为默认值，因为我们要最大化SNR

    # 将标准范围[0,1]转换为实际参数范围
    xVecReal = s2rv(xVec, params)

    # 计算每个有效点的适应度
    for i in range(nPoints):
        if validPts[i]:
            # 生成信号
            qc = crcbgenqcsig(params['dataX'], xVecReal[i, 0], xVecReal[i, 1],
                              xVecReal[i, 2], xVecReal[i, 3], xVecReal[i, 4], xVecReal[i, 5])

            # 对信号进行归一化
            qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

            # 计算内积
            inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdHigh'])

            # 适应度采用内积的平方（与原始代码保持一致）
            fitVal[i] = cp.abs(inPrd) ** 2

    if returnxVec:
        return fitVal, xVecReal
    else:
        return fitVal


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    基于PSD对信号进行归一化

    参数:
    sigVec -- 输入信号
    sampFreq -- 采样频率
    psdVec -- 功率谱密度值
    snr -- 目标信噪比

    返回:
    normalized_signal -- 归一化后的信号
    norm_factor -- 归一化因子
    """
    nSamples = len(sigVec)
    psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    normSigSqrd = cp.sum((cp.fft.fft(sigVec) / psdVec4Norm) * cp.conj(cp.fft.fft(sigVec))) / (sampFreq * nSamples)
    normFac = snr / cp.sqrt(normSigSqrd)
    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """
    计算两个信号在给定PSD下的内积

    参数:
    xVec -- 第一个信号
    yVec -- 第二个信号
    sampFreq -- 采样频率
    psdVals -- 功率谱密度值

    返回:
    inner_product -- 内积值
    """
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)
    psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    return cp.real(cp.sum((fftX / psdVec4Norm) * cp.conj(fftY)) / (sampFreq * len(xVec)))


def s2rv(xVec, params):
    """
    将标准范围 [0,1] 的参数转换到实际范围
    """
    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    """
    检查粒子是否在标准范围 [0,1] 内
    """
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)

    # 检查每行中的所有元素是否都在[0,1]范围内
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)


# PyCBC SNR calculation function
def calculate_snr_pycbc(signal, psd, fs):
    """
    Uses PyCBC's matched_filter to calculate SNR
    """
    # Create PyCBC TimeSeries object
    signal_np = cp.asnumpy(signal) if isinstance(signal, cp.ndarray) else signal
    delta_t = 1.0 / fs
    ts_signal = TimeSeries(signal_np, delta_t=delta_t)

    # Create PyCBC FrequencySeries object
    delta_f = 1.0 / (len(signal_np) * delta_t)
    # Ensure PSD length matches frequency points
    psd_np = cp.asnumpy(psd) if isinstance(psd, cp.ndarray) else psd

    # 确保PSD长度匹配
    psd_length = len(signal_np) // 2 + 1

    if len(psd_np) > psd_length:
        psd_np = psd_np[:psd_length]
    elif len(psd_np) < psd_length:
        # Extend PSD
        extended_psd = np.ones(psd_length) * psd_np[-1]
        extended_psd[:len(psd_np)] = psd_np
        psd_np = extended_psd

    psd_series = FrequencySeries(psd_np, delta_f=delta_f)

    # Use matched_filter to calculate SNR
    try:
        # Match template with signal (both the same)
        snr = matched_filter(ts_signal, ts_signal, psd=psd_series, low_frequency_cutoff=10.0)
        # Get maximum SNR value
        max_snr = abs(snr).max()
        return float(max_snr)
    except Exception as e:
        print(f"PyCBC method failed: {e}")
        return 0.0  # Return 0 if calculation fails


def analyze_mismatch(data, h_lens, samples, psdHigh):
    """
    计算信号匹配度的失配
    """
    # 创建PyCBC TimeSeries对象
    data_np = cp.asnumpy(data) if isinstance(data, cp.ndarray) else data
    h_lens_np = cp.asnumpy(h_lens) if isinstance(h_lens, cp.ndarray) else h_lens

    delta_t = 1.0 / samples
    ts_data = TimeSeries(data_np, delta_t=delta_t)
    ts_signal = TimeSeries(h_lens_np, delta_t=delta_t)

    # 创建PyCBC FrequencySeries对象
    delta_f = 1.0 / (len(data_np) * delta_t)
    psd_np = cp.asnumpy(psdHigh) if isinstance(psdHigh, cp.ndarray) else psdHigh
    psd_length = len(data_np) // 2 + 1

    if len(psd_np) > psd_length:
        psd_np = psd_np[:psd_length]
    elif len(psd_np) < psd_length:
        extended_psd = np.ones(psd_length) * psd_np[-1]
        extended_psd[:len(psd_np)] = psd_np
        psd_np = extended_psd

    psd_series = FrequencySeries(psd_np, delta_f=delta_f)

    # 计算匹配度
    try:
        from pycbc.filter import match
        match_value, _ = match(ts_signal, ts_data, psd=psd_series, low_frequency_cutoff=10.0)
        # 计算失配
        epsilon = 1 - match_value
        return float(epsilon)
    except Exception as e:
        print(f"PyCBC match calculation failed: {e}")
        return 1.0  # 返回最大失配值


def classify_signal(snr, flux_ratio, time_delay, total_mass):
    """
    根据SNR、流量比和时间延迟对信号进行分类
    """
    flux_threshold = 2 * (snr ** (-2))
    inverse_mass = 1 / (2 ** (4 / 5) * total_mass)  # 质量的倒数（用于时间延迟阈值）

    # 分类标准
    if snr < 8:
        classification = "Pure Noise (SNR too low)"
        is_lensed = False
    else:
        if flux_ratio >= flux_threshold and time_delay >= inverse_mass:
            classification = "Lensed Signal (matches both criteria)"
            is_lensed = True
        elif flux_ratio >= flux_threshold:
            classification = "Potential Lensed Signal (matches flux ratio criterion only,I)"
            is_lensed = False
        elif time_delay >= inverse_mass:
            classification = "Potential Lensed Signal (matches time delay criterion only,Δtd)"
            is_lensed = False
        else:
            classification = "Unlensed Signal (doesn't meet lensing criteria)"
            is_lensed = False

    return classification, flux_threshold, inverse_mass, is_lensed