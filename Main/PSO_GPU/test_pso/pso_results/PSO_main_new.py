import cupy as cp
from cupyx.scipy.fftpack import fft
from tqdm import tqdm
import numpy as np
import pycbc.types
from pycbc.filter import match
import scipy.constants as const
import time

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
    'ssrqc',
    'normsig4psd',
    'innerprodpsd',
    's2rv',
    'crcbchkstdsrchrng',
    'calculate_snr',
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
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1) # 计划删除进行检查
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdHigh']) # 计划删除进行检查
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

    bestRun = np.argmin(fitVal_np)
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
    # 默认PSO参数，实则无意义
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
        'bestFitness': cp.inf,
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
    gbest_idx = cp.argmin(pbest_fitness)
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

                # 使用numpy的argmin而不是cupy的argmin，因为neighbors是Python列表
                neighbor_fitness = [float(pbest_fitness[n]) for n in neighbors]
                best_neighbor_idx = np.argmin(neighbor_fitness)
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

                # 更新个体最佳
                if new_fitness < pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = new_fitness

            # 更新全局最佳
            current_best_idx = cp.argmin(pbest_fitness)
            if pbest_fitness[current_best_idx] < gbest_fitness:
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
    改进的适应度函数计算
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
    fitVal = cp.full(nPoints, cp.inf)

    # 将标准范围[0,1]转换为实际参数范围
    xVecReal = s2rv(xVec, params)

    # 计算每个有效点的适应度
    for i in range(nPoints):
        if validPts[i]:
            fitVal[i] = ssrqc(xVecReal[i], params)

    if returnxVec:
        return fitVal, xVecReal
    else:
        return fitVal


def ssrqc(x, params):
    """
    计算信号的平方和残差（负对数似然）
    """
    # 生成信号
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5])

    # 归一化信号
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # 计算内积（投影）
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdHigh'])

    # 返回负的平方内积（最小化问题）
    return -cp.abs(inPrd) ** 2

def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    根据PSD归一化信号
    """
    nSamples = len(sigVec)

    # 确保PSD向量长度正确
    if len(psdVec) != nSamples // 2 + 1:
        # 调整PSD向量长度
        psdVec_adjusted = cp.zeros(nSamples // 2 + 1)
        min_len = min(len(psdVec), nSamples // 2 + 1)
        psdVec_adjusted[:min_len] = psdVec[:min_len]
        psdVec = psdVec_adjusted

    # 构建完整的PSD向量（正负频率）
    if psdVec.shape[0] > 1:  # 确保有多于一个元素
        psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    else:
        # 处理特殊情况
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[0] = psdVec[0]

    # 避免除以零
    psdVec4Norm = cp.maximum(psdVec4Norm, 1e-47)

    # 计算信号的归一化因子
    fft_sig = cp.fft.fft(sigVec)

    # 确保长度匹配
    if len(fft_sig) > len(psdVec4Norm):
        psdVec4Norm = cp.pad(psdVec4Norm, (0, len(fft_sig) - len(psdVec4Norm)), 'constant',
                             constant_values=psdVec4Norm[-1])
    elif len(fft_sig) < len(psdVec4Norm):
        psdVec4Norm = psdVec4Norm[:len(fft_sig)]

    # 计算归一化平方和
    normSigSqrd = cp.sum((cp.abs(fft_sig) ** 2) / psdVec4Norm) / (sampFreq * nSamples)

    # 计算归一化因子
    normFac = snr / cp.sqrt(cp.abs(normSigSqrd))  # 使用绝对值避免复数问题

    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """
    计算考虑PSD的内积
    """
    # 确保输入向量长度一致
    if len(xVec) != len(yVec):
        # 调整长度使其匹配
        min_len = min(len(xVec), len(yVec))
        xVec = xVec[:min_len]
        yVec = yVec[:min_len]

    nSamples = len(xVec)

    # 计算FFT
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)

    # 准备PSD向量
    if len(psdVals) != nSamples // 2 + 1:
        # 调整PSD向量长度
        psdVals_adjusted = cp.zeros(nSamples // 2 + 1)
        min_len = min(len(psdVals), nSamples // 2 + 1)
        psdVals_adjusted[:min_len] = psdVals[:min_len]
        psdVals = psdVals_adjusted

    # 构建完整的PSD向量
    if psdVals.shape[0] > 1:  # 确保有多于一个元素
        psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    else:
        # 处理特殊情况
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[0] = psdVals[0]

    # 确保长度匹配
    if len(fftX) > len(psdVec4Norm):
        psdVec4Norm = cp.pad(psdVec4Norm, (0, len(fftX) - len(psdVec4Norm)), 'constant',
                             constant_values=psdVec4Norm[-1])
    elif len(fftX) < len(psdVec4Norm):
        psdVec4Norm = psdVec4Norm[:len(fftX)]
    # 计算内积
    inner_product = cp.sum((fftX * cp.conj(fftY)) / psdVec4Norm) / (sampFreq * nSamples)

    # 返回实部
    return cp.real(inner_product)


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

# 定义新的SNR计算函数
def calculate_snr(signal, data, psd, sampFreq):

    # 计算信号与数据的匹配度
    match_value = innerprodpsd(signal, data, sampFreq, psd)

    # 信号自身的内积
    signal_norm = cp.sqrt(innerprodpsd(signal, signal, sampFreq, psd))

    # 真实SNR是信号能量的平方根
    snr = match_value / signal_norm

    return cp.abs(snr)


def analyze_mismatch(data, h_lens, samples, psdHigh):
    # Convert inputs to CuPy arrays
    data_cupy = cp.asarray(data)
    h_lens_cupy = cp.asarray(h_lens)
    # Calculate match value
    match_value = (innerprodpsd(h_lens_cupy, data_cupy, samples, psdHigh) /
                   cp.sqrt(innerprodpsd(h_lens_cupy, h_lens_cupy, samples, psdHigh) *
                           innerprodpsd(data_cupy, data_cupy, samples, psdHigh)))

    # Calculate mismatch
    epsilon = 1 - match_value
    return epsilon


def classify_signal(snr, flux_ratio, time_delay, total_mass):

    flux_threshold = 2 * (snr ** (-2))
    inverse_mass = 1/ (2 ** (4/5) * total_mass)  # 质量的倒数（用于时间延迟阈值）

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
