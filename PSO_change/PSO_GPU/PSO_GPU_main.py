import cupy as cp
from cupyx.scipy.fftpack import fft
from tqdm import tqdm
from scipy.io import savemat
import numpy as np

# 常量定义
G = 6.67430e-11  # 万有引力常数, m^3 kg^-1 s^-2
c = 2.998e8  # 光速, m/s
M_sun = 1.989e30  # 太阳质量, kg
pc = 3.086e16  # pc到m的转换

__all__ = ['crcbqcpsopsd', 'crcbpso', 'crcbgenqcsig', 'glrtqcsig4pso',
           'ssrqc', 'normsig4psd', 'innerprodpsd', 's2rv', 'crcbchkstdsrchrng']


def crcbqcpsopsd(inParams, psoParams, nRuns):
    # 将输入数据转移到GPU
    inParams['dataX'] = cp.asarray(inParams['dataX'])
    inParams['dataY'] = cp.asarray(inParams['dataY'])
    inParams['psdPosFreq'] = cp.asarray(inParams['psdPosFreq'])
    inParams['rmax'] = cp.asarray(inParams['rmax'])  # 新增
    inParams['rmin'] = cp.asarray(inParams['rmin'])  # 新增

    nSamples = len(inParams['dataX'])
    nDim = 6
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
        'mlz': None,
        'y': None,
    }

    for lpruns in range(nRuns):
        cp.random.seed(lpruns)
        outStruct[lpruns] = crcbpso(fHandle, nDim, **psoParams)

    # 在CPU处理最终结果
    fitVal = cp.zeros(nRuns)
    for lpruns in range(nRuns):
        allRunsOutput = {
            'fitVal': 0,
            'r': 0, 'm_c': 0, 'tc': 0,
            'phi_c': 0, 'mlz': 0, 'y': 0,
            'estSig': cp.zeros(nSamples),
            'totalFuncEvals': [],
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']
        _, params = fHandle(outStruct[lpruns]['bestLocation'][cp.newaxis, ...], returnxVec=1)

        # 将GPU数据转回CPU
        params = cp.asnumpy(params[0])
        r, m_c, tc, phi_c, mlz, y = params

        estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, mlz, y)
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdPosFreq'], 1)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdPosFreq'])
        estSig = estAmp * estSig

        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns].get()),
            'r': r, 'm_c': m_c, 'tc': tc,
            'phi_c': phi_c, 'mlz': mlz, 'y': y,
            'estSig': cp.asnumpy(estSig),
            'totalFuncEvals': outStruct[lpruns]['totalFuncEvals']
        })
        outResults['allRunsOutput'].append(allRunsOutput)

    bestRun = cp.argmin(fitVal).get()
    outResults.update({
        'bestRun': int(bestRun),
        'bestFitness': outResults['allRunsOutput'][bestRun]['fitVal'],
        'bestSig': outResults['allRunsOutput'][bestRun]['estSig'],
        'r': outResults['allRunsOutput'][bestRun]['r'],
        'm_c': outResults['allRunsOutput'][bestRun]['m_c'],
        'tc': outResults['allRunsOutput'][bestRun]['tc'],
        'phi_c': outResults['allRunsOutput'][bestRun]['phi_c'],
        'mlz': outResults['allRunsOutput'][bestRun]['mlz'],
        'y': outResults['allRunsOutput'][bestRun]['y'],
    })
    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, O=0, **varargin):
    psoParams = {
        'popsize': 40, 'maxSteps': 2000,
        'c1': 2, 'c2': 2, 'max_velocity': 0.5,
        'dcLaw_a': 0.9, 'dcLaw_b': 0.4,
        'dcLaw_c': 2000 - 1, 'dcLaw_d': 0.2,
        'bndryCond': '', 'nbrhdSz': 3
    }
    psoParams.update(varargin)

    returnData = {
        'totalFuncEvals': [],
        'bestLocation': cp.zeros((1, nDim)),
        'bestFitness': [],
    }

    # GPU矩阵初始化
    partCoordCols = slice(0, nDim)
    partVelCols = slice(nDim, 2 * nDim)
    partPbestCols = slice(2 * nDim, 3 * nDim)
    partFitPbestCols = 3 * nDim
    partFitCurrCols = partFitPbestCols + 1
    partFitLbestCols = partFitCurrCols + 1
    partInertiaCols = partFitLbestCols + 1
    partLocalBestCols = slice(partInertiaCols, partInertiaCols + nDim)
    partFlagFitEvalCols = partLocalBestCols.stop
    partFitEvalsCols = partFlagFitEvalCols + 1

    nColsPop = partFitEvalsCols + 1
    pop = cp.zeros((psoParams['popsize'], nColsPop))

    # 初始化种群
    pop[:, partCoordCols] = cp.random.rand(psoParams['popsize'], nDim)
    pop[:, partVelCols] = -pop[:, partCoordCols] + cp.random.rand(psoParams['popsize'], nDim)
    pop[:, partPbestCols] = pop[:, partCoordCols]
    pop[:, partFitPbestCols] = cp.inf
    pop[:, partFitCurrCols] = 0
    pop[:, partFitLbestCols] = cp.inf
    pop[:, partLocalBestCols] = 0
    pop[:, partFlagFitEvalCols] = 1
    pop[:, partInertiaCols] = 0
    pop[:, partFitEvalsCols] = 0

    gbestVal = cp.inf
    gbestLoc = 2 * cp.ones((1, nDim))

    for lpc_steps in tqdm(range(psoParams['maxSteps'])):
        if psoParams['bndryCond']:
            fitnessValues, pop[:, partCoordCols] = fitfuncHandle(pop[:, partCoordCols], returnxVec=1)
        else:
            fitnessValues = fitfuncHandle(pop[:, partCoordCols], returnxVec=0)
        # 在使用 fitnessValues 之前，确保它是 CuPy 数组
        if isinstance(fitnessValues, np.ndarray):
            fitnessValues = cp.asarray(fitnessValues)

        pop[:, partFitCurrCols] = fitnessValues
        update_mask = pop[:, partFitPbestCols] > pop[:, partFitCurrCols]
        pop[update_mask, partFitPbestCols] = pop[update_mask, partFitCurrCols]
        pop[update_mask, partPbestCols] = pop[update_mask, partCoordCols]

        bestFitness = cp.min(pop[:, partFitCurrCols])
        if gbestVal > bestFitness:
            gbestVal = bestFitness
            bestParticle = cp.argmin(pop[:, partFitCurrCols])
            gbestLoc = pop[bestParticle, partCoordCols]

        # 局部最优更新 (使用GPU矩阵运算优化)
        ringIndices = cp.arange(psoParams['popsize']).reshape(-1, 1) + cp.arange(psoParams['nbrhdSz'])
        ringIndices %= psoParams['popsize']
        minIndices = cp.argmin(pop[ringIndices, partFitCurrCols], axis=1)
        lbestLoc = pop[ringIndices[cp.arange(psoParams['popsize']), minIndices], partCoordCols]

        pop[:, partLocalBestCols] = lbestLoc
        pop[:, partFitLbestCols] = pop[ringIndices[cp.arange(psoParams['popsize']), minIndices], partFitCurrCols]

        # 速度更新
        inertiaWt = cp.maximum(psoParams['dcLaw_a'] - (psoParams['dcLaw_b'] / psoParams['dcLaw_c']) * lpc_steps,
                               psoParams['dcLaw_d'])
        chi1 = cp.random.rand(psoParams['popsize'], nDim)
        chi2 = cp.random.rand(psoParams['popsize'], nDim)

        pop[:, partVelCols] = (inertiaWt * pop[:, partVelCols] +
                               psoParams['c1'] * (pop[:, partPbestCols] - pop[:, partCoordCols]) * chi1 +
                               psoParams['c2'] * (pop[:, partLocalBestCols] - pop[:, partCoordCols]) * chi2)
        pop[:, partVelCols] = cp.clip(pop[:, partVelCols], -psoParams['max_velocity'], psoParams['max_velocity'])
        pop[:, partCoordCols] += pop[:, partVelCols]

        invalid = cp.any((pop[:, partCoordCols] < 0) | (pop[:, partCoordCols] > 1), axis=1)
        pop[invalid, partFitCurrCols] = cp.inf
        pop[invalid, partFlagFitEvalCols] = 0

    returnData.update({
        'totalFuncEvals': int(cp.sum(pop[:, partFitEvalsCols]).get()),
        'bestLocation': cp.asnumpy(gbestLoc),
        'bestFitness': float(gbestVal.get())
    })
    return returnData


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, mlz, y):
    r = (10 ** r) * 1e6 * pc
    m_c = (10 ** m_c) * M_sun
    mlz = (10 ** mlz) * M_sun

    # GPU向量化运算
    theta_t = c ** 3 * (tc - dataX) / (5 * G * m_c)
    theta_t = cp.where(dataX < tc, theta_t, 0)
    h = G * m_c / (c ** 2 * r) * theta_t ** (-0.25) * cp.cos(2 * phi_c - 2 * theta_t ** (0.625))
    h = cp.where(dataX >= tc, 0, h)

    h_f = cp.fft.rfft(h)
    freqs = cp.fft.rfftfreq(len(dataX), d=cp.diff(dataX)[0])
    omega = 2 * cp.pi * freqs
    w = G * 4 * mlz * omega / c ** 3

    F_geo = cp.sqrt(1 + 1 / y) - 1j * cp.sqrt(cp.abs(-1 + 1 / y)) * cp.exp(1j * w * 2 * y)
    F_geo = cp.where(y >= 1, cp.sqrt(1 + 1 / y), F_geo)

    sigVec_f = h_f * F_geo
    return cp.fft.irfft(sigVec_f)


def glrtqcsig4pso(xVec, params, returnxVec=0):
    # 确保 xVec 是 CuPy 数组
    if isinstance(xVec, np.ndarray):
        xVec = cp.asarray(xVec)
    validPts = crcbchkstdsrchrng(xVec)  # 返回 CuPy 数组
    # 显式保留在 GPU
    xVec_valid = xVec[validPts]
    xVec[validPts] = s2rv(xVec_valid, params)  # 直接传递 CuPy 数组

    fitVal = cp.full(xVec.shape[0], cp.inf)
    validIndices = cp.where(validPts)[0]

    for idx in validIndices.get().tolist():  # 分批处理避免内存不足
        x = xVec[idx]
        fitVal[idx] = ssrqc(x, params)

    return (fitVal.get(), xVec.get()) if returnxVec else fitVal.get()


def ssrqc(x, params):
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5])
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdPosFreq'], 1)
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdPosFreq'])
    return -cp.abs(inPrd) ** 2


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    nSamples = len(sigVec)
    kNyq = nSamples // 2 + 1
    psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    normSigSqrd = cp.sum((cp.fft.fft(sigVec) / psdVec4Norm) * cp.conj(cp.fft.fft(sigVec))) / (sampFreq * nSamples)
    normFac = snr / cp.sqrt(normSigSqrd)
    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)
    psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    return cp.real(cp.sum((fftX / psdVec4Norm) * cp.conj(fftY)) / (sampFreq * len(xVec)))


def s2rv(xVec, params):
    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    # 强制输入为 CuPy 数组
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)
    # 返回 CuPy 布尔数组
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)