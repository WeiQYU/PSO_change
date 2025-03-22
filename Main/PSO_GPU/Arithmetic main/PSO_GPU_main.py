import cupy as cp
from cupyx.scipy.fftpack import fft
from tqdm import tqdm
import numpy as np
import pycbc.types
from pycbc.filter import match

cp.random.seed(5511)
# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
c = 2.998e8      # Speed of light, m/s
M_sun = 1.989e30 # Solar mass, kg
pc = 3.086e16    # Parsec to meters

__all__ = [
    'crcbqcpsopsd',
    'crcbpso',
    'crcbgenqcsig',
    'glrtqcsig4pso',
    'ssrqc',
    'normsig4psd',
    'innerprodpsd',
    's2rv',
    'crcbchkstdsrchrng'
]

def crcbqcpsopsd(inParams, psoParams, nRuns):
    # Transfer data to GPU
    inParams['dataX'] = cp.asarray(inParams['dataX'])
    inParams['dataY'] = cp.asarray(inParams['dataY'])
    # inParams['psdPosFreq'] = cp.asarray(inParams['psdPosFreq'])
    inParams['psdHigh'] = cp.asarray(inParams['psdHigh'])
    inParams['rmax'] = cp.asarray(inParams['rmax'])
    inParams['rmin'] = cp.asarray(inParams['rmin'])

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
        'A': None,
        'delta_t': None,
    }

    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1
        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)

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
        _, params = fHandle(outStruct[lpruns]['bestLocation'][cp.newaxis, ...], returnxVec=1)
        params = cp.asnumpy(params[0])
        r, m_c, tc, phi_c, A, delta_t = params

        estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t)
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdHigh'])
        estSig = estAmp * estSig
        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns].get()),
            'r': r,
            'm_c': m_c,
            'tc': tc,
            'phi_c': phi_c,
            'A': A,
            'delta_t': delta_t,
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
        'A': outResults['allRunsOutput'][bestRun]['A'],
        'delta_t': outResults['allRunsOutput'][bestRun]['delta_t'],
    })
    return outResults, outStruct

def crcbpso(fitfuncHandle, nDim, **kwargs):
    psoParams = {
        'popsize': 220,
        'maxSteps': 2000,
        'c1': 1.729606984651714,
        'c2': 1.3457415961968024,
        'max_velocity': 0.5445020314499609,
        'dcLaw_a': 0.9,
        'dcLaw_b': 0.4,
        'dcLaw_c': 2000 - 1,
        'dcLaw_d': 0.2,
        'bndryCond': '',
        'nbrhdSz': 4
    }
    psoParams.update(kwargs)
    run = psoParams.get('run', 1)

    returnData = {
        'totalFuncEvals': 0,
        'bestLocation': cp.zeros((1, nDim)),
        'bestFitness': cp.inf,
    }

    partCoordCols = slice(0, nDim)
    partVelCols = slice(nDim, 2 * nDim)
    partPbestCols = slice(2 * nDim, 3 * nDim)
    partFitPbestCols = 3 * nDim
    partFitCurrCols = partFitPbestCols + 1
    partFitLbestCols = partFitCurrCols + 1
    partInertiaCols = partFitLbestCols + 1
    partLocalBestCols = slice(partInertiaCols + 1, partInertiaCols + 1 + nDim)
    partFlagFitEvalCols = partLocalBestCols.stop
    partFitEvalsCols = partFlagFitEvalCols + 1

    nColsPop = partFitEvalsCols + 1
    pop = cp.zeros((psoParams['popsize'], nColsPop))

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
    gbestLoc = cp.ones((1, nDim))
    total_evals = 0

    with tqdm(range(psoParams['maxSteps']), desc=f'Run {run}', position=0, leave=True) as pbar:
        for lpc_steps in pbar:
            if psoParams['bndryCond']:
                fitnessValues, pop[:, partCoordCols] = fitfuncHandle(pop[:, partCoordCols], returnxVec=1)
            else:
                fitnessValues = fitfuncHandle(pop[:, partCoordCols], returnxVec=0)
            if isinstance(fitnessValues, np.ndarray):
                fitnessValues = cp.asarray(fitnessValues)
            total_evals += psoParams['popsize']

            pop[:, partFitCurrCols] = fitnessValues
            update_mask = pop[:, partFitPbestCols] > pop[:, partFitCurrCols]
            pop[update_mask, partFitPbestCols] = pop[update_mask, partFitCurrCols]
            pop[update_mask, partPbestCols] = pop[update_mask, partCoordCols]

            bestFitness = cp.min(pop[:, partFitCurrCols])
            if gbestVal > bestFitness:
                gbestVal = bestFitness
                bestParticle = cp.argmin(pop[:, partFitCurrCols])
                gbestLoc = pop[bestParticle, partCoordCols]

            ringIndices = cp.arange(psoParams['popsize']).reshape(-1, 1) + cp.arange(psoParams['nbrhdSz'])
            ringIndices %= psoParams['popsize']
            minIndices = cp.argmin(pop[ringIndices, partFitCurrCols], axis=1)
            lbestLoc = pop[ringIndices[cp.arange(psoParams['popsize']), minIndices], partCoordCols]

            pop[:, partLocalBestCols] = lbestLoc
            pop[:, partFitLbestCols] = pop[ringIndices[cp.arange(psoParams['popsize']), minIndices], partFitCurrCols]

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
        'totalFuncEvals': total_evals,
        'bestLocation': cp.asnumpy(gbestLoc),
        'bestFitness': float(gbestVal.get())
    })
    return returnData


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t):
    r = (10 ** r) * 1e6 * pc
    m_c = (10 ** m_c) * M_sun
    delta_t = (10 ** delta_t)

    def generate_h_t(t_array, M_c, r, phi_c, t_c):
        # Vectorized computation using CuPy
        mask = t_array < t_c
        theta_t = cp.zeros_like(t_array)
        theta_t[mask] = c ** 3 * (t_c - t_array[mask]) / (5 * G * M_c)
        h = cp.zeros_like(t_array)
        h[mask] = G * M_c / (c ** 2 * r) * theta_t[mask] ** (-1 / 4) * cp.cos(2 * phi_c - 2 * theta_t[mask] ** (5 / 8))
        return h

    # Convert input array to CuPy
    dataX_gpu = cp.asarray(dataX)

    # Generate signal
    h = generate_h_t(dataX_gpu, m_c, r, phi_c, tc)

    # Convert to frequency domain
    h_f = cp.fft.rfft(h)
    freqs = cp.fft.rfftfreq(len(h), dataX_gpu[1] - dataX_gpu[0])

    # Apply low frequency cutoff
    low_freq_cutoff = 3.0
    freq_mask = freqs >= low_freq_cutoff
    h_f = cp.where(freq_mask, h_f, 0)

    def parametric_lens_model(h_f, freqs, A, dt):
        phi = 2 * cp.pi * freqs * dt
        F_f = 1 + A * cp.exp(1j * phi)
        return h_f * F_f

    # Apply lensing effect
    h_lens_f = parametric_lens_model(h_f, freqs, A, delta_t)

    # Convert back to time domain
    return cp.fft.irfft(h_lens_f)

def glrtqcsig4pso(xVec, params, returnxVec=0):
    if isinstance(xVec, np.ndarray):
        xVec = cp.asarray(xVec)
    validPts = crcbchkstdsrchrng(xVec)
    xVec_valid = xVec[validPts]
    xVec[validPts] = s2rv(xVec_valid, params)

    fitVal = cp.full(xVec.shape[0], cp.inf)
    validIndices = cp.where(validPts)[0]

    for idx in validIndices.get().tolist():
        x = xVec[idx]
        fitVal[idx] = ssrqc(x, params)

    return (fitVal.get(), xVec.get()) if returnxVec else fitVal.get()

def ssrqc(x, params):
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5])
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdHigh'])
    return -cp.abs(inPrd) ** 2

def normsig4psd(sigVec, sampFreq, psdVec, snr):
    nSamples = len(sigVec)
    psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    normSigSqrd = cp.sum((cp.fft.fft(sigVec) / psdVec4Norm) * cp.conj(cp.fft.fft(sigVec))) / (sampFreq * nSamples)
    normFac = snr / cp.sqrt(normSigSqrd)
    return normFac * sigVec, normFac

def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)
    psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    return cp.real(cp.sum((fftX / psdVec4Norm) * cp.conj(fftY)) / (sampFreq * len(xVec)))

# def s2rv(xVec, params):
#     rmax = cp.asarray(params['rmax'])
#     rmin = cp.asarray(params['rmin'])
#     return xVec * (rmax - rmin) + rmin
def s2rv(xVec, params):
    """综合改进版参数映射函数"""
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)

    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])

    # 基本映射
    transformed = xVec * (rmax - rmin) + rmin

    # 1. 参数耦合 - r和m_c关联
    r_idx, mc_idx, phi_idx = 0, 1, 3
    r_val = transformed[:, r_idx]
    r_norm = (r_val - rmin[r_idx]) / (rmax[r_idx] - rmin[r_idx])
    mc_adjustment = 0.1 * r_norm
    transformed[:, mc_idx] = transformed[:, mc_idx] * (1 - mc_adjustment)

    # 2. 物理约束 - 极端值检查与调整
    mc_val = transformed[:, mc_idx]
    amplitude_factor = cp.power(10 ** mc_val, 5 / 3) / (10 ** r_val)
    target_amplitude = (13.4881 ** (5 / 3)) / 100.0
    ratio = amplitude_factor / target_amplitude

    # 对极端异常值进行修正
    extreme_mask = (ratio > 1000) | (ratio < 0.001)
    if cp.any(extreme_mask):
        # 调整方向取决于比值是过大还是过小
        high_ratio_mask = (ratio > 1000) & extreme_mask
        low_ratio_mask = (ratio < 0.001) & extreme_mask

        # 因子过大时减小m_c值
        if cp.any(high_ratio_mask):
            transformed[high_ratio_mask, mc_idx] *= 0.95

        # 因子过小时增加m_c值
        if cp.any(low_ratio_mask):
            transformed[low_ratio_mask, mc_idx] *= 1.05

    # 3. 相位处理改进
    # 规范化相位到[0, 2π]范围
    transformed[:, phi_idx] = transformed[:, phi_idx] % (2 * cp.pi)

    # 相位参数轻微偏向目标值
    target_phi = cp.pi  # 目标相位值为π
    phi_values = transformed[:, phi_idx]

    # 计算周期性距离
    dist_to_target = cp.minimum(
        cp.abs(phi_values - target_phi),
        cp.minimum(
            cp.abs(phi_values - target_phi - 2 * cp.pi),
            cp.abs(phi_values - target_phi + 2 * cp.pi)
        )
    )

    # 仅对相位偏差很大的点进行轻微调整
    far_mask = dist_to_target > cp.pi / 2
    if cp.any(far_mask):
        adjustment = 0.05 * (target_phi - phi_values[far_mask])
        transformed[far_mask, phi_idx] += adjustment
        transformed[:, phi_idx] = transformed[:, phi_idx] % (2 * cp.pi)

    return transformed

def crcbchkstdsrchrng(xVec):
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)



