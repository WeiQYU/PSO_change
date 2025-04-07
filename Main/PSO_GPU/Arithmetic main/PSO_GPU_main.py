import cupy as cp
from cupyx.scipy.fftpack import fft
from tqdm import tqdm
import numpy as np
import pycbc.types
from pycbc.filter import match
import scipy.constants as const

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c     # Speed of light, m/s
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

        # Fix for dimension handling
        bestLocation = cp.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)  # Ensure 2D shape (1, nDim)

        _, params = fHandle(bestLocation, returnxVec=1)

        # Handle params dimensionality
        if isinstance(params, list) and len(params) > 0:
            params = params[0]
        elif params.ndim > 1 and params.shape[0] == 1:
            params = params[0]

        # Convert to numpy if needed
        if isinstance(params, cp.ndarray):
            params = cp.asnumpy(params)

        r, m_c, tc, phi_c, A, delta_t = params

        estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t)
        estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
        estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdHigh'])
        estSig = estAmp * estSig
        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns].get()) if hasattr(fitVal[lpruns], 'get') else float(fitVal[lpruns]),
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

    # Convert to numpy for comparison if needed
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
    psoParams = {
        'popsize': 200,
        'maxSteps': 2000,
        'c1': 2,
        'c2': 2,
        'max_velocity': 0.5,
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

    # Fix for the error: Handle both CuPy array and float types
    best_fitness = float(gbestVal) if isinstance(gbestVal, (int, float)) else float(gbestVal.get())

    returnData.update({
        'totalFuncEvals': total_evals,
        'bestLocation': cp.asnumpy(gbestLoc),
        'bestFitness': best_fitness
    })
    return returnData


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t):
    r = (10 ** r) * 1e6 * pc
    m_c = (10 ** m_c) * M_sun
    # delta_t = 10 ** delta_t

    def generate_gw_signal(t):
        # 确保在合并前截止
        valid_idx = t < tc
        t_valid = t[valid_idx]
        # 计算Θ(t)，控制信号的频率演化
        Theta = c ** 3 * (tc - t_valid) / (5 * G * m_c)
        # 计算振幅部分
        A = (G * m_c / (c ** 2 * r)) * Theta ** (-1 / 4)

        # 原始相位计算
        phase = 2 * phi_c - 2 * Theta ** (5 / 8)

        # 计算波形
        h = np.zeros_like(t)
        h[valid_idx] = A * np.cos(phase)
        return h

    # Convert input array to CuPy
    dataX_gpu = cp.asarray(dataX)

    # Generate signal
    h = generate_gw_signal(dataX_gpu)

    # Convert to frequency domain
    h_f = cp.fft.rfft(h)
    freqs = cp.fft.rfftfreq(len(h), dataX_gpu[1] - dataX_gpu[0])

    def parametric_lens_model(h_f, freqs, A, dt):
        phi = 2 * cp.pi * freqs * dt
        F_f = 1 + A * cp.exp(1j * phi)
        return h_f * F_f

    # Apply lensing effect
    h_lens_f = parametric_lens_model(h_f, freqs, A, delta_t)
    # print(f'r:{r / 1e6 / pc},m_c:{m_c / M_sun},tc:{tc},phi_c:{phi_c},A:{A},delta_t:{delta_t}') # 参数的估计正确
    # print(f'signal:{cp.fft.irfft(h_lens_f)}')  # 信号正确
    # Convert back to time domain
    return cp.fft.irfft(h_lens_f)


def glrtqcsig4pso(xVec, params, returnxVec=0):
    if isinstance(xVec, np.ndarray):
        xVec = cp.asarray(xVec)

    # Fix for the IndexError
    # The error happens because validPts shape doesn't match xVec properly
    # We need to handle the dimensionality correctly

    # Check dimensions and reshape if needed
    if xVec.ndim == 2:
        # When processing a batch of vectors
        validPts = crcbchkstdsrchrng(xVec)
        fitVal = cp.full(xVec.shape[0], cp.inf)

        # Process each valid point individually
        for i in range(xVec.shape[0]):
            if validPts[i]:
                # Apply s2rv to valid points
                xVec[i] = s2rv(xVec[i:i + 1], params)[0]
                # Calculate fitness
                fitVal[i] = ssrqc(xVec[i], params)
    else:
        # Single vector case (should not normally happen but added for completeness)
        validPts = crcbchkstdsrchrng(xVec.reshape(1, -1))[0]
        fitVal = cp.inf

        if validPts:
            xVec = s2rv(xVec.reshape(1, -1), params)[0]
            fitVal = ssrqc(xVec, params)

    return (fitVal.get(), xVec.get()) if returnxVec else fitVal.get()

def ssrqc(x, params):
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5])
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdHigh'])
    return -cp.abs(inPrd) ** 2

def normsig4psd(sigVec, sampFreq, psdVec, snr):
    nSamples = len(sigVec)
    # print(f'psdVec:{psdVec}')
    psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    normSigSqrd = cp.sum((cp.fft.fft(sigVec) / psdVec4Norm) * cp.conj(cp.fft.fft(sigVec))) / (sampFreq * nSamples)
    normFac = snr / cp.sqrt(normSigSqrd)
    return normFac * sigVec, normFac

def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)
    psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    # print(cp.isnan(fftX),cp.isnan(fftY),cp.isnan(psdVec4Norm),cp.isinf(fftX),cp.isinf(fftY),cp.isinf(psdVec4Norm))
    return cp.real(cp.sum((fftX / psdVec4Norm) * cp.conj(fftY)) / (sampFreq * len(xVec)))

def s2rv(xVec, params):
    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)

    # Ensure we're checking if all elements in each row are valid
    # This will return a 1D array with one boolean per row in xVec
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)