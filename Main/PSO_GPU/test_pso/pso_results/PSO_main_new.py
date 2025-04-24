import cupy as cp
from cupyx.scipy.fftpack import fft
from pycbc.types import FrequencySeries, TimeSeries
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
    'generate_unlensed_gw',
    'apply_lensing_effect',
    'crcbgenqcsig',
    'glrtqcsig4pso',
    'ssrqc',
    'normsig4psd',
    'innerprodpsd',
    's2rv',
    'crcbchkstdsrchrng',
    'calculate_snr_pycbc',
    'analyze_mismatch',
    'pycbc_calculate_match',
    'two_step_matching'
]


def generate_unlensed_gw(dataX, r, m_c, tc, phi_c):
    # Convert parameter units
    r = (10 ** r) * 1e6 * pc  # Distance (meters)
    m_c = (10 ** m_c) * M_sun  # Combined mass (kg)

    # Ensure input is CuPy array
    if not isinstance(dataX, cp.ndarray):
        dataX_gpu = cp.asarray(dataX)
    else:
        dataX_gpu = dataX

    # Generate gravitational wave signal
    t = dataX_gpu  # Time series

    # Calculate signal in valid region before merger
    valid_idx = t < tc
    t_valid = t[valid_idx]

    # Initialize waveform
    h = cp.zeros_like(t)

    if cp.sum(valid_idx) > 0:  # Ensure there's a valid region
        # Calculate frequency evolution parameter Theta
        Theta = c ** 3 * (tc - t_valid) / (5 * G * m_c)

        # Calculate amplitude
        A_gw = (G * m_c / (c ** 2 * r)) * Theta ** (-1 / 4)

        # Calculate phase
        phase = 2 * phi_c - 2 * Theta ** (5 / 8)

        # Generate waveform
        h[valid_idx] = A_gw * cp.cos(phase)

    return h


def apply_lensing_effect(h, t, A, delta_t, tc):
    # Calculate FFT of signal
    n = len(h)
    h_fft = cp.fft.fft(h)

    # Calculate frequency array
    dt = t[1] - t[0]  # Sampling interval
    fs = 1 / dt  # Sampling frequency
    freqs = cp.fft.fftfreq(n, dt)

    # Calculate lens transfer function F(f) = 1 + A * exp(i * Phi)
    # where Phi = 2πf * delta_t
    Phi = 2 * cp.pi * freqs * delta_t
    lens_transfer = 1 + A * cp.exp(1j * Phi)

    # Apply lensing effect in frequency domain
    h_lensed_fft = h_fft * lens_transfer

    # Convert back to time domain
    h_lens = cp.real(cp.fft.ifft(h_lensed_fft))

    # Ensure signal is zero after merger time
    h_lens[t > tc] = 0

    return h_lens


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True):
    # Generate unlensed waveform
    h = generate_unlensed_gw(dataX, r, m_c, tc, phi_c)

    # Apply lensing effect if needed
    if use_lensing:
        # Ensure input is CuPy array
        if not isinstance(dataX, cp.ndarray):
            t = cp.asarray(dataX)
        else:
            t = dataX

        h = apply_lensing_effect(h, t, A, delta_t, tc)

    return h


def pycbc_calculate_match(signal1, signal2, fs, psd):
    # Ensure data is NumPy arrays
    if isinstance(signal1, cp.ndarray):
        signal1 = cp.asnumpy(signal1)
    if isinstance(signal2, cp.ndarray):
        signal2 = cp.asnumpy(signal2)
    if isinstance(psd, cp.ndarray):
        psd = cp.asnumpy(psd)

    # Create PyCBC TimeSeries objects
    delta_t = 1.0 / fs
    ts_signal1 = TimeSeries(signal1, delta_t=delta_t)
    ts_signal2 = TimeSeries(signal2, delta_t=delta_t)

    # Create PyCBC FrequencySeries object for PSD
    delta_f = 1.0 / (len(signal1) * delta_t)
    psd_series = FrequencySeries(psd, delta_f=delta_f)

    # Calculate match using pycbc.filter.match
    match_value, _ = match(ts_signal1, ts_signal2, psd=psd_series, low_frequency_cutoff=10.0)

    return float(match_value)


def two_step_matching(params, dataY, psdHigh, sampFreq):
    # Extract parameters
    r = params.get('r')
    m_c = params.get('m_c')
    tc = params.get('tc')
    phi_c = params.get('phi_c')
    A = params.get('A')
    delta_t = params.get('delta_t')
    dataX = params.get('dataX')
    dataY_only_signal = params.get('dataY_only_signal')  # Add this to accept signal-only data

    # Initialize result
    result = {
        'unlensed_signal': None,
        'unlensed_snr': None,
        'unlensed_mismatch': None,
        'lensed_signal': None,
        'lensed_snr': None,
        'lensed_mismatch': None,
        'threshold': None,
        'is_lensed': False,
        'message': "",
        'classification': "noise"  # Default classification
    }

    # Generate unlensed signal
    unlensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=False)

    # Normalize signal
    unlensed_signal, _ = normsig4psd(unlensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY, unlensed_signal, sampFreq, psdHigh)
    unlensed_signal = estAmp * unlensed_signal

    # Calculate SNR
    unlensed_snr = calculate_snr_pycbc(unlensed_signal, psdHigh, sampFreq)

    # Update result with unlensed_signal info
    result.update({
        'unlensed_signal': unlensed_signal,
        'unlensed_snr': unlensed_snr
    })

    # Check if SNR < 8 (noise) - highest priority
    print("检测是不是噪声")
    if unlensed_snr < 8:
        print("ok,是噪声")
        result['message'] = "This is noise"
        result['classification'] = "noise"
        return result

    # Calculate mismatch using pycbc.filter.match
    unlensed_mismatch = 1 - pycbc_calculate_match(unlensed_signal, dataY_only_signal, sampFreq, psdHigh)
    result['unlensed_mismatch'] = unlensed_mismatch

    # Calculate threshold
    threshold = 1.0 / unlensed_snr
    result['threshold'] = threshold

    # If unlensed template matches well, it's an unlensed signal - second priority
    if unlensed_mismatch >= threshold:
        print('通过了第一次检测,有信号')
        result['message'] = "This is a signal"
        result['classification'] = "signal"
        return result

    # 进一步检查是否为透镜化模型
    # Generate lensed signal
    lensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)

    # Normalize signal
    # lensed_signal, _ = normsig4psd(lensed_signal, sampFreq, psdHigh, 1)
    lensed_signal, _ = normsig4psd(lensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY, lensed_signal, sampFreq, psdHigh)
    lensed_signal = estAmp * lensed_signal

    # Calculate SNR
    lensed_snr = calculate_snr_pycbc(lensed_signal, psdHigh, sampFreq)

    # Calculate mismatch using pycbc.filter.match
    lensed_mismatch = 1 - pycbc_calculate_match(lensed_signal, dataY_only_signal, sampFreq, psdHigh)

    # Update result
    result.update({
        'lensed_signal': lensed_signal,
        'lensed_snr': lensed_snr,
        'lensed_mismatch': lensed_mismatch
    })

    # Double check for lensed signal: mismatch must be less than threshold
    print('进行第二次检测')
    if lensed_mismatch <= threshold:
        print("通过了透镜检测")
        result['is_lensed'] = True
        result['message'] = "This is a lens signal"
        result['classification'] = "lens_signal"
    else:
        # Neither unlensed nor lensed matched well - still classify as signal but note it's not a good match
        print("没通过透镜检测")
        result['message'] = "This is a signal (signal_mismatch > threshold and len_mismatch < threshold)"
        result['classification'] = "signal"

    return result


def crcbqcpsopsd(inParams, psoParams, nRuns, use_two_step=True):
    """
    Particle Swarm Optimization main function for multiple runs
    """
    # Transfer data to GPU
    inParams['dataX'] = cp.asarray(inParams['dataX'])
    inParams['dataY'] = cp.asarray(inParams['dataY'])
    inParams['psdHigh'] = cp.asarray(inParams['psdHigh'])
    inParams['rmax'] = cp.asarray(inParams['rmax'])
    inParams['rmin'] = cp.asarray(inParams['rmin'])

    # Add signal-only data if provided
    if 'dataY_only_signal' in inParams:
        inParams['dataY_only_signal'] = cp.asarray(inParams['dataY_only_signal'])
    else:
        inParams['dataY_only_signal'] = inParams['dataY']  # Use full data if signal-only not provided

    nSamples = len(inParams['dataX'])
    nDim = 6
    # Create fitness function handle
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
        'is_lensed': False,
        'lensing_message': "",
        'classification': None  # Add classification field
    }

    # Run PSO multiple times with different random seeds
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1
        # Set different random seeds to ensure different results in multiple runs
        cp.random.seed(int(time.time()) + lpruns * 1000)
        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)
        # Print best fitness for each run
        print(f"Run {lpruns + 1} completed with best fitness: {outStruct[lpruns]['bestFitness']}")

    # Process results from all runs
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
            'is_lensed': False,
            'lensing_message': "",
            'classification': "noise"  # Default classification
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']

        # Ensure dimensions are handled correctly
        bestLocation = cp.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)  # Ensure 2D shape (1, nDim)

        # Get parameters from best location
        _, params = fHandle(bestLocation, returnxVec=1)

        # Handle parameter dimensions
        if isinstance(params, list) and len(params) > 0:
            params = params[0]
        elif isinstance(params, cp.ndarray) and params.ndim > 1 and params.shape[0] == 1:
            params = params[0]

        # Convert to numpy if needed
        if isinstance(params, cp.ndarray):
            params = cp.asnumpy(params)

        r, m_c, tc, phi_c, A, delta_t = params

        # Add two-step matching process if requested
        if use_two_step:
            # Prepare parameter dictionary
            param_dict = {
                'r': r,
                'm_c': m_c,
                'tc': tc,
                'phi_c': phi_c,
                'A': A,
                'delta_t': delta_t,
                'dataX': inParams['dataX'],
                'dataY_only_signal': inParams['dataY_only_signal']  # Pass signal-only data
            }

            # Execute two-step matching
            matching_result = two_step_matching(
                param_dict,
                inParams['dataY'],
                inParams['psdHigh'],
                inParams['sampFreq']
            )

            # Use matching results
            is_lensed = matching_result['is_lensed']
            lensing_message = matching_result['message']
            classification = matching_result['classification']

            # Decide which signal to use based on classification
            if classification == "noise":
                # Use unlensed signal for noise (doesn't matter much since SNR is low)
                print("是噪声")
                estSig = matching_result['unlensed_signal']
            elif classification == "signal":
                # Use unlensed signal for unlensed signal
                print("第一判断是未透镜")
                estSig = matching_result['unlensed_signal']
            elif classification == "lens_signal":
                # Use lensed signal for lensed signal
                print("是透镜")
                estSig = matching_result['lensed_signal']
            else:
                # Fallback to unlensed signal
                print("二次判断是未透镜")
                estSig = matching_result['unlensed_signal']
        else:
            # Use original method to generate signal
            print('被迫无奈了？')
            estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t)
            estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
            estAmp = innerprodpsd(inParams['dataY'], estSig, inParams['sampFreq'], inParams['psdHigh'])
            estSig = estAmp * estSig

            # Default values
            is_lensed = False
            lensing_message = "Two-step matching not performed"
            classification = "unknown"

        # Update output
        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns].get()) if hasattr(fitVal[lpruns], 'get') else float(fitVal[lpruns]),
            'r': r,
            'm_c': m_c,
            'tc': tc,
            'phi_c': phi_c,
            'A': A,
            'delta_t': delta_t,
            'estSig': cp.asarray(estSig),
            'totalFuncEvals': outStruct[lpruns]['totalFuncEvals'],
            'is_lensed': is_lensed,
            'lensing_message': lensing_message,
            'classification': classification
        })
        outResults['allRunsOutput'].append(allRunsOutput)

    # Find best run
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
        'is_lensed': outResults['allRunsOutput'][bestRun]['is_lensed'],
        'lensing_message': outResults['allRunsOutput'][bestRun]['lensing_message'],
        'classification': outResults['allRunsOutput'][bestRun]['classification']
    })
    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """
    Completely rewritten PSO core algorithm implementation with early stopping
    """
    # Default PSO parameters
    psoParams = {
        'popsize': 200,
        'maxSteps': 2000,
        'c1': 2.0,  # Individual learning factor
        'c2': 2.0,  # Social learning factor
        'max_velocity': 0.5,  # Maximum velocity limit
        'w_start': 0.9,  # Initial inertia weight
        'w_end': 0.4,  # Final inertia weight
        'run': 1,  # Run number
        'nbrhdSz': 4  # Neighborhood size
    }
    # Update parameters
    psoParams.update(kwargs)

    # Ensure random number reproducibility
    if 'seed' in psoParams:
        cp.random.seed(psoParams['seed'])

    # Initialize return data structure
    returnData = {
        'totalFuncEvals': 0,
        'bestLocation': cp.zeros((1, nDim)),
        'bestFitness': cp.inf,
        'fitnessHistory': []  # Record fitness history
    }

    # Initialize particle swarm
    particles = cp.random.rand(psoParams['popsize'], nDim)  # Position: uniform distribution in [0,1]
    velocities = cp.random.uniform(-0.1, 0.1, (psoParams['popsize'], nDim))  # Velocity: small random values

    # Evaluate initial fitness
    fitness = cp.zeros(psoParams['popsize'])
    for i in range(psoParams['popsize']):
        fitness[i] = fitfuncHandle(particles[i:i + 1], returnxVec=0)

    # Initialize personal best and global best
    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    # Find global best
    gbest_idx = cp.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx].copy()

    # Record initial fitness
    returnData['fitnessHistory'].append(float(gbest_fitness))

    total_evals = psoParams['popsize']  # Counter: number of fitness evaluations

    # Initialize early stopping variables
    no_improvement_count = 0
    prev_best_fitness = float(gbest_fitness)
    max_no_improvement = psoParams['maxSteps'] // 2  # Stop if no improvement for half the total iterations

    # Create progress bar
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # Update inertia weight - linear decrease
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

            # Update each particle
            for i in range(psoParams['popsize']):
                # Get local best (ring topology)
                neighbors = []
                for j in range(psoParams['nbrhdSz']):
                    idx = (i + j) % psoParams['popsize']
                    neighbors.append(idx)

                # Use numpy's argmin instead of cupy's argmin because neighbors is a Python list
                neighbor_fitness = [float(pbest_fitness[n]) for n in neighbors]
                best_neighbor_idx = np.argmin(neighbor_fitness)
                lbest_idx = neighbors[best_neighbor_idx]
                lbest = pbest[lbest_idx].copy()

                # Generate random coefficients
                r1 = cp.random.rand(nDim)
                r2 = cp.random.rand(nDim)

                # Update velocity
                velocities[i] = (w * velocities[i] +
                                 psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                 psoParams['c2'] * r2 * (lbest - particles[i]))

                # Limit velocity
                velocities[i] = cp.clip(velocities[i], -psoParams['max_velocity'], psoParams['max_velocity'])

                # Update position
                particles[i] += velocities[i]

                # Handle boundary constraints - reflective boundary
                # If position is out of bounds, reflect back and reverse velocity direction
                out_low = particles[i] < 0
                out_high = particles[i] > 1

                particles[i] = cp.where(out_low, -particles[i], particles[i])
                particles[i] = cp.where(out_high, 2 - particles[i], particles[i])

                # Ensure position is in [0,1] range (prevent numerical errors)
                particles[i] = cp.clip(particles[i], 0, 1)

                # Reverse velocity at boundaries
                velocities[i] = cp.where(out_low | out_high, -velocities[i], velocities[i])

                # Evaluate new position
                new_fitness = fitfuncHandle(particles[i:i + 1], returnxVec=0)
                fitness[i] = new_fitness
                total_evals += 1

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = new_fitness

            # Update global best
            current_best_idx = cp.argmin(pbest_fitness)
            if pbest_fitness[current_best_idx] < gbest_fitness:
                gbest = pbest[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx].copy()

                # Update progress bar information
                pbar.set_postfix({'fitness': float(gbest_fitness)})

            # Record best fitness at each step
            returnData['fitnessHistory'].append(float(gbest_fitness))

            # Check early stopping condition
            current_best_fitness = float(gbest_fitness)
            if abs(current_best_fitness - prev_best_fitness) < 1e-10:  # Consider floating point errors
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                prev_best_fitness = current_best_fitness

            # Stop early if no improvement for half the iterations
            if no_improvement_count >= max_no_improvement:
                print(
                    f"Run {psoParams['run']} stopped early after {step + 1} iterations: No improvement for {max_no_improvement} iterations")
                break

    # Update return data when finished
    returnData.update({
        'totalFuncEvals': total_evals,
        'bestLocation': cp.asnumpy(gbest.reshape(1, -1)),
        'bestFitness': float(gbest_fitness)
    })

    return returnData


def glrtqcsig4pso(xVec, params, returnxVec=0):
    """
    Improved fitness function calculation
    """
    # Ensure input is CuPy array
    if isinstance(xVec, np.ndarray):
        xVec = cp.asarray(xVec)

    # Ensure input dimensions are correct
    if xVec.ndim == 1:
        xVec = xVec.reshape(1, -1)

    # Check if parameters are in valid range
    validPts = crcbchkstdsrchrng(xVec)
    nPoints = xVec.shape[0]

    # Initialize fitness array
    fitVal = cp.full(nPoints, cp.inf)

    # Convert standard range [0,1] to actual parameter range
    xVecReal = s2rv(xVec, params)

    # Calculate fitness for each valid point
    for i in range(nPoints):
        if validPts[i]:
            fitVal[i] = ssrqc(xVecReal[i], params)

    if returnxVec:
        return fitVal, xVecReal
    else:
        return fitVal


def ssrqc(x, params):
    """
    Calculate optimal SNR for signal self-match
    """
    # Generate signal
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5])

    # Normalize signal
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # Calculate inner product (projection)
    inPrd = innerprodpsd(params['dataY'], qc, params['sampFreq'], params['psdHigh'])

    # Return negative squared inner product (minimization problem)
    return -cp.abs(inPrd) ** 2


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    Normalize signal according to PSD
    """
    nSamples = len(sigVec)

    # Build complete PSD vector (positive and negative frequencies)
    if psdVec.shape[0] > 1:  # Ensure there's more than one element
        psdVec4Norm = cp.concatenate([psdVec, psdVec[-2:0:-1]])
    else:
        # Handle special case
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[0] = psdVec[0]

    # Calculate normalization factor for signal
    fft_sig = cp.fft.fft(sigVec)

    # Calculate normalized squared sum
    normSigSqrd = cp.sum((cp.abs(fft_sig) ** 2) / psdVec4Norm) / (sampFreq * nSamples)

    # Calculate normalization factor
    normFac = snr / cp.sqrt(cp.abs(normSigSqrd))  # Use absolute value to avoid complex number issues

    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """
    Calculate inner product considering PSD
    """
    # Ensure input vectors have consistent length
    if len(xVec) != len(yVec):
        # Adjust length to match
        min_len = min(len(xVec), len(yVec))
        xVec = xVec[:min_len]
        yVec = yVec[:min_len]

    nSamples = len(xVec)

    # Build complete PSD vector
    if psdVals.shape[0] > 1:  # Ensure there's more than one element
        psdVec4Norm = cp.concatenate([psdVals, psdVals[-2:0:-1]])
    else:
        # Handle special case
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[0] = psdVals[0]

    # Ensure length matches
    if len(psdVec4Norm) != nSamples:
        if len(psdVec4Norm) > nSamples:
            psdVec4Norm = psdVec4Norm[:nSamples]
        else:
            psdVec4Norm = cp.pad(psdVec4Norm, (0, nSamples - len(psdVec4Norm)), 'constant',
                                 constant_values=psdVec4Norm[-1])

    # Calculate FFT
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)

    # Calculate inner product
    inner_product = cp.sum((fftX * cp.conj(fftY)) / psdVec4Norm) / (sampFreq * nSamples)

    # Return real part
    return cp.real(inner_product)


def s2rv(xVec, params):
    """
    Convert parameters from standard range [0,1] to actual range
    """
    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    """
    Check if particles are within standard range [0,1]
    """
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)

    # Check if all elements in each row are within [0,1] range
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)


def calculate_snr_pycbc(signal, psd, fs):
    """
    Calculate SNR using PyCBC
    """
    # Ensure data is NumPy array
    if isinstance(signal, cp.ndarray):
        signal = cp.asnumpy(signal)
    if isinstance(psd, cp.ndarray):
        psd = cp.asnumpy(psd)

    # Create PyCBC TimeSeries object
    delta_t = 1.0 / fs
    ts_signal = TimeSeries(signal, delta_t=delta_t)

    # Create PyCBC FrequencySeries object
    delta_f = 1.0 / (len(signal) * delta_t)
    psd_series = FrequencySeries(psd, delta_f=delta_f)

    # Calculate SNR using matched_filter
    snr = pycbc.filter.matched_filter(ts_signal, ts_signal, psd=psd_series, low_frequency_cutoff=10.0)

    # Get maximum SNR value
    max_snr = abs(snr).max()

    return float(max_snr)


def analyze_mismatch(data, h_lens, samples, psdHigh):
    """
    Calculate mismatch using PyCBC match function
    """
    # Use PyCBC match function for more accurate mismatch calculation
    match_value = pycbc_calculate_match(h_lens, data, samples, psdHigh)

    # Calculate mismatch as 1 - match
    epsilon = 1 - match_value
    return epsilon
