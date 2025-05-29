import cupy as cp
from cupyx.scipy.fftpack import fft
from pycbc.types import FrequencySeries, TimeSeries
from tqdm import tqdm
import numpy as np
import pycbc.types
from pycbc.filter import match, matched_filter
import scipy.constants as const
import time
import math

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
    'two_step_matching',
    'calculate_matched_filter_snr'
]


def generate_unlensed_gw(dataX, r, m_c, tc, phi_c):
    """
    Generate unlensed gravitational wave signal

    Parameters:
    -----------
    dataX : array
        Time series
    r : float
        Distance (log10 of distance in Mpc)
    m_c : float
        Log10 of chirp mass in solar masses
    tc : float
        Merger time
    phi_c : float
        Coalescence phase

    Returns:
    --------
    h : array
        Gravitational wave signal
    """
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
    """
    Apply lensing effect to a gravitational wave signal

    Parameters:
    -----------
    h : array
        Unlensed gravitational wave signal
    t : array
        Time series
    A : float
        Magnification factor
    delta_t : float
        Time delay
    tc : float
        Merger time

    Returns:
    --------
    h_lens : array
        Lensed gravitational wave signal
    """
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
    """
    Generate gravitational wave signal with optional lensing effect

    Parameters:
    -----------
    dataX : array
        Time series
    r : float
        Distance (log10 of distance in Mpc)
    m_c : float
        Log10 of chirp mass in solar masses
    tc : float
        Merger time
    phi_c : float
        Coalescence phase
    A : float
        Magnification factor
    delta_t : float
        Time delay
    use_lensing : bool
        Whether to apply lensing effect

    Returns:
    --------
    h : array
        Gravitational wave signal
    """
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
    """
    Calculate match between two signals using PyCBC

    Parameters:
    -----------
    signal1 : array
        First signal
    signal2 : array
        Second signal
    fs : float
        Sampling frequency
    psd : array
        Power spectral density

    Returns:
    --------
    match_value : float
        Match value between signals
    """
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


def two_step_matching(params, dataY, psdHigh, sampFreq, actual_params=None):
    """
    Enhanced two-step matching process to classify gravitational wave signals
    based on the value of A. A < 0.01 indicates unlensed signal, A >= 0.01 indicates
    lensed signal.

    Parameters:
    -----------
    params : dict
        Signal parameters
    dataY : array
        Observed data
    psdHigh : array
        Power spectral density
    sampFreq : float
        Sampling frequency
    actual_params : dict, optional
        Actual parameters for validation and error calculation only

    Returns:
    --------
    result : dict
        Classification result and details with enhanced metrics
    """
    print("=========== 判断结果 ===========")
    # Extract parameters
    r = params.get('r')
    m_c = params.get('m_c')
    tc = params.get('tc')
    phi_c = params.get('phi_c')
    A = params.get('A')
    delta_t = params.get('delta_t')
    dataX = params.get('dataX')
    dataY_only_signal = params.get('dataY_only_signal')  # 纯信号数据用于匹配

    # Initialize enhanced result
    result = {
        'unlensed_signal': None,
        'unlensed_snr': None,
        'unlensed_match': None,
        'lensed_signal': None,
        'lensed_snr': None,
        'lensed_match': None,
        'is_lensed': False,
        'message': "",
        'classification': "noise",  # Default classification
        'parameter_errors': {},  # Store errors compared to actual values (only for evaluation)
        'model_comparison': {},  # Store model comparison metrics
        'actual_comparison': {}  # Store comparison with actual values (only for evaluation)
    }

    # Step 1: Generate unlensed signal and calculate match
    unlensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=False)

    # Normalize signal
    unlensed_signal, normFac = normsig4psd(unlensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY_only_signal, unlensed_signal, sampFreq, psdHigh)
    unlensed_signal = estAmp * unlensed_signal

    # Calculate SNR and match for unlensed model
    unlensed_snr = calculate_matched_filter_snr(unlensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'unlensed_snr: {unlensed_snr}')

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
    print("不是噪声")

    # Calculate match using pycbc.filter.match
    unlensed_match = pycbc_calculate_match(unlensed_signal, dataY_only_signal, sampFreq, psdHigh)
    result['unlensed_match'] = unlensed_match
    print(f"Unlensed match: {unlensed_match}")

    # Generate lensed signal for comparison
    lensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)

    # Normalize signal
    lensed_signal, _ = normsig4psd(lensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY_only_signal, lensed_signal, sampFreq, psdHigh)
    lensed_signal = estAmp * lensed_signal

    # Calculate SNR and match for lensed model
    lensed_snr = calculate_matched_filter_snr(lensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'lensed_snr: {lensed_snr}')

    lensed_match = pycbc_calculate_match(lensed_signal, dataY_only_signal, sampFreq, psdHigh)
    print(f"Lensed match: {lensed_match}")

    # Update result with lensed signal info
    result.update({
        'lensed_signal': lensed_signal,
        'lensed_snr': lensed_snr,
        'lensed_match': lensed_match
    })

    # Model comparison metrics
    model_comparison = {
        'snr_difference': lensed_snr - unlensed_snr,
        'match_difference': lensed_match - unlensed_match,  # Changed from mismatch difference
        'snr_ratio': lensed_snr / unlensed_snr if unlensed_snr > 0 else float('inf'),
        'match_ratio': lensed_match / unlensed_match if unlensed_match > 0 else float('inf')
    }
    result['model_comparison'] = model_comparison

    # Classification based solely on the value of A
    # A < 0.01 indicates unlensed signal, A >= 0.01 indicates lensed signal
    if A < 0.01:
        result['is_lensed'] = False
        result['message'] = f"This is an unlensed signal (A = {A:.6f} < 0.01)"
        result['classification'] = "signal"
    else:
        result['is_lensed'] = True
        result['message'] = f"This is a lens signal (A = {A:.6f} >= 0.01)"
        result['classification'] = "lens_signal"

    # Compare with actual parameters if provided (for evaluation only)
    if actual_params is not None:
        # Convert parameters to the same units for comparison
        actual_r_log10 = cp.log10(actual_params.get('source_distance', 0)) if actual_params.get('source_distance',
                                                                                                0) > 0 else 0
        actual_m_c_log10 = cp.log10(actual_params.get('chirp_mass', 0)) if actual_params.get('chirp_mass', 0) > 0 else 0

        # Calculate relative errors
        param_errors = {
            'r_error': (10 ** r - 10 ** actual_r_log10) / 10 ** actual_r_log10 if actual_r_log10 > 0 else float('inf'),
            'm_c_error': (
                                 10 ** m_c - 10 ** actual_m_c_log10) / 10 ** actual_m_c_log10 if actual_m_c_log10 > 0 else float(
                'inf'),
            'tc_error': (tc - actual_params.get('merger_time', 0)) / actual_params.get('merger_time', 1),
            'phi_c_error': min(abs(phi_c - actual_params.get('phase', 0) * 2 * cp.pi),
                               abs(phi_c - actual_params.get('phase', 0) * 2 * cp.pi - 2 * cp.pi)) / (2 * cp.pi),
            'A_error': (A - actual_params.get('flux_ratio', 0)) / actual_params.get('flux_ratio', 1),
            'delta_t_error': (delta_t - actual_params.get('time_delay', 0)) / actual_params.get('time_delay', 1),
            'classification_correct': (result['is_lensed'] == (actual_params.get('flux_ratio', 0) >= 0.01))
        }
        result['parameter_errors'] = param_errors

        # Create detailed actual value comparison
        actual_comparison = {
            'actual_is_lensed': actual_params.get('flux_ratio', 0) >= 0.01,
            'estimated_is_lensed': result['is_lensed'],
            'classification_matches_actual': param_errors['classification_correct'],
            'parameters': {
                'r': {'estimated': r, 'actual_log10': actual_r_log10,
                      'actual': 10 ** actual_r_log10 if actual_r_log10 > 0 else 0},
                'm_c': {'estimated': m_c, 'actual_log10': actual_m_c_log10,
                        'actual': 10 ** actual_m_c_log10 if actual_m_c_log10 > 0 else 0},
                'tc': {'estimated': tc, 'actual': actual_params.get('merger_time', 0)},
                'phi_c': {'estimated': phi_c, 'actual_radians': actual_params.get('phase', 0) * 2 * cp.pi},
                'A': {'estimated': A, 'actual': actual_params.get('flux_ratio', 0)},
                'delta_t': {'estimated': delta_t, 'actual': actual_params.get('time_delay', 0)}
            }
        }
        result['actual_comparison'] = actual_comparison

        # Enhance message with actual comparison
        if result['is_lensed'] != (actual_params.get('flux_ratio', 0) >= 0.01):
            actual_type = "unlensed" if actual_params.get('flux_ratio', 0) < 0.01 else "lensed"
            result['message'] += f" - MISCLASSIFIED (actual signal is {actual_type})"
        else:
            result['message'] += f" - CORRECT CLASSIFICATION"

    return result


def calculate_matched_filter_snr(signal, template, psd, fs):
    """计算匹配滤波SNR，使用template作为模板"""
    # 确保数据是NumPy数组
    if isinstance(signal, cp.ndarray):
        signal_np = cp.asnumpy(signal)
    else:
        signal_np = signal

    if isinstance(template, cp.ndarray):
        template_np = cp.asnumpy(template)
    else:
        template_np = template

    if isinstance(psd, cp.ndarray):
        psd_np = cp.asnumpy(psd)
    else:
        psd_np = psd

    # 创建PyCBC TimeSeries对象
    delta_t = 1.0 / fs
    ts_signal = TimeSeries(signal_np, delta_t=delta_t)
    ts_template = TimeSeries(template_np, delta_t=delta_t)

    # 创建PSD对象
    delta_f = 1.0 / (len(signal_np) * delta_t)
    psd_series = FrequencySeries(psd_np, delta_f=delta_f)

    # 使用matched_filter计算SNR
    snr = matched_filter(ts_template, ts_signal, psd=psd_series, low_frequency_cutoff=10.0)

    # 返回最大SNR值
    return abs(snr).max()


def crcbqcpsopsd(inParams, psoParams, nRuns, use_two_step=True, actual_params=None):
    """
    Particle Swarm Optimization main function for multiple runs

    Parameters:
    -----------
    inParams : dict
        Input parameters
    psoParams : dict
        PSO configuration parameters
    nRuns : int
        Number of PSO runs
    use_two_step : bool
        Whether to use two-step matching process
    actual_params : dict, optional
        Actual parameters for validation and error calculation only

    Returns:
    --------
    outResults : dict
        Enhanced results of PSO optimization
    outStruct : list
        Detailed results of each PSO run
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

    # Set default use_lensing parameter to False to start with unlensed models
    inParams['use_lensing'] = False

    nSamples = len(inParams['dataX'])
    nDim = 6  # Fixed to 6 dimensions for gravitational wave problem

    # Create fitness function handle
    fHandle = lambda x, returnxVec: glrtqcsig4pso(x, inParams, returnxVec)

    outStruct = [{} for _ in range(nRuns)]

    # Enhanced output structure with actual parameter comparison
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
        'classification': None,  # Add classification field
        'actual_params': actual_params,  # Store the actual parameters for comparison only
        'param_errors': {},  # Will store parameter errors
        'model_comparison': {}  # Will store model comparison metrics
    }

    # Run PSO multiple times with different random seeds
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1

        # Set different random seeds to ensure different results in multiple runs
        seed_value = int(time.time()) + lpruns * 1000
        cp.random.seed(seed_value)

        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)

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

            # Execute two-step matching with new A-based classification
            matching_result = two_step_matching(
                param_dict,
                inParams['dataY'],
                inParams['psdHigh'],
                inParams['sampFreq'],
                actual_params  # Pass actual parameters for evaluation only
            )

            # Use matching results
            is_lensed = matching_result['is_lensed']
            lensing_message = matching_result['message']
            classification = matching_result['classification']

            # Decide which signal to use based on classification
            if classification == "noise":
                print("是噪声")
                estSig = matching_result['unlensed_signal']
            elif classification == "signal":
                print("是未透镜")
                estSig = matching_result['unlensed_signal']
            elif classification == "lens_signal":
                print("是透镜")
                estSig = matching_result['lensed_signal']
            else:
                print("默认使用未透镜")
                estSig = matching_result['unlensed_signal']

            # Store model comparison metrics
            model_comparison = matching_result.get('model_comparison', {})

            # Store parameter errors if actual parameters were provided
            param_errors = matching_result.get('parameter_errors', {})
            actual_comparison = matching_result.get('actual_comparison', {})

        else:
            # Use original method to generate signal
            print('未使用两步匹配过程')
            # IMPORTANT: Use lensing flag based on A value
            is_lensed = A >= 0.01
            use_lensing = is_lensed

            estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t, use_lensing=use_lensing)
            estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
            # 使用纯信号而不是带噪声的数据进行匹配
            estAmp = innerprodpsd(inParams['dataY_only_signal'], estSig, inParams['sampFreq'], inParams['psdHigh'])
            estSig = estAmp * estSig

            # Generate classification message
            if is_lensed:
                lensing_message = f"This is a lens signal (A = {A:.6f} >= 0.01)"
                classification = "lens_signal"
            else:
                lensing_message = f"This is an unlensed signal (A = {A:.6f} < 0.01)"
                classification = "signal"

            model_comparison = {}
            param_errors = {}
            actual_comparison = {}

        # Calculate SNR using matched filtering against the pure signal
        run_sig = cp.real(estSig)
        run_snr_pycbc = calculate_matched_filter_snr(run_sig, inParams['dataY_only_signal'],
                                                     inParams['psdHigh'], inParams['sampFreq'])

        # Update output with SNR calculation and enhanced metrics
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
            'classification': classification,
            'SNR_pycbc': float(run_snr_pycbc),  # Add SNR value here
            'model_comparison': model_comparison,  # Add model comparison metrics
            'param_errors': param_errors,  # Add parameter errors
            'actual_comparison': actual_comparison  # Add actual comparison
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
        'classification': outResults['allRunsOutput'][bestRun]['classification'],
        'model_comparison': outResults['allRunsOutput'][bestRun].get('model_comparison', {}),
        'param_errors': outResults['allRunsOutput'][bestRun].get('param_errors', {}),
        'actual_comparison': outResults['allRunsOutput'][bestRun].get('actual_comparison', {})
    })

    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """
    PSO core algorithm implementation

    Parameters:
    -----------
    fitfuncHandle : function
        Fitness function
    nDim : int
        Dimensionality of search space
    **kwargs : dict
        PSO configuration parameters

    Returns:
    --------
    returnData : dict
        Results of PSO optimization
    """
    # Default PSO parameters
    psoParams = {
        'popsize': 50,
        'maxSteps': 2000,
        'c1': 2.0,  # Individual learning factor
        'c2': 2.0,  # Social learning factor
        'max_velocity': 0.5,  # Maximum velocity limit
        'w_start': 0.9,  # Initial inertia weight
        'w_end': 0.4,  # Final inertia weight
        'run': 1,  # Run number
        'nbrhdSz': 4,  # Neighborhood size
        'init_strategy': 'uniform',  # Initialization strategy
        'disable_early_stop': False  # Whether to disable early stopping
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
        'fitnessHistory': []  # Record fitness history for plotting
    }

    # Standard initialization strategy
    if psoParams['init_strategy'] == 'uniform':
        # Standard uniform initialization
        particles = cp.random.rand(psoParams['popsize'], nDim)
    elif psoParams['init_strategy'] == 'gaussian':
        # Gaussian initialization around the middle of the range
        particles = cp.random.normal(0.5, 0.15, (psoParams['popsize'], nDim))
        particles = cp.clip(particles, 0, 1)  # Clip to [0,1] range
    elif psoParams['init_strategy'] == 'sobol':
        # Basic quasi-random initialization with segment division
        particles = cp.zeros((psoParams['popsize'], nDim))
        for i in range(psoParams['popsize']):
            for j in range(nDim):
                segment = i % 10  # Divide range into 10 segments
                particles[i, j] = (segment / 10) + cp.random.rand() / 10
    elif psoParams['init_strategy'] == 'boundary':
        # Boundary-biased initialization (more particles near boundaries)
        particles = cp.zeros((psoParams['popsize'], nDim))
        for i in range(psoParams['popsize']):
            if i % 3 == 0:  # 1/3 of particles near lower boundary
                particles[i] = cp.random.rand(nDim) * 0.3
            elif i % 3 == 1:  # 1/3 of particles near upper boundary
                particles[i] = 0.7 + cp.random.rand(nDim) * 0.3
            else:  # 1/3 of particles uniformly distributed
                particles[i] = cp.random.rand(nDim)
    else:
        # Default to uniform distribution
        particles = cp.random.rand(psoParams['popsize'], nDim)

    # Initialize velocities - smaller initial velocities
    velocities = cp.random.uniform(-0.05, 0.05, (psoParams['popsize'], nDim))

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

    # Early stopping setup
    if psoParams['disable_early_stop']:
        no_improvement_count = 0
        max_no_improvement = psoParams['maxSteps'] * 10  # Set to an unreachable value
    else:
        no_improvement_count = 0
        prev_best_fitness = float(gbest_fitness)
        max_no_improvement = 10000  # Very loose early stopping condition
        min_fitness_improvement = 1e-20  # Very small improvement threshold

    # Create progress bar
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # Update inertia weight - linear decrease
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

            # Velocity reset to escape local optima
            if step > 0 and step % 30 == 0:
                # Reset velocity for 25% of particles
                reset_indices = cp.random.choice(psoParams['popsize'], size=psoParams['popsize'] // 4, replace=False)
                for idx in reset_indices:
                    velocities[idx] = cp.random.uniform(-0.4, 0.4, nDim)

                # Periodically change inertia weight to increase search diversity
                if step % 100 == 0:
                    w = cp.random.uniform(psoParams['w_end'], psoParams['w_start'])

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

                # Standard velocity update
                velocities[i] = (w * velocities[i] +
                                 psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                 psoParams['c2'] * r2 * (lbest - particles[i]))

                # More conservative velocity limit to prevent overshooting
                max_vel = psoParams['max_velocity'] * (1 - 0.3 * step / psoParams['maxSteps'])

                # Standard uniform velocity limit
                velocities[i] = cp.clip(velocities[i], -max_vel, max_vel)

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

                # Reset improvement counter (if early stopping enabled)
                if not psoParams['disable_early_stop']:
                    no_improvement_count = 0
                    prev_best_fitness = float(gbest_fitness)

            # Record best fitness at each step
            returnData['fitnessHistory'].append(float(gbest_fitness))

            # Early stopping logic (if enabled)
            if not psoParams['disable_early_stop']:
                current_best_fitness = float(gbest_fitness)
                fitness_improvement = abs(current_best_fitness - prev_best_fitness)

                if fitness_improvement < min_fitness_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    prev_best_fitness = current_best_fitness

                # Only stop in extreme cases, and only after completing 90% of iterations
                if step > 0.9 * psoParams['maxSteps'] and no_improvement_count >= max_no_improvement:
                    print(
                        f"Run {psoParams['run']} stopped after {step + 1} iterations: No improvement for {no_improvement_count} iterations")
                    break

            # More aggressive particle reinitialization strategy to break out of local optima
            if step > 0 and step % 40 == 0:
                # Find the worst 20% of particles
                worst_indices = cp.argsort(fitness)[-psoParams['popsize'] // 5:]

                # Standard reset logic
                for idx in worst_indices:
                    # Choose initialization strategy
                    init_method = step % 5
                    if init_method == 0:
                        # Uniform random
                        particles[idx] = cp.random.rand(nDim)
                    elif init_method == 1:
                        # Add noise around global best
                        particles[idx] = cp.clip(gbest + cp.random.normal(0, 0.3, nDim), 0, 1)
                    elif init_method == 2:
                        # Explore near boundaries
                        if cp.random.rand() < 0.5:  # Simple coin flip
                            particles[idx] = cp.random.uniform(0, 0.2, nDim)
                        else:
                            particles[idx] = cp.random.uniform(0.8, 1.0, nDim)
                    elif init_method == 3:
                        # Large random jumps
                        particles[idx] = cp.clip(particles[idx] + cp.random.uniform(-0.7, 0.7, nDim), 0, 1)
                    else:
                        # Random sampling within parameter space
                        for j in range(nDim):
                            particles[idx, j] = cp.random.uniform(0.1, 0.9)

                    # Use larger velocity range to improve exploration
                    velocities[idx] = cp.random.uniform(-0.3, 0.3, nDim)

                    # Evaluate new position
                    fitness[idx] = fitfuncHandle(particles[idx:idx + 1], returnxVec=0)
                    total_evals += 1

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

    Parameters:
    -----------
    xVec : array
        Particle position
    params : dict
        Input parameters
    returnxVec : int
        Whether to return particle position

    Returns:
    --------
    fitVal : array
        Fitness value
    xVecReal : array (optional)
        Particle position in real parameter space
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
    Calculate optimal SNR for signal self-match using matched filtering

    Parameters:
    -----------
    x : array
        Signal parameters
    params : dict
        Input parameters

    Returns:
    --------
    fitness : float
        Negative matched filtering result
    """
    # IMPORTANT: Determine lensing usage based on A parameter value
    # If A < 0.01, the signal should be unlensed; if A >= 0.01, the signal should be lensed
    A = x[4]  # The A parameter
    use_lensing = A >= 0.01

    # Generate signal based on the A parameter value
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5], use_lensing=use_lensing)

    # Normalize signal
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # 使用纯信号作为参考模板（而不是带噪声的数据）
    dataY_templ = params.get('dataY_only_signal', params['dataY_only_signal'])

    # 使用内积计算作为匹配滤波的结果
    inPrd = innerprodpsd(dataY_templ, qc, params['sampFreq'], params['psdHigh'])

    # Return negative squared inner product (to minimize)
    return -cp.abs(inPrd) ** 2


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    Normalize signal according to PSD

    Parameters:
    -----------
    sigVec : array
        Signal vector
    sampFreq : float
        Sampling frequency
    psdVec : array
        Power spectral density
    snr : float
        Desired SNR

    Returns:
    --------
    normalizedSig : array
        Normalized signal
    normFac : float
        Normalization factor
    """
    nSamples = len(sigVec)

    # 更好地处理PSD向量的边缘情况
    if psdVec.shape[0] > 1:  # 确保有多个元素
        # 如果PSD长度与FFT长度不匹配，调整大小
        psd_len = len(psdVec)
        if psd_len < nSamples // 2 + 1:
            # 扩展PSD以覆盖所有正频率
            extended_psd = cp.zeros(nSamples // 2 + 1)
            extended_psd[:psd_len] = psdVec
            # 用最后一个值填充剩余部分
            extended_psd[psd_len:] = psdVec[-1]
            psdVec = extended_psd

        # 为正负频率创建完整的PSD向量
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[:nSamples // 2 + 1] = psdVec[:nSamples // 2 + 1]  # 正频率
        psdVec4Norm[nSamples // 2 + 1:] = psdVec[1:nSamples // 2][::-1]  # 负频率（镜像）
    else:
        # 处理单值PSD的特殊情况
        psdVec4Norm = cp.ones(nSamples) * psdVec[0]

    # 确保PSD没有零值（避免除以零）
    min_psd = cp.max(psdVec4Norm) * 1e-14
    psdVec4Norm = cp.maximum(psdVec4Norm, min_psd)

    # 计算信号的归一化因子
    fft_sig = cp.fft.fft(sigVec)

    # 计算归一化平方和
    normSigSqrd = cp.sum((cp.abs(fft_sig) ** 2) / psdVec4Norm) / (sampFreq * nSamples)

    # 避免除以零或非常小的值
    if cp.abs(normSigSqrd) < 1e-10:
        normFac = 0
    else:
        # 计算归一化因子
        normFac = snr / cp.sqrt(cp.abs(normSigSqrd))  # 使用绝对值避免复数问题

    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """
    Calculate inner product considering PSD - improved matched filtering

    Parameters:
    -----------
    xVec : array
        First signal
    yVec : array
        Second signal
    sampFreq : float
        Sampling frequency
    psdVals : array
        Power spectral density

    Returns:
    --------
    inner_product : float
        Inner product
    """
    # 确保输入向量具有一致的长度
    if len(xVec) != len(yVec):
        # 调整长度匹配
        min_len = min(len(xVec), len(yVec))
        xVec = xVec[:min_len]
        yVec = yVec[:min_len]

    nSamples = len(xVec)

    # 改进PSD处理，类似于normsig4psd
    if psdVals.shape[0] > 1:  # 确保有多个元素
        # 如果PSD长度与FFT长度不匹配，调整大小
        psd_len = len(psdVals)
        if psd_len < nSamples // 2 + 1:
            # 扩展PSD以覆盖所有正频率
            extended_psd = cp.zeros(nSamples // 2 + 1)
            extended_psd[:psd_len] = psdVals
            # 用最后一个值填充剩余部分
            extended_psd[psd_len:] = psdVals[-1]
            psdVals = extended_psd

        # 为正负频率创建完整的PSD向量
        psdVec4Norm = cp.zeros(nSamples)
        psdVec4Norm[:nSamples // 2 + 1] = psdVals[:nSamples // 2 + 1]  # 正频率
        psdVec4Norm[nSamples // 2 + 1:] = psdVals[1:nSamples // 2][::-1]  # 负频率（镜像）
    else:
        # 处理单值PSD的特殊情况
        psdVec4Norm = cp.ones(nSamples) * psdVals[0]

    # 确保PSD没有零值（避免除以零）
    min_psd = cp.max(psdVec4Norm) * 1e-14
    psdVec4Norm = cp.maximum(psdVec4Norm, min_psd)

    # 计算FFT
    fftX = cp.fft.fft(xVec)
    fftY = cp.fft.fft(yVec)

    # 计算内积（匹配滤波）
    inner_product = cp.sum((fftX * cp.conj(fftY)) / psdVec4Norm) / (sampFreq * nSamples)

    # 返回实部
    return cp.real(inner_product)


def s2rv(xVec, params):
    """
    Convert parameters from standard range [0,1] to actual range

    Parameters:
    -----------
    xVec : array
        Particle position in standard range [0,1]
    params : dict
        Input parameters

    Returns:
    --------
    xVecReal : array
        Particle position in real parameter space
    """
    rmax = cp.asarray(params['rmax'])
    rmin = cp.asarray(params['rmin'])

    # Use standard ranges
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    """
    Check if particles are within standard range [0,1]

    Parameters:
    -----------
    xVec : array
        Particle position

    Returns:
    --------
    valid_pts : array
        Boolean array indicating valid points
    """
    if not isinstance(xVec, cp.ndarray):
        xVec = cp.asarray(xVec)

    # Check if all elements in each row are within [0,1] range
    return cp.all((xVec >= 0) & (xVec <= 1), axis=1)


def calculate_snr_pycbc(signal, psd, fs):
    """
    Calculate SNR using PyCBC

    Parameters:
    -----------
    signal : array
        Signal
    psd : array
        Power spectral density
    fs : float
        Sampling frequency

    Returns:
    --------
    max_snr : float
        Maximum SNR value
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

    Parameters:
    -----------
    data : array
        Reference data
    h_lens : array
        Signal to compare
    samples : float
        Sampling frequency
    psdHigh : array
        Power spectral density

    Returns:
    --------
    epsilon : float
        Mismatch value
    """
    # Use PyCBC match function for more accurate mismatch calculation
    match_value = pycbc_calculate_match(h_lens, data, samples, psdHigh)

    # Calculate mismatch as 1 - match
    epsilon = 1 - match_value
    return epsilon