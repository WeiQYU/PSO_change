import numpy as np
from scipy.fftpack import fft
from pycbc.types import FrequencySeries, TimeSeries
from tqdm import tqdm
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

    # Ensure input is numpy array
    if not isinstance(dataX, np.ndarray):
        dataX_cpu = np.asarray(dataX)
    else:
        dataX_cpu = dataX

    # Generate gravitational wave signal
    t = dataX_cpu  # Time series

    # Calculate signal in valid region before merger
    valid_idx = t < tc
    t_valid = t[valid_idx]

    # Initialize waveform
    h = np.zeros_like(t)

    if np.sum(valid_idx) > 0:  # Ensure there's a valid region
        # Calculate frequency evolution parameter Theta
        Theta = c ** 3 * (tc - t_valid) / (5 * G * m_c)

        # Calculate amplitude
        A_gw = (G * m_c / (c ** 2 * r)) * Theta ** (-1 / 4)

        # Calculate phase
        phase = 2 * phi_c - 2 * Theta ** (5 / 8)

        # Generate waveform
        h[valid_idx] = A_gw * np.cos(phase)

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
    h_fft = np.fft.fft(h)

    # Calculate frequency array
    dt = t[1] - t[0]  # Sampling interval
    fs = 1 / dt  # Sampling frequency
    freqs = np.fft.fftfreq(n, dt)

    # Calculate lens transfer function F(f) = 1 + A * exp(i * Phi)
    # where Phi = 2πf * delta_t
    Phi = 2 * np.pi * freqs * delta_t
    lens_transfer = 1 + A * np.exp(1j * Phi)

    # Apply lensing effect in frequency domain
    h_lensed_fft = h_fft * lens_transfer

    # Convert back to time domain
    h_lens = np.real(np.fft.ifft(h_lensed_fft))

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
        # Ensure input is numpy array
        if not isinstance(dataX, np.ndarray):
            t = np.asarray(dataX)
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
    if isinstance(signal1, np.ndarray):
        signal1 = np.asarray(signal1)
    if isinstance(signal2, np.ndarray):
        signal2 = np.asarray(signal2)
    if isinstance(psd, np.ndarray):
        psd = np.asarray(psd)

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
        Actual parameters for validation and error calculation only (NOT used in algorithm)

    Returns:
    --------
    result : dict
        Classification result and details with enhanced metrics
    """
    print("=========== 判断结果 (CPU版本) ===========")
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

    # 只进行标准归一化，不进行振幅优化
    unlensed_signal, normFac = normsig4psd(unlensed_signal, sampFreq, psdHigh, 1)

    # Calculate SNR and match for unlensed model
    unlensed_snr = calculate_matched_filter_snr(unlensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'unlensed_snr (CPU版本): {unlensed_snr}')

    # Update result with unlensed_signal info
    result.update({
        'unlensed_signal': unlensed_signal,
        'unlensed_snr': unlensed_snr
    })

    # Check if SNR < 8 (noise) - highest priority
    print("检测是不是噪声")
    if unlensed_snr < 8:
        print("ok,是噪声")
        result['message'] = "This is noise (CPU版本)"
        result['classification'] = "noise"
        return result
    print("不是噪声")

    # Calculate match using pycbc.filter.match
    unlensed_match = pycbc_calculate_match(unlensed_signal, dataY_only_signal, sampFreq, psdHigh)
    result['unlensed_match'] = unlensed_match
    print(f"Unlensed match (CPU版本): {unlensed_match}")

    # Generate lensed signal for comparison
    lensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)

    # 只进行标准归一化，不进行振幅优化
    lensed_signal, _ = normsig4psd(lensed_signal, sampFreq, psdHigh, 1)

    # Calculate SNR and match for lensed model
    lensed_snr = calculate_matched_filter_snr(lensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'lensed_snr (CPU版本): {lensed_snr}')

    lensed_match = pycbc_calculate_match(lensed_signal, dataY_only_signal, sampFreq, psdHigh)
    print(f"Lensed match (CPU版本): {lensed_match}")

    # Update result with lensed signal info
    result.update({
        'lensed_signal': lensed_signal,
        'lensed_snr': lensed_snr,
        'lensed_match': lensed_match
    })

    # Model comparison metrics
    model_comparison = {
        'snr_difference': lensed_snr - unlensed_snr,
        'match_difference': lensed_match - unlensed_match,
        'snr_ratio': lensed_snr / unlensed_snr if unlensed_snr > 0 else float('inf'),
        'match_ratio': lensed_match / unlensed_match if unlensed_match > 0 else float('inf')
    }
    result['model_comparison'] = model_comparison

    # Classification based solely on the value of A
    # A < 0.01 indicates unlensed signal, A >= 0.01 indicates lensed signal
    if A < 0.01:
        result['is_lensed'] = False
        result['message'] = f"This is an unlensed signal (A = {A:.6f} < 0.01) [CPU版本]"
        result['classification'] = "signal"
    else:
        result['is_lensed'] = True
        result['message'] = f"This is a lens signal (A = {A:.6f} >= 0.01) [CPU版本]"
        result['classification'] = "lens_signal"

    # Compare with actual parameters if provided (ONLY for evaluation, NOT used in algorithm)
    if actual_params is not None:
        # Convert parameters to the same units for comparison
        actual_r_log10 = np.log10(actual_params.get('source_distance', 0)) if actual_params.get('source_distance', 0) > 0 else 0
        actual_m_c_log10 = np.log10(actual_params.get('chirp_mass', 0)) if actual_params.get('chirp_mass', 0) > 0 else 0

        # Calculate relative errors
        param_errors = {
            'r_error': (10 ** r - 10 ** actual_r_log10) / 10 ** actual_r_log10 if actual_r_log10 > 0 else float('inf'),
            'm_c_error': (10 ** m_c - 10 ** actual_m_c_log10) / 10 ** actual_m_c_log10 if actual_m_c_log10 > 0 else float('inf'),
            'tc_error': (tc - actual_params.get('merger_time', 0)) / actual_params.get('merger_time', 1),
            'phi_c_error': min(abs(phi_c - actual_params.get('phase', 0) * 2 * np.pi),
                               abs(phi_c - actual_params.get('phase', 0) * 2 * np.pi - 2 * np.pi)) / (2 * np.pi),
            'A_error': (A - actual_params.get('flux_ratio', 0)) / actual_params.get('flux_ratio', 1),
            'delta_t_error': (delta_t - actual_params.get('time_delay', 0)) / actual_params.get('time_delay', 1),
            'classification_correct': (result['is_lensed'] == (actual_params.get('flux_ratio', 0) >= 0.01))
        }
        result['parameter_errors'] = param_errors

    return result


def calculate_matched_filter_snr(signal, template, psd, fs):
    """计算匹配滤波SNR，使用template作为模板"""
    # 确保数据是NumPy数组
    if isinstance(signal, np.ndarray):
        signal_np = np.asarray(signal)
    else:
        signal_np = signal

    if isinstance(template, np.ndarray):
        template_np = np.asarray(template)
    else:
        template_np = template

    if isinstance(psd, np.ndarray):
        psd_np = np.asarray(psd)
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
    Particle Swarm Optimization main function for multiple runs (CPU版本)

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
        Actual parameters for validation and error calculation only (NOT used in algorithm)

    Returns:
    --------
    outResults : dict
        Enhanced results of PSO optimization
    outStruct : list
        Detailed results of each PSO run
    """
    print("运行PSO分析 - CPU版本")

    # Transfer data to CPU (ensure numpy arrays)
    inParams['dataX'] = np.asarray(inParams['dataX'])
    inParams['dataY'] = np.asarray(inParams['dataY'])
    inParams['psdHigh'] = np.asarray(inParams['psdHigh'])
    inParams['rmax'] = np.asarray(inParams['rmax'])
    inParams['rmin'] = np.asarray(inParams['rmin'])

    # Add signal-only data if provided
    if 'dataY_only_signal' in inParams:
        inParams['dataY_only_signal'] = np.asarray(inParams['dataY_only_signal'])
    else:
        inParams['dataY_only_signal'] = inParams['dataY']  # Use full data if signal-only not provided

    nSamples = len(inParams['dataX'])
    nDim = 6  # Fixed to 6 dimensions for gravitational wave problem

    # Create fitness function handle
    fHandle = lambda x, returnxVec: glrtqcsig4pso(x, inParams, returnxVec)

    outStruct = [{} for _ in range(nRuns)]

    # Enhanced output structure
    outResults = {
        'allRunsOutput': [],
        'bestRun': None,
        'bestFitness': None,
        'bestSig': np.zeros(nSamples),
        'r': None,
        'm_c': None,
        'tc': None,
        'phi_c': None,
        'A': None,
        'delta_t': None,
        'is_lensed': False,
        'lensing_message': "",
        'classification': None,
        'actual_params': actual_params,  # Store for evaluation only
        'param_errors': {},
        'model_comparison': {}
    }

    # Run PSO multiple times with different random seeds
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1

        # Set different random seeds to ensure different results in multiple runs
        seed_value = int(time.time()) + lpruns * 1000
        np.random.seed(seed_value)

        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)

        print(f"Run {lpruns + 1} completed with best fitness: {outStruct[lpruns]['bestFitness']}")

    # Process results from all runs
    fitVal = np.zeros(nRuns)
    for lpruns in range(nRuns):
        allRunsOutput = {
            'fitVal': 0,
            'r': 0,
            'm_c': 0,
            'tc': 0,
            'phi_c': 0,
            'A': 0,
            'delta_t': 0,
            'estSig': np.zeros(nSamples),
            'totalFuncEvals': [],
            'is_lensed': False,
            'lensing_message': "",
            'classification': "noise"
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']

        # Ensure dimensions are handled correctly
        bestLocation = np.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)

        # Get parameters from best location
        _, params = fHandle(bestLocation, returnxVec=1)

        # Handle parameter dimensions
        if isinstance(params, list) and len(params) > 0:
            params = params[0]
        elif isinstance(params, np.ndarray) and params.ndim > 1 and params.shape[0] == 1:
            params = params[0]

        # Convert to numpy if needed
        if isinstance(params, np.ndarray):
            params = np.asarray(params)

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
                'dataY_only_signal': inParams['dataY_only_signal']
            }

            # Execute two-step matching
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
            param_errors = matching_result.get('parameter_errors', {})

        else:
            # Use original method to generate signal
            print('未使用两步匹配过程 - CPU版本')
            is_lensed = A >= 0.01
            use_lensing = is_lensed

            estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t, use_lensing=use_lensing)
            estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)

            # Generate classification message
            if is_lensed:
                lensing_message = f"This is a lens signal (A = {A:.6f} >= 0.01) [CPU版本]"
                classification = "lens_signal"
            else:
                lensing_message = f"This is an unlensed signal (A = {A:.6f} < 0.01) [CPU版本]"
                classification = "signal"

            model_comparison = {}
            param_errors = {}

        # Calculate SNR using matched filtering against the pure signal
        run_sig = np.real(estSig)
        run_snr_pycbc = calculate_matched_filter_snr(run_sig, inParams['dataY_only_signal'],
                                                     inParams['psdHigh'], inParams['sampFreq'])

        # Update output with SNR calculation and enhanced metrics
        allRunsOutput.update({
            'fitVal': float(fitVal[lpruns]),
            'r': r,
            'm_c': m_c,
            'tc': tc,
            'phi_c': phi_c,
            'A': A,
            'delta_t': delta_t,
            'estSig': np.asarray(estSig),
            'totalFuncEvals': outStruct[lpruns]['totalFuncEvals'],
            'is_lensed': is_lensed,
            'lensing_message': lensing_message,
            'classification': classification,
            'SNR_pycbc': float(run_snr_pycbc),
            'model_comparison': model_comparison,
            'param_errors': param_errors
        })

        outResults['allRunsOutput'].append(allRunsOutput)

    # Find best run
    fitVal_np = np.asarray(fitVal)
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
        'param_errors': outResults['allRunsOutput'][bestRun].get('param_errors', {})
    })

    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """
    PSO core algorithm implementation (CPU版本)

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
        'popsize': 100,  # Updated default to match requirements
        'maxSteps': 3000,  # Updated default to match requirements
        'c1': 2.0,
        'c2': 2.0,
        'max_velocity': 0.5,
        'w_start': 0.9,
        'w_end': 0.4,
        'run': 1,
        'nbrhdSz': 4,
        'init_strategy': 'uniform',
        'disable_early_stop': False
    }

    # Update parameters
    psoParams.update(kwargs)

    # Ensure random number reproducibility
    if 'seed' in psoParams:
        np.random.seed(psoParams['seed'])

    # Initialize return data structure
    returnData = {
        'totalFuncEvals': 0,
        'bestLocation': np.zeros((1, nDim)),
        'bestFitness': np.inf,
        'fitnessHistory': []
    }

    # Standard initialization strategy
    if psoParams['init_strategy'] == 'uniform':
        particles = np.random.rand(psoParams['popsize'], nDim)
    elif psoParams['init_strategy'] == 'gaussian':
        particles = np.random.normal(0.5, 0.15, (psoParams['popsize'], nDim))
        particles = np.clip(particles, 0, 1)
    elif psoParams['init_strategy'] == 'sobol':
        particles = np.zeros((psoParams['popsize'], nDim))
        for i in range(psoParams['popsize']):
            for j in range(nDim):
                segment = i % 10
                particles[i, j] = (segment / 10) + np.random.rand() / 10
    elif psoParams['init_strategy'] == 'boundary':
        particles = np.zeros((psoParams['popsize'], nDim))
        for i in range(psoParams['popsize']):
            if i % 3 == 0:
                particles[i] = np.random.rand(nDim) * 0.3
            elif i % 3 == 1:
                particles[i] = 0.7 + np.random.rand(nDim) * 0.3
            else:
                particles[i] = np.random.rand(nDim)
    else:
        particles = np.random.rand(psoParams['popsize'], nDim)

    # Initialize velocities
    velocities = np.random.uniform(-0.05, 0.05, (psoParams['popsize'], nDim))

    # Evaluate initial fitness
    fitness = np.zeros(psoParams['popsize'])
    for i in range(psoParams['popsize']):
        fitness[i] = fitfuncHandle(particles[i:i + 1], returnxVec=0)

    # Initialize personal best and global best
    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    # Find global best
    gbest_idx = np.argmin(pbest_fitness)
    gbest = pbest[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx].copy()

    # Record initial fitness
    returnData['fitnessHistory'].append(float(gbest_fitness))

    total_evals = psoParams['popsize']

    # Early stopping setup
    if psoParams['disable_early_stop']:
        no_improvement_count = 0
        max_no_improvement = psoParams['maxSteps'] * 10
    else:
        no_improvement_count = 0
        prev_best_fitness = float(gbest_fitness)
        max_no_improvement = 10000
        min_fitness_improvement = 1e-20

    # Create progress bar
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # Update inertia weight - linear decrease
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

            # Velocity reset to escape local optima
            if step > 0 and step % 30 == 0:
                reset_indices = np.random.choice(psoParams['popsize'], size=psoParams['popsize'] // 4, replace=False)
                for idx in reset_indices:
                    velocities[idx] = np.random.uniform(-0.4, 0.4, nDim)

                if step % 100 == 0:
                    w = np.random.uniform(psoParams['w_end'], psoParams['w_start'])

            # Update each particle
            for i in range(psoParams['popsize']):
                # Get local best (ring topology)
                neighbors = []
                for j in range(psoParams['nbrhdSz']):
                    idx = (i + j) % psoParams['popsize']
                    neighbors.append(idx)

                neighbor_fitness = [float(pbest_fitness[n]) for n in neighbors]
                best_neighbor_idx = np.argmin(neighbor_fitness)
                lbest_idx = neighbors[best_neighbor_idx]
                lbest = pbest[lbest_idx].copy()

                # Generate random coefficients
                r1 = np.random.rand(nDim)
                r2 = np.random.rand(nDim)

                # Standard velocity update
                velocities[i] = (w * velocities[i] +
                                 psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                 psoParams['c2'] * r2 * (lbest - particles[i]))

                # Velocity limit
                max_vel = psoParams['max_velocity'] * (1 - 0.3 * step / psoParams['maxSteps'])
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                # Update position
                particles[i] += velocities[i]

                # Handle boundary constraints - reflective boundary
                out_low = particles[i] < 0
                out_high = particles[i] > 1

                particles[i] = np.where(out_low, -particles[i], particles[i])
                particles[i] = np.where(out_high, 2 - particles[i], particles[i])

                # Ensure position is in [0,1] range
                particles[i] = np.clip(particles[i], 0, 1)

                # Reverse velocity at boundaries
                velocities[i] = np.where(out_low | out_high, -velocities[i], velocities[i])

                # Evaluate new position
                new_fitness = fitfuncHandle(particles[i:i + 1], returnxVec=0)
                fitness[i] = new_fitness
                total_evals += 1

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = new_fitness

            # Update global best
            current_best_idx = np.argmin(pbest_fitness)
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

                if step > 0.9 * psoParams['maxSteps'] and no_improvement_count >= max_no_improvement:
                    print(
                        f"Run {psoParams['run']} stopped after {step + 1} iterations: No improvement for {no_improvement_count} iterations")
                    break

            # Particle reinitialization strategy
            if step > 0 and step % 40 == 0:
                worst_indices = np.argsort(fitness)[-psoParams['popsize'] // 5:]

                for idx in worst_indices:
                    init_method = step % 5
                    if init_method == 0:
                        particles[idx] = np.random.rand(nDim)
                    elif init_method == 1:
                        particles[idx] = np.clip(gbest + np.random.normal(0, 0.3, nDim), 0, 1)
                    elif init_method == 2:
                        if np.random.rand() < 0.5:
                            particles[idx] = np.random.uniform(0, 0.2, nDim)
                        else:
                            particles[idx] = np.random.uniform(0.8, 1.0, nDim)
                    elif init_method == 3:
                        particles[idx] = np.clip(particles[idx] + np.random.uniform(-0.7, 0.7, nDim), 0, 1)
                    else:
                        for j in range(nDim):
                            particles[idx, j] = np.random.uniform(0.1, 0.9)

                    velocities[idx] = np.random.uniform(-0.3, 0.3, nDim)

                    fitness[idx] = fitfuncHandle(particles[idx:idx + 1], returnxVec=0)
                    total_evals += 1

    # Update return data when finished
    returnData.update({
        'totalFuncEvals': total_evals,
        'bestLocation': np.asarray(gbest.reshape(1, -1)),
        'bestFitness': float(gbest_fitness)
    })

    return returnData


def glrtqcsig4pso(xVec, params, returnxVec=0):
    """
    Improved fitness function calculation (CPU版本)

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
    # Ensure input is numpy array
    if isinstance(xVec, np.ndarray):
        xVec = np.asarray(xVec)

    # Ensure input dimensions are correct
    if xVec.ndim == 1:
        xVec = xVec.reshape(1, -1)

    # Check if parameters are in valid range
    validPts = crcbchkstdsrchrng(xVec)
    nPoints = xVec.shape[0]

    # Initialize fitness array
    fitVal = np.full(nPoints, np.inf)

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
    修复的适应度函数：使用归一化匹配度确保距离参数正确收敛 (CPU版本)

    Parameters:
    -----------
    x : array
        Signal parameters
    params : dict
        Input parameters

    Returns:
    --------
    fitness : float
        负的归一化匹配度（用于最小化）
    """
    # 确定是否使用透镜效应
    A = x[4]
    use_lensing = A >= 0.01

    # 生成估计信号
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5], use_lensing=use_lensing)

    # 只进行标准归一化，不进行振幅优化
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # 获取参考信号
    dataY_templ = params.get('dataY_only_signal', params['dataY_only_signal'])

    # 使用PyCBC的归一化匹配度而不是原始内积
    try:
        # 转换为numpy数组
        qc_np = np.asarray(qc)
        ref_np = np.asarray(dataY_templ)
        psd_np = np.asarray(params['psdHigh'])

        # 创建PyCBC TimeSeries对象
        delta_t = 1.0 / params['sampFreq']
        ts_estimated = TimeSeries(qc_np, delta_t=delta_t)
        ts_reference = TimeSeries(ref_np, delta_t=delta_t)

        # 创建PSD对象
        delta_f = 1.0 / (len(qc_np) * delta_t)
        psd_series = FrequencySeries(psd_np, delta_f=delta_f)

        # 计算归一化匹配度（0-1之间，1表示完美匹配）
        match_value, _ = match(ts_estimated, ts_reference, psd=psd_series, low_frequency_cutoff=10.0)

        # 检查匹配度有效性
        if match_value is None or np.isnan(match_value) or match_value <= 0:
            return np.array(1e10)

        # 返回负匹配度，这样PSO会最大化匹配度
        fitness = -float(match_value)

        return np.array(fitness)

    except Exception as e:
        # 如果PyCBC计算失败，使用备用方法
        return backup_fitness_calculation(qc, dataY_templ, params)


def backup_fitness_calculation(estimated_signal, reference_signal, params):
    """
    备用适应度计算方法：归一化相关系数 (CPU版本)
    """
    # 计算归一化相关系数
    est_energy = np.sqrt(np.sum(estimated_signal ** 2))
    ref_energy = np.sqrt(np.sum(reference_signal ** 2))

    # 避免除以零
    if est_energy < 1e-20 or ref_energy < 1e-20:
        return np.array(1e10)

    # 计算归一化内积（相关系数）
    correlation = np.sum(estimated_signal * reference_signal) / (est_energy * ref_energy)

    # 返回负相关系数
    return -np.abs(correlation)


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """
    Normalize signal according to PSD (CPU版本)

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
    if psdVec.shape[0] > 1:
        psd_len = len(psdVec)
        if psd_len < nSamples // 2 + 1:
            extended_psd = np.zeros(nSamples // 2 + 1)
            extended_psd[:psd_len] = psdVec
            extended_psd[psd_len:] = psdVec[-1]
            psdVec = extended_psd

        # 为正负频率创建完整的PSD向量
        psdVec4Norm = np.zeros(nSamples)
        psdVec4Norm[:nSamples // 2 + 1] = psdVec[:nSamples // 2 + 1]
        psdVec4Norm[nSamples // 2 + 1:] = psdVec[1:nSamples // 2][::-1]
    else:
        psdVec4Norm = np.ones(nSamples) * psdVec[0]

    # 确保PSD没有零值
    min_psd = np.max(psdVec4Norm) * 1e-14
    psdVec4Norm = np.maximum(psdVec4Norm, min_psd)

    # 计算信号的归一化因子
    fft_sig = np.fft.fft(sigVec)

    # 计算归一化平方和
    normSigSqrd = np.sum((np.abs(fft_sig) ** 2) / psdVec4Norm) / (sampFreq * nSamples)

    # 避免除以零或非常小的值
    if np.abs(normSigSqrd) < 1e-10:
        normFac = 0
    else:
        normFac = snr / np.sqrt(np.abs(normSigSqrd))

    return normFac * sigVec, normFac


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """
    Calculate inner product considering PSD (CPU版本)

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
        min_len = min(len(xVec), len(yVec))
        xVec = xVec[:min_len]
        yVec = yVec[:min_len]

    nSamples = len(xVec)

    # 改进PSD处理
    if psdVals.shape[0] > 1:
        psd_len = len(psdVals)
        if psd_len < nSamples // 2 + 1:
            extended_psd = np.zeros(nSamples // 2 + 1)
            extended_psd[:psd_len] = psdVals
            extended_psd[psd_len:] = psdVals[-1]
            psdVals = extended_psd

        psdVec4Norm = np.zeros(nSamples)
        psdVec4Norm[:nSamples // 2 + 1] = psdVals[:nSamples // 2 + 1]
        psdVec4Norm[nSamples // 2 + 1:] = psdVals[1:nSamples // 2][::-1]
    else:
        psdVec4Norm = np.ones(nSamples) * psdVals[0]

    # 确保PSD没有零值
    min_psd = np.max(psdVec4Norm) * 1e-14
    psdVec4Norm = np.maximum(psdVec4Norm, min_psd)

    # 计算FFT
    fftX = np.fft.fft(xVec)
    fftY = np.fft.fft(yVec)

    # 计算内积
    inner_product = np.sum((fftX * np.conj(fftY)) / psdVec4Norm) / (sampFreq * nSamples)

    return np.real(inner_product)


def s2rv(xVec, params):
    """
    Convert parameters from standard range [0,1] to actual range (CPU版本)

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
    rmax = np.asarray(params['rmax'])
    rmin = np.asarray(params['rmin'])

    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    """
    Check if particles are within standard range [0,1] (CPU版本)

    Parameters:
    -----------
    xVec : array
        Particle position

    Returns:
    --------
    valid_pts : array
        Boolean array indicating valid points
    """
    if not isinstance(xVec, np.ndarray):
        xVec = np.asarray(xVec)

    return np.all((xVec >= 0) & (xVec <= 1), axis=1)


def calculate_snr_pycbc(signal, psd, fs):
    """
    Calculate SNR using PyCBC (CPU版本)

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
    if isinstance(signal, np.ndarray):
        signal = np.asarray(signal)
    if isinstance(psd, np.ndarray):
        psd = np.asarray(psd)

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
    Calculate mismatch using PyCBC match function (CPU版本)

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
    match_value = pycbc_calculate_match(h_lens, data, samples, psdHigh)
    epsilon = 1 - match_value
    return epsilon