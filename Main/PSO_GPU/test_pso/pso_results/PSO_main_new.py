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
    'calculate_matched_filter_snr',
    'analyze_dimension_exploration',
    'init_specialized_particles',
    'bayesian_ssrqc',
    'calculate_log_prior',
    'adaptive_pso_fitness'
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
    based on the value of A. A < 1 indicates lensed signal, A > 1 indicates
    unlensed signal.

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
        'unlensed_mismatch': None,
        'lensed_signal': None,
        'lensed_snr': None,
        'lensed_mismatch': None,
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

    # Calculate SNR and mismatch for unlensed model
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

    # Calculate mismatch using pycbc.filter.match
    unlensed_mismatch = 1 - pycbc_calculate_match(unlensed_signal, dataY_only_signal, sampFreq, psdHigh)
    result['unlensed_mismatch'] = unlensed_mismatch
    print(f"Unlensed mismatch: {unlensed_mismatch}")

    # Generate lensed signal for comparison
    lensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)

    # Normalize signal
    lensed_signal, _ = normsig4psd(lensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY_only_signal, lensed_signal, sampFreq, psdHigh)
    lensed_signal = estAmp * lensed_signal

    # Calculate SNR and mismatch for lensed model
    lensed_snr = calculate_matched_filter_snr(lensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'lensed_snr: {lensed_snr}')

    lensed_mismatch = 1 - pycbc_calculate_match(lensed_signal, dataY_only_signal, sampFreq, psdHigh)
    print(f"Lensed mismatch: {lensed_mismatch}")

    # Update result with lensed signal info
    result.update({
        'lensed_signal': lensed_signal,
        'lensed_snr': lensed_snr,
        'lensed_mismatch': lensed_mismatch
    })

    # Model comparison metrics
    model_comparison = {
        'snr_difference': lensed_snr - unlensed_snr,
        'mismatch_difference': unlensed_mismatch - lensed_mismatch,
        'snr_ratio': lensed_snr / unlensed_snr if unlensed_snr > 0 else float('inf'),
        'mismatch_ratio': unlensed_mismatch / lensed_mismatch if lensed_mismatch > 0 else float('inf')
    }
    result['model_comparison'] = model_comparison

    # Classification based solely on the value of A
    # A < 1 indicates lensed signal, A > 1 indicates unlensed signal
    if A < 1:
        result['is_lensed'] = True
        result['message'] = f"This is a lens signal (A = {A:.6f} < 1)"
        result['classification'] = "lens_signal"
    else:
        result['is_lensed'] = False
        result['message'] = f"This is an unlensed signal (A = {A:.6f} >= 1)"
        result['classification'] = "signal"

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
            'classification_correct': (result['is_lensed'] == (actual_params.get('flux_ratio', 0) < 1))
        }
        result['parameter_errors'] = param_errors

        # Create detailed actual value comparison
        actual_comparison = {
            'actual_is_lensed': actual_params.get('flux_ratio', 0) < 1,
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
        if result['is_lensed'] != (actual_params.get('flux_ratio', 0) < 1):
            actual_type = "lensed" if actual_params.get('flux_ratio', 0) < 1 else "unlensed"
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


def analyze_dimension_exploration(particles_history, dimension_names=['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']):
    """
    分析PSO在各维度上的搜索情况

    Parameters:
    -----------
    particles_history : list of arrays
        粒子的历史位置
    dimension_names : list of str
        每个维度的名称

    Returns:
    --------
    exploration : dict
        各维度的探索情况分析
    """
    exploration = {}

    # 转换粒子历史到NumPy格式，组织为每个维度的数据
    particles_np = []
    for particles in particles_history:
        if isinstance(particles, cp.ndarray):
            particles_np.append(cp.asnumpy(particles))
        else:
            particles_np.append(particles)

    particles_np = np.array(particles_np)
    n_dims = particles_np.shape[2] if len(particles_np.shape) > 2 else particles_np.shape[1]

    # 确保dimension_names和维度数量匹配
    if len(dimension_names) != n_dims:
        dimension_names = [f'dim_{i}' for i in range(n_dims)]

    # 对每个维度进行分析
    for dim in range(n_dims):
        # 提取该维度上所有粒子的历史位置
        if len(particles_np.shape) > 2:  # 如果包含多个迭代步骤
            dim_positions = particles_np[:, :, dim].flatten()
        else:  # 如果只有一个时间点
            dim_positions = particles_np[:, dim].flatten()

        # 计算覆盖率 - 将[0,1]范围分成10个桶，计算有多少桶被访问过
        bins = np.zeros(10)
        for pos in dim_positions:
            if 0 <= pos <= 1:  # 确保位置在[0,1]范围内
                bin_idx = min(int(pos * 10), 9)  # 将[0,1]映射到[0,9]
                bins[bin_idx] += 1

        coverage = np.sum(bins > 0) / 10.0
        variance = np.var(dim_positions)

        # 计算局部覆盖率 - 检查粒子是否集中在某一狭窄区域
        sorted_pos = np.sort(dim_positions)
        if len(sorted_pos) > 10:
            # 取上下四分位范围
            q1, q3 = np.percentile(sorted_pos, [25, 75])
            iqr = q3 - q1
            concentration = iqr / (max(sorted_pos) - min(sorted_pos) + 1e-10)
        else:
            concentration = 1.0

        exploration[dimension_names[dim]] = {
            'coverage': coverage,  # 整体覆盖率 [0,1]
            'variance': variance,  # 位置方差
            'concentration': concentration,  # 集中度 (较小值表示局部过度集中)
            'histogram': bins.tolist(),  # 直方图分布
            'min': np.min(dim_positions),  # 最小值
            'max': np.max(dim_positions)  # 最大值
        }

    # 计算维度间的探索平衡性
    coverage_values = [info['coverage'] for info in exploration.values()]
    variance_values = [info['variance'] for info in exploration.values()]

    exploration['overall'] = {
        'mean_coverage': np.mean(coverage_values),
        'coverage_imbalance': np.max(coverage_values) - np.min(coverage_values),
        'mean_variance': np.mean(variance_values),
        'variance_imbalance': np.max(variance_values) / (np.min(variance_values) + 1e-10)
    }

    return exploration


def init_specialized_particles(popsize, nDim, rmin, rmax, balanced=True):
    """
    初始化专门的粒子，确保涵盖所有维度

    Parameters:
    -----------
    popsize : int
        粒子群大小
    nDim : int
        维度数量
    rmin : array
        参数下界
    rmax : array
        参数上界
    balanced : bool
        是否使用平衡策略

    Returns:
    --------
    particles : array
        初始化的粒子群
    """
    # 确保rmin和rmax是CuPy数组
    if not isinstance(rmin, cp.ndarray):
        rmin = cp.asarray(rmin)
    if not isinstance(rmax, cp.ndarray):
        rmax = cp.asarray(rmax)

    # 初始化粒子群
    particles = cp.random.rand(popsize, nDim)

    if balanced and popsize >= nDim * 2:
        # 对于每个维度创建一对专门的粒子
        dim_particles = min(popsize // 2, nDim)

        for dim in range(dim_particles):
            # 创建两个粒子：一个接近下界，一个接近上界
            # 对于dim维度，使用更窄的分布；对于其他维度，使用标准随机分布

            # 接近下界的粒子
            particles[dim * 2] = cp.random.rand(nDim)  # 标准随机分布
            particles[dim * 2, dim] = cp.random.uniform(0, 0.3)  # 偏向下界

            # 接近上界的粒子
            particles[dim * 2 + 1] = cp.random.rand(nDim)  # 标准随机分布
            particles[dim * 2 + 1, dim] = cp.random.uniform(0.7, 1.0)  # 偏向上界

    return particles


def crcbqcpsopsd(inParams, psoParams, nRuns, use_two_step=True, actual_params=None, balanced_pso=True,
                 use_bayesian=True):
    """
    Enhanced Particle Swarm Optimization main function for multiple runs
    with balanced dimensions, Bayesian approach, and blind parameter estimation

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
    balanced_pso : bool
        Whether to use balanced dimension PSO strategy
    use_bayesian : bool
        Whether to use Bayesian approach for fitness calculation

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

    # Set balanced dimension PSO flag
    inParams['balanced_pso'] = balanced_pso

    # Set Bayesian approach flag
    inParams['use_bayesian'] = use_bayesian

    # Set actual parameters for comparison only (NOT for prior construction)
    if actual_params is not None:
        inParams['actual_params_for_comparison'] = actual_params

    nSamples = len(inParams['dataX'])
    nDim = 6  # Fixed to 6 dimensions for gravitational wave problem

    # Create fitness function handle based on selected approach
    if use_bayesian:
        # Use Bayesian approach with adaptive strategy
        fHandle = lambda x, returnxVec, iteration=0: adaptive_pso_fitness(x, inParams, returnxVec, iteration)
    else:
        # Use standard approach
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
        'model_comparison': {},  # Will store model comparison metrics
        'dimension_analysis': {},  # Will store dimension exploration analysis
        'bayesian_used': use_bayesian  # Track whether Bayesian approach was used
    }

    if balanced_pso:
        print("Using Balanced Dimension PSO Strategy")
        # For balanced PSO, ensure parameter names are passed to dimension analysis
        inParams['dim_names'] = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']

    if use_bayesian:
        print("Using Bayesian Approach with Adaptive Hybrid Optimization")

    # Run PSO multiple times with different random seeds
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1

        # Set different random seeds to ensure different results in multiple runs
        seed_value = int(time.time()) + lpruns * 1000
        cp.random.seed(seed_value)

        # Run PSO with balanced dimensions if requested
        if balanced_pso:
            # Different strategy for each run to ensure diversity
            dim_strategy = lpruns % 4  # Expanded to include new hybrid strategy

            if dim_strategy == 0:
                # Standard balanced dimension strategy
                currentPSOParams['use_dimension_balance'] = True
                currentPSOParams['use_dimension_groups'] = False
                currentPSOParams['use_dimension_restart'] = True
                currentPSOParams['use_adaptive_strategy'] = False
            elif dim_strategy == 1:
                # Dimension group strategy (focus on different dimension groups in different phases)
                currentPSOParams['use_dimension_balance'] = True
                currentPSOParams['use_dimension_groups'] = True
                currentPSOParams['use_dimension_restart'] = False
                currentPSOParams['use_adaptive_strategy'] = False
            elif dim_strategy == 2:
                # Hybrid strategy
                currentPSOParams['use_dimension_balance'] = True
                currentPSOParams['use_dimension_groups'] = True
                currentPSOParams['use_dimension_restart'] = True
                currentPSOParams['use_adaptive_strategy'] = False
            else:
                # New adaptive hybrid strategy
                currentPSOParams['use_dimension_balance'] = True
                currentPSOParams['use_dimension_groups'] = True
                currentPSOParams['use_dimension_restart'] = True
                currentPSOParams['use_adaptive_strategy'] = True

            # Add specialized initialization
            currentPSOParams['use_specialized_init'] = True

            # Create dimension groups for improved search
            currentPSOParams['dimension_groups'] = [
                [0, 1],  # r, m_c: Distance and mass parameters
                [2, 3],  # tc, phi_c: Time and phase parameters
                [4, 5]  # A, delta_t: Lensing parameters
            ]

        outStruct[lpruns] = crcbpso(fHandle, nDim, **currentPSOParams)

        print(f"Run {lpruns + 1} completed with best fitness: {outStruct[lpruns]['bestFitness']}")

        # Store dimension exploration analysis if available
        if 'dimension_analysis' in outStruct[lpruns]:
            outResults['dimension_analysis'][f'run_{lpruns + 1}'] = outStruct[lpruns]['dimension_analysis']

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
            'classification': "noise",  # Default classification
            'bayesian_used': use_bayesian  # Track whether Bayesian approach was used
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']

        # Ensure dimensions are handled correctly
        bestLocation = cp.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)  # Ensure 2D shape (1, nDim)

        # Get parameters from best location
        if use_bayesian:
            # For Bayesian approach, use the last iteration for final evaluation
            _, params = fHandle(bestLocation, returnxVec=1, iteration=psoParams['maxSteps'])
        else:
            # Standard approach
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

            # Update use_lensing parameter for future runs based on classification
            inParams['use_lensing'] = is_lensed

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
            use_lensing = inParams.get('use_lensing', False)  # Default to unlensed
            estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t, use_lensing=use_lensing)
            estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
            # 使用纯信号而不是带噪声的数据进行匹配
            estAmp = innerprodpsd(inParams['dataY_only_signal'], estSig, inParams['sampFreq'], inParams['psdHigh'])
            estSig = estAmp * estSig

            # Default values
            is_lensed = False
            lensing_message = "Two-step matching not performed"
            classification = "unknown"
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

        # Add dimension analysis if available
        if 'dimension_analysis' in outStruct[lpruns]:
            allRunsOutput['dimension_analysis'] = outStruct[lpruns]['dimension_analysis']

        # Add strategy information
        if 'strategy_shifts' in outStruct[lpruns]:
            allRunsOutput['strategy_shifts'] = outStruct[lpruns]['strategy_shifts']

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
        'actual_comparison': outResults['allRunsOutput'][bestRun].get('actual_comparison', {}),
        'balanced_pso_used': balanced_pso,
        'bayesian_used': use_bayesian
    })

    # Add best run dimension analysis
    if 'dimension_analysis' in outResults['allRunsOutput'][bestRun]:
        outResults['best_dimension_analysis'] = outResults['allRunsOutput'][bestRun]['dimension_analysis']

    # Add strategy shifts if available
    if 'strategy_shifts' in outResults['allRunsOutput'][bestRun]:
        outResults['strategy_shifts'] = outResults['allRunsOutput'][bestRun]['strategy_shifts']

    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """
    Improved PSO core algorithm implementation with balanced dimension exploration
    and adaptive strategy switching

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
        'disable_early_stop': False,  # Whether to disable early stopping

        # Balanced dimension PSO parameters
        'use_dimension_balance': False,  # Whether to use dimension balance
        'use_dimension_groups': False,  # Whether to use dimension groups
        'use_dimension_restart': False,  # Whether to use dimension-specific restarts
        'use_specialized_init': False,  # Whether to use specialized initialization for different dimensions
        'dimension_groups': [],  # Groups of related dimensions

        # New adaptive parameters
        'use_adaptive_strategy': False,  # Whether to use adaptive strategy switching
        'adaptation_interval': 100,  # How often to check and adapt strategy
        'exploration_threshold': 0.5,  # Threshold to switch between exploration and exploitation
        'local_search_factor': 3.0,  # Factor to increase local search intensity
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
        'fitnessHistory': [],  # Record fitness history
        'strategy_shifts': []  # Track strategy shifts for adaptive PSO
    }

    # Initialize particles differently depending on specialization setting
    if psoParams['use_specialized_init'] and 'rmin' in psoParams and 'rmax' in psoParams:
        particles = init_specialized_particles(
            psoParams['popsize'],
            nDim,
            psoParams['rmin'],
            psoParams['rmax'],
            balanced=True
        )
    else:
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

    # Evaluate initial fitness using iteration number for adaptive fitness
    fitness = cp.zeros(psoParams['popsize'])
    for i in range(psoParams['popsize']):
        fitness[i] = fitfuncHandle(particles[i:i + 1], returnxVec=0, iteration=0)

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

    # For balanced dimension PSO: initialize dimension weights and history tracking
    dimension_weights = cp.ones(nDim)  # Equal weights initially
    dimension_progress = cp.zeros(nDim)  # Track progress in each dimension
    dimension_stagnation = cp.zeros(nDim)  # Track stagnation in each dimension
    dimension_importance = cp.ones(nDim)  # Importance of each dimension

    # History of particles for dimension analysis
    particles_history = []
    if psoParams['use_dimension_balance']:
        # Store initial particles for later analysis
        particles_history.append(particles.copy())

        # Set different importance for dimension groups if specified
        if psoParams['use_dimension_groups'] and psoParams['dimension_groups']:
            # Initialize equal importance
            dimension_importance = cp.ones(nDim)

            # Weight importance based on group size (smaller groups get higher per-dimension importance)
            for group in psoParams['dimension_groups']:
                group_importance = 1.0 / len(group)
                for dim in group:
                    if 0 <= dim < nDim:
                        dimension_importance[dim] = group_importance * nDim

            # Normalize importance weights
            dimension_importance = dimension_importance / cp.sum(dimension_importance) * nDim

    # Early stopping setup
    if psoParams['disable_early_stop']:
        no_improvement_count = 0
        max_no_improvement = psoParams['maxSteps'] * 10  # 设置一个不可能达到的值
    else:
        no_improvement_count = 0
        prev_best_fitness = float(gbest_fitness)
        max_no_improvement = 10000  # 非常宽松的早停条件
        min_fitness_improvement = 1e-20  # 极小的改进阈值

    # For dimension-focused PSO: phase management
    phase_counter = 0
    current_phase = 0
    phase_length = psoParams['maxSteps'] // 4  # Default to 4 phases
    active_group = None

    if psoParams['use_dimension_groups'] and psoParams['dimension_groups']:
        # Calculate phase length based on number of groups
        n_groups = len(psoParams['dimension_groups'])
        if n_groups > 0:
            phase_length = psoParams['maxSteps'] // (n_groups + 1)  # +1 for final full-dimensional phase

            # Initialize first active group
            active_group = psoParams['dimension_groups'][0]
            print(f"Starting with dimension group: {active_group}")

    # For adaptive strategy: initialize strategy state
    current_strategy = "exploration"  # Start with exploration
    exploration_score = 1.0  # Initialize with full exploration

    # Parameters for adaptive strategy
    if psoParams['use_adaptive_strategy']:
        # Add initial strategy to history
        returnData['strategy_shifts'].append({
            'iteration': 0,
            'new_strategy': current_strategy,
            'reason': "Initial strategy"
        })

    # Create progress bar
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # For adaptive strategy, adjust parameters based on current strategy and progress
            if psoParams['use_adaptive_strategy'] and step % psoParams['adaptation_interval'] == 0 and step > 0:
                # Calculate exploration score based on recent performance
                # High variance in fitness = exploration, low variance = exploitation
                recent_fitness = returnData['fitnessHistory'][-psoParams['adaptation_interval']:]
                fitness_variance = np.var(recent_fitness) if len(recent_fitness) > 1 else 0

                # Calculate improvement rate
                improvement_rate = (returnData['fitnessHistory'][0] - recent_fitness[-1]) / max(
                    abs(returnData['fitnessHistory'][0]), 1e-10)
                improvement_rate = max(0, min(1, improvement_rate))  # Normalize between 0 and 1

                # Calculate exploration score based on recent improvement and current stage
                step_progress = step / psoParams['maxSteps']  # Progress through optimization (0 to 1)
                exploration_score = 0.7 * (1 - step_progress) + 0.3 * (1 - improvement_rate)

                # Determine optimal strategy
                old_strategy = current_strategy
                if exploration_score > psoParams['exploration_threshold']:
                    # High exploration score - focus on exploration
                    current_strategy = "exploration"

                    # Adjust parameters for exploration
                    psoParams['c1'] = 1.5  # Reduce cognitive parameter (personal best attraction)
                    psoParams['c2'] = 2.5  # Increase social parameter (global best attraction)
                    psoParams['w_start'] = 0.9  # Higher inertia for exploration
                    psoParams['w_end'] = 0.6  # Higher minimum inertia
                    psoParams['max_velocity'] = 0.6  # Higher velocity limit

                else:
                    # Low exploration score - focus on exploitation
                    current_strategy = "exploitation"

                    # Adjust parameters for exploitation
                    psoParams['c1'] = 2.5  # Increase cognitive parameter
                    psoParams['c2'] = 1.5  # Reduce social parameter
                    psoParams['w_start'] = 0.6  # Lower inertia for local search
                    psoParams['w_end'] = 0.2  # Lower minimum inertia
                    psoParams['max_velocity'] = 0.3  # Lower velocity limit for fine-tuning

                # Record strategy shift if changed
                if old_strategy != current_strategy:
                    returnData['strategy_shifts'].append({
                        'iteration': step,
                        'old_strategy': old_strategy,
                        'new_strategy': current_strategy,
                        'exploration_score': float(exploration_score),
                        'fitness_improvement': float(improvement_rate)
                    })
                    print(f"Strategy shifted to {current_strategy} at iteration {step}, score: {exploration_score:.3f}")

            # Update inertia weight - linear decrease
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

            # Dimension group phase management for balanced PSO
            if psoParams['use_dimension_groups'] and psoParams['dimension_groups']:
                # Check if we should switch to next phase
                if phase_counter >= phase_length:
                    phase_counter = 0
                    current_phase += 1

                    # Cycle through groups, with final phase using all dimensions
                    if current_phase < len(psoParams['dimension_groups']):
                        active_group = psoParams['dimension_groups'][current_phase]
                        print(f"Switching to dimension group: {active_group}")
                    else:
                        # Final phase: all dimensions active
                        active_group = list(range(nDim))
                        print("Final phase: all dimensions active")

                # Update dimension importance based on active group
                if active_group is not None:
                    # Reset importance weights
                    dimension_importance = cp.ones(nDim) * 0.1  # Low base importance

                    # Increase importance for active dimensions
                    for dim in active_group:
                        if 0 <= dim < nDim:
                            dimension_importance[dim] = 1.0

                    # Normalize
                    dimension_importance = dimension_importance / cp.sum(dimension_importance) * nDim

                phase_counter += 1

            # Update dimension weights for balanced PSO
            if psoParams['use_dimension_balance'] and step > 0 and step % 50 == 0:
                # Store current particles for analysis
                particles_history.append(particles.copy())

                # If enough history, analyze dimension exploration
                if len(particles_history) >= 3:
                    # Limit history to last 10 snapshots to save memory
                    if len(particles_history) > 10:
                        particles_history = particles_history[-10:]

                    # Analyze dimension exploration
                    dim_names = psoParams.get('dim_names', [f'dim_{i}' for i in range(nDim)])
                    dim_analysis = analyze_dimension_exploration(particles_history, dim_names)

                    # Update dimension weights based on exploration analysis
                    for i, name in enumerate(dim_names):
                        if name in dim_analysis:
                            # Low coverage or variance indicates need for more exploration
                            coverage = dim_analysis[name]['coverage']
                            variance = dim_analysis[name]['variance']

                            # Adjust weights: increase for dimensions with poor exploration
                            exploration_score = coverage * min(variance * 10, 1.0)
                            dimension_weights[i] = max(0.5, 1.5 - exploration_score)

                    # Apply importance weighting from group settings
                    dimension_weights = dimension_weights * dimension_importance

                    # Normalize weights
                    dimension_weights = dimension_weights / cp.sum(dimension_weights) * nDim

                    # Store dimension analysis
                    returnData['dimension_analysis'] = dim_analysis

            # 频繁进行速度重置以逃离局部最优
            if step > 0 and step % 30 == 0:
                # 为25%的粒子重置速度
                reset_indices = cp.random.choice(psoParams['popsize'], size=psoParams['popsize'] // 4, replace=False)
                for idx in reset_indices:
                    # For balanced PSO, vary speed resets by dimension importance
                    if psoParams['use_dimension_balance']:
                        for dim in range(nDim):
                            # More important dimensions get larger random resets
                            velocities[idx, dim] = cp.random.uniform(-0.4, 0.4) * dimension_weights[dim] / (nDim / 6)
                    else:
                        # Standard uniform reset
                        velocities[idx] = cp.random.uniform(-0.4, 0.4, nDim)

                # 定期改变惯性权重以增加搜索多样性
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

                # Update velocity based on current strategy
                if psoParams['use_adaptive_strategy'] and current_strategy == "exploitation":
                    # In exploitation mode, focus more on local search around personal best
                    if psoParams['use_dimension_balance']:
                        # Dimension-weighted exploitation velocity update
                        for dim in range(nDim):
                            # Apply dimension-specific weights with exploitation focus
                            velocities[i, dim] = (w * velocities[i, dim] +
                                                  psoParams['c1'] * r1[dim] * (pbest[i, dim] - particles[i, dim]) *
                                                  dimension_weights[dim] * 1.5 +
                                                  psoParams['c2'] * r2[dim] * (lbest[dim] - particles[i, dim]) *
                                                  dimension_weights[dim] * 0.8)
                    else:
                        # Standard exploitation velocity update
                        velocities[i] = (w * velocities[i] +
                                         psoParams['c1'] * r1 * (pbest[i] - particles[i]) * 1.5 +
                                         psoParams['c2'] * r2 * (lbest - particles[i]) * 0.8)
                else:
                    # Standard or exploration velocity update
                    if psoParams['use_dimension_balance']:
                        # Dimension-weighted velocity update
                        for dim in range(nDim):
                            # Apply dimension-specific weights to each component
                            velocities[i, dim] = (w * velocities[i, dim] +
                                                  psoParams['c1'] * r1[dim] * (pbest[i, dim] - particles[i, dim]) *
                                                  dimension_weights[dim] +
                                                  psoParams['c2'] * r2[dim] * (lbest[dim] - particles[i, dim]) *
                                                  dimension_weights[dim])
                    else:
                        # Standard velocity update
                        velocities[i] = (w * velocities[i] +
                                         psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                         psoParams['c2'] * r2 * (lbest - particles[i]))

                # 更保守的速度限制，防止过冲
                max_vel = psoParams['max_velocity'] * (1 - 0.3 * step / psoParams['maxSteps'])

                # Adjust velocity limits based on adaptive strategy
                if psoParams['use_adaptive_strategy']:
                    if current_strategy == "exploitation":
                        # Tighter velocity limits for exploitation phases
                        max_vel *= 0.7

                # For balanced PSO, apply dimension-specific velocity limits
                if psoParams['use_dimension_balance']:
                    for dim in range(nDim):
                        # Scale velocity limit by dimension weight
                        dim_vel_limit = max_vel * dimension_weights[dim] / (nDim / 6)
                        velocities[i, dim] = cp.clip(velocities[i, dim], -dim_vel_limit, dim_vel_limit)
                else:
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

                # Evaluate new position with current iteration for adaptive fitness
                new_fitness = fitfuncHandle(particles[i:i + 1], returnxVec=0, iteration=step)
                fitness[i] = new_fitness
                total_evals += 1

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    # For balanced PSO, track which dimensions changed significantly
                    if psoParams['use_dimension_balance']:
                        # Calculate dimension-wise changes
                        dim_changes = cp.abs(particles[i] - pbest[i])
                        # Update dimension progress (which dimensions are contributing to improvement)
                        significant_dims = dim_changes > 0.01  # Threshold for significant change
                        if cp.any(significant_dims):
                            dimension_progress[significant_dims] += 1
                            # Reset stagnation counter for active dimensions
                            dimension_stagnation[significant_dims] = 0

                    # Update personal best
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = new_fitness
                elif psoParams['use_dimension_balance']:
                    # Increment stagnation counter for dimensions that aren't improving
                    dimension_stagnation += 1

            # Update global best
            current_best_idx = cp.argmin(pbest_fitness)
            if pbest_fitness[current_best_idx] < gbest_fitness:
                gbest = pbest[current_best_idx].copy()
                gbest_fitness = pbest_fitness[current_best_idx].copy()

                # Update progress bar information
                pbar.set_postfix({'fitness': float(gbest_fitness)})

                # 重置改进计数（如果启用早停）
                if not psoParams['disable_early_stop']:
                    no_improvement_count = 0
                    prev_best_fitness = float(gbest_fitness)

            # Record best fitness at each step
            returnData['fitnessHistory'].append(float(gbest_fitness))

            # 早停逻辑（如果启用）
            if not psoParams['disable_early_stop']:
                current_best_fitness = float(gbest_fitness)
                fitness_improvement = abs(current_best_fitness - prev_best_fitness)

                if fitness_improvement < min_fitness_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    prev_best_fitness = current_best_fitness

                # 仅在极端情况下早停，且仅在完成90%迭代后
                if step > 0.9 * psoParams['maxSteps'] and no_improvement_count >= max_no_improvement:
                    print(
                        f"Run {psoParams['run']} stopped after {step + 1} iterations: No improvement for {no_improvement_count} iterations")
                    break

            # 更激进的粒子重初始化策略，打破局部最优
            if step > 0 and step % 40 == 0:
                # 查找最差的20%粒子
                worst_indices = cp.argsort(fitness)[-psoParams['popsize'] // 5:]

                # Reset logic for balanced PSO
                if psoParams['use_dimension_balance'] and psoParams['use_dimension_restart']:
                    # Identify stagnant dimensions (those with least progress)
                    if cp.sum(dimension_progress) > 0:
                        stagnant_dims = dimension_stagnation > 100  # Long period without improvement
                        if not cp.any(stagnant_dims):
                            # If no clearly stagnant dimensions, use relative progress
                            norm_progress = dimension_progress / cp.sum(dimension_progress)
                            stagnant_dims = norm_progress < (1.0 / nDim / 2)  # Less than half average progress

                        # For each worst particle, do targeted resets for stagnant dimensions
                        for idx in worst_indices:
                            # Standard random reset
                            particles[idx] = cp.random.rand(nDim)

                            # Different reset strategy based on adaptive phase
                            if psoParams['use_adaptive_strategy'] and current_strategy == "exploitation":
                                # For exploitation phase, reset around global best with small perturbations
                                for dim in range(nDim):
                                    particles[idx, dim] = gbest[dim] + cp.random.uniform(-0.2, 0.2)
                                particles[idx] = cp.clip(particles[idx], 0, 1)
                            else:
                                # More aggressive exploration for stagnant dimensions
                                for dim in range(nDim):
                                    if stagnant_dims[dim]:
                                        # Choose specific area of search space based on step
                                        if step % 120 < 40:  # Low region
                                            particles[idx, dim] = cp.random.uniform(0, 0.33)
                                        elif step % 120 < 80:  # Middle region
                                            particles[idx, dim] = cp.random.uniform(0.33, 0.67)
                                        else:  # High region
                                            particles[idx, dim] = cp.random.uniform(0.67, 1.0)

                            # Reset velocity with dimension-specific scale
                            for dim in range(nDim):
                                if stagnant_dims[dim]:
                                    # Stronger random velocity for stagnant dimensions
                                    velocities[idx, dim] = cp.random.uniform(-0.5, 0.5) * 1.5
                                else:
                                    # Normal random velocity for other dimensions
                                    velocities[idx, dim] = cp.random.uniform(-0.3, 0.3)

                            # Evaluate new position with current iteration for adaptive fitness
                            fitness[idx] = fitfuncHandle(particles[idx:idx + 1], returnxVec=0, iteration=step)
                            total_evals += 1
                else:
                    # Standard reset logic
                    for idx in worst_indices:
                        # 决定初始化策略
                        init_method = step % 5
                        if init_method == 0:
                            # 均匀随机
                            particles[idx] = cp.random.rand(nDim)
                        elif init_method == 1:
                            # 在全局最优附近添加噪声探索
                            particles[idx] = cp.clip(gbest + cp.random.normal(0, 0.3, nDim), 0, 1)
                        elif init_method == 2:
                            # 在边界附近探索
                            if cp.random.rand() < 0.5:  # 简单硬币翻转
                                particles[idx] = cp.random.uniform(0, 0.2, nDim)
                            else:
                                particles[idx] = cp.random.uniform(0.8, 1.0, nDim)
                        elif init_method == 3:
                            # 大幅度随机跳跃
                            particles[idx] = cp.clip(particles[idx] + cp.random.uniform(-0.7, 0.7, nDim), 0, 1)
                        else:
                            # 在参数空间内随机采样
                            for j in range(nDim):
                                particles[idx, j] = cp.random.uniform(0.1, 0.9)

                        # 使用更大范围的速度重置以提高探索能力
                        velocities[idx] = cp.random.uniform(-0.3, 0.3, nDim)

                        # 评估新位置 with current iteration for adaptive fitness
                        fitness[idx] = fitfuncHandle(particles[idx:idx + 1], returnxVec=0, iteration=step)
                        total_evals += 1

    # Final dimension analysis for balanced PSO
    if psoParams['use_dimension_balance'] and len(particles_history) > 0:
        # Add final particles to history
        particles_history.append(particles.copy())

        # Perform final dimension analysis
        dim_names = psoParams.get('dim_names', [f'dim_{i}' for i in range(nDim)])
        final_analysis = analyze_dimension_exploration(particles_history, dim_names)
        returnData['dimension_analysis'] = final_analysis

        # Create dimension balance report
        balance_report = {
            'dimension_weights': cp.asnumpy(dimension_weights).tolist() if isinstance(dimension_weights,
                                                                                      cp.ndarray) else dimension_weights,
            'dimension_progress': cp.asnumpy(dimension_progress).tolist() if isinstance(dimension_progress,
                                                                                        cp.ndarray) else dimension_progress,
            'dimension_stagnation': cp.asnumpy(dimension_stagnation).tolist() if isinstance(dimension_stagnation,
                                                                                            cp.ndarray) else dimension_stagnation
        }
        returnData['balance_report'] = balance_report

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
            # Choose standard fitness calculation
            use_bayesian = params.get('use_bayesian', False)
            if use_bayesian:
                fitVal[i] = bayesian_ssrqc(xVecReal[i], params)
            else:
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
    # Get the use_lensing flag based on previous classification
    use_lensing = params.get('use_lensing', False)  # Default to unlensed model

    # Generate signal based on whether to use lensing or not
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5], use_lensing=use_lensing)

    # Normalize signal
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # 使用纯信号作为参考模板（而不是带噪声的数据）
    dataY_templ = params.get('dataY_only_signal', params['dataY_only_signal'])

    # 使用内积计算作为匹配滤波的结果
    inPrd = innerprodpsd(dataY_templ, qc, params['sampFreq'], params['psdHigh'])

    # Return negative squared inner product (to minimize)
    return -cp.abs(inPrd) ** 2


def bayesian_ssrqc(x, params):
    """
    Bayesian fitness function incorporating physically motivated priors
    for GW parameters without using actual parameter values

    Parameters:
    -----------
    x : array
        Signal parameters (r, m_c, tc, phi_c, A, delta_t)
    params : dict
        Input parameters

    Returns:
    --------
    fitness : float
        Negative log posterior (likelihood * prior)
    """
    # Get the use_lensing flag based on previous classification
    use_lensing = params.get('use_lensing', False)  # Default to unlensed model

    # Generate signal based on whether to use lensing or not
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5], use_lensing=use_lensing)

    # Normalize signal
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # 使用纯信号作为参考模板（而不是带噪声的数据）
    dataY_templ = params.get('dataY_only_signal', params['dataY_only_signal'])

    # 使用内积计算作为匹配滤波的结果（对应于对数似然）
    inPrd = innerprodpsd(dataY_templ, qc, params['sampFreq'], params['psdHigh'])
    log_likelihood = cp.abs(inPrd) ** 2  # 似然对数值

    # 计算先验对数概率
    log_prior = calculate_log_prior(x, params)

    # 计算负的后验概率（最小化问题）
    # 后验 ∝ 似然 × 先验
    posterior = -log_likelihood - log_prior

    return posterior


def calculate_log_prior(x, params):
    """
    Calculate log prior probability for GW parameters
    using physically motivated priors (not based on actual parameter values)

    Parameters:
    -----------
    x : array
        Signal parameters (r, m_c, tc, phi_c, A, delta_t)
    params : dict
        Input parameters

    Returns:
    --------
    log_prior : float
        Log prior probability
    """
    # Extract parameters
    r, m_c, tc, phi_c, A, delta_t = x

    # Initialize log prior
    log_prior = 0.0

    # Distance prior (log-normal) - Cosmologically motivated
    r_real = 10 ** r  # Convert log10 to linear
    # Center around 1000 Mpc with broad distribution
    r_mean = 1000.0  # Average distance in Mpc
    r_sigma = 800.0  # Broad standard deviation to allow for wide range

    # Log-normal prior for distance
    if r_real > 0:
        log_r_prior = -0.5 * ((r_real - r_mean) / r_sigma) ** 2 - cp.log(r_sigma * cp.sqrt(2 * cp.pi))
        log_prior += log_r_prior

    # Chirp mass prior (log-normal) - Based on observed BH-BH mergers
    m_c_real = 10 ** m_c  # Convert log10 to linear
    m_c_mean = 30.0  # Typical chirp mass (solar masses)
    m_c_sigma = 15.0  # Broad standard deviation

    # Log-normal prior for chirp mass
    if m_c_real > 0:
        log_m_c_prior = -0.5 * ((m_c_real - m_c_mean) / m_c_sigma) ** 2 - cp.log(m_c_sigma * cp.sqrt(2 * cp.pi))
        log_prior += log_m_c_prior

    # Merger time prior (gaussian centered in observation window)
    # Default to center of observation window
    tc_mean = cp.mean(params['dataX'])
    tc_sigma = (cp.max(params['dataX']) - cp.min(params['dataX'])) / 4  # 1/4 the observation window

    # Gaussian prior for merger time
    log_tc_prior = -0.5 * ((tc - tc_mean) / tc_sigma) ** 2 - cp.log(tc_sigma * cp.sqrt(2 * cp.pi))
    log_prior += log_tc_prior

    # Phase prior - uniform over [0, 2π] (no contribution to log prior)

    # Flux ratio (A) prior - Physically motivated to favor smaller values
    # For lensed gravitational waves, flux ratio is typically < 1
    if A <= 1.0:
        # Higher probability for A <= 1
        log_prior += cp.log(2.0)  # Twice the probability compared to A > 1

    # Time delay prior - physically motivated for typical lensing
    # Favor delay times in physically plausible range
    if 0.2 <= delta_t <= 2.0:
        # Higher probability for reasonable delay times
        log_prior += 0.5  # Small bonus for reasonable delay times

    return log_prior


def adaptive_pso_fitness(xVec, params, returnxVec=0, iteration=0):
    """
    Adaptive fitness function that changes strategy based on optimization progress

    Parameters:
    -----------
    xVec : array
        Particle position
    params : dict
        Input parameters
    returnxVec : int
        Whether to return particle position
    iteration : int
        Current iteration number for adaptive strategy

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

    # Get maximum iterations (for phased approach)
    max_steps = params.get('maxSteps', 2000)

    # Calculate fitness for each valid point
    for i in range(nPoints):
        if validPts[i]:
            # Adaptive strategy based on iteration progress
            phase = iteration / max_steps

            if phase < 0.3:
                # Early phase: Broad exploration with simpler likelihood-based fitness
                # Use likelihood without priors for broader search
                fitVal[i] = ssrqc(xVecReal[i], params)
            elif phase < 0.7:
                # Middle phase: Transition to Bayesian approach with gradually increasing prior weights
                prior_weight = (phase - 0.3) / 0.4  # Linearly increase from 0 to 1

                # Calculate likelihood (standard fitness)
                likelihood_fitness = ssrqc(xVecReal[i], params)

                # Calculate full Bayesian posterior
                bayesian_fitness = bayesian_ssrqc(xVecReal[i], params)

                # Weighted average based on phase
                fitVal[i] = (1 - prior_weight) * likelihood_fitness + prior_weight * bayesian_fitness
            else:
                # Final phase: Full Bayesian approach with strong priors for precise convergence
                fitVal[i] = bayesian_ssrqc(xVecReal[i], params)

    if returnxVec:
        return fitVal, xVecReal
    else:
        return fitVal


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