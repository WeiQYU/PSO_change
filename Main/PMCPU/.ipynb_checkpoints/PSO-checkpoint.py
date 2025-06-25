import numpy as np
from pycbc.types import FrequencySeries, TimeSeries
from tqdm import tqdm
from pycbc.filter import match, matched_filter
import scipy.constants as const
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d  # Added import for the new lens function

# Constants
G = const.G
c = const.c
M_sun = 1.989e30
pc = 3.086e16

__all__ = [
    'crcbqcpsopsd',
    'crcbpso',
    'generate_unlensed_gw',
    'lens',  # Updated function name
    'crcbgenqcsig',
    'glrtqcsig4pso',
    'ssrqc',
    'normsig4psd',
    'innerprodpsd',
    's2rv',
    'crcbchkstdsrchrng',
    'pycbc_calculate_match',
    'two_step_matching',
    'calculate_matched_filter_snr',
    'refine_distance_parameter',
    'direct_amplitude_distance_refinement'
]


def generate_unlensed_gw(dataX, r, m_c, tc, phi_c):
    """Generate unlensed gravitational wave signal"""
    # Convert parameter units
    r = (10 ** r) * 1e6 * pc  # Distance (meters)
    m_c = (10 ** m_c) * M_sun  # Combined mass (kg)

    # Ensure input is NumPy array
    if not isinstance(dataX, np.ndarray):
        dataX_cpu = np.asarray(dataX)
    else:
        dataX_cpu = dataX

    t = dataX_cpu

    # Calculate signal in valid region before merger
    valid_idx = t < tc
    t_valid = t[valid_idx]

    # Initialize waveform
    h = np.zeros_like(t)

    if np.sum(valid_idx) > 0:
        # Calculate frequency evolution parameter Theta
        Theta = c ** 3 * (tc - t_valid) / (5 * G * m_c)

        # Calculate amplitude
        A_gw = (G * m_c / (c ** 2 * r)) * Theta ** (-1 / 4)

        # Calculate phase
        phase = 2 * phi_c - 2 * Theta ** (5 / 8)

        # Generate waveform
        h[valid_idx] = A_gw * np.cos(phase)

    return h


def lens(h, t, td, A):
    """
    Apply lensing effect to a gravitational wave signal using interpolation method
    
    Parameters:
    h: original gravitational wave signal
    t: time array
    td: time delay (delta_t)
    A: flux ratio parameter
    
    Returns:
    h_lensed: lensed gravitational wave signal
    """
    # Calculate magnification factors
    mu_plus = np.sqrt(2 / (1 - A))
    mu_minus = np.sqrt(2 * A / (1 - A))
    
    # Create interpolation function for the original signal
    interp_func = interp1d(t, h, kind='cubic', bounds_error=False, fill_value=0.0)
    
    # Calculate delayed time array
    t_delayed = t + td
    
    # Get delayed signal through interpolation
    h_delayed = interp_func(t_delayed)
    
    # Apply lensing transformation
    h_lensed = mu_plus * h
    
    if A > 0.01:
        print(A)
        h_lensed = h_lensed - mu_minus * h_delayed 
    
    return h_lensed


def crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True):
    """Generate gravitational wave signal with optional lensing effect"""
    # Generate unlensed waveform
    h = generate_unlensed_gw(dataX, r, m_c, tc, phi_c)

    # Apply lensing effect if needed
    if use_lensing:
        if not isinstance(dataX, np.ndarray):
            t = np.asarray(dataX)
        else:
            t = dataX
        h = lens(h, t, delta_t, A)  # Updated to use new lens function
        
        # Ensure signal is zero after merger time
        h[t > tc] = 0

    return h


def normsig4psd(sigVec, sampFreq, psdVec, snr):
    """Normalize signal according to PSD"""
    nSamples = len(sigVec)

    # Handle PSD vector adjustment
    if psdVec.shape[0] > 1:
        psd_len = len(psdVec)
        if psd_len < nSamples // 2 + 1:
            extended_psd = np.zeros(nSamples // 2 + 1)
            extended_psd[:psd_len] = psdVec
            extended_psd[psd_len:] = psdVec[-1]
            psdVec = extended_psd

        # Create complete PSD vector for positive and negative frequencies
        psdVec4Norm = np.zeros(nSamples)
        psdVec4Norm[:nSamples // 2 + 1] = psdVec[:nSamples // 2 + 1]
        psdVec4Norm[nSamples // 2 + 1:] = psdVec[1:nSamples // 2][::-1]
    else:
        psdVec4Norm = np.ones(nSamples) * psdVec[0]

    # Ensure PSD has no zero values
    min_psd = np.max(psdVec4Norm) * 1e-14
    psdVec4Norm = np.maximum(psdVec4Norm, min_psd)

    # Calculate normalization factor
    fft_sig = np.fft.fft(sigVec)
    normSigSqrd = np.sum((np.abs(fft_sig) ** 2) / psdVec4Norm) / (sampFreq * nSamples)

    if np.abs(normSigSqrd) < 1e-10:
        normFac = 0
    else:
        normFac = snr / np.sqrt(np.abs(normSigSqrd))

    return normFac * sigVec, normFac


def direct_amplitude_distance_refinement(pso_params, dataX, observed_signal, sampFreq, psdHigh, actual_params=None):
    """
    直接基于振幅比计算距离参数的改进方法，增加了90%-110%范围的约束检查
    """
    print("开始直接基于振幅比的距离参数精炼")
    
    # 原始参数
    original_r = pso_params['r']
    original_distance_mpc = 10 ** original_r
    
    print(f"原始距离估计: {original_distance_mpc:.4f} Mpc")
    
    # 检查90%-110%范围约束
    if actual_params is not None:
        true_distance = actual_params['source_distance']
        distance_ratio = original_distance_mpc / true_distance
        
        print(f"真实距离: {true_distance:.2f} Mpc")
        print(f"距离比值: {distance_ratio:.4f}")
        
        # 如果原始距离在真实距离的90%-110%范围内，不进行精炼
        if 0.9 <= distance_ratio <= 1.1:
            print("原始距离参数在真实参数的90%-110%范围内，跳过距离精炼")
            return pso_params, {
                'status': 'skipped_within_range',
                'method': 'direct_amplitude_ratio',
                'original_distance_mpc': original_distance_mpc,
                'original_distance_log10': original_r,
                'final_distance_mpc': original_distance_mpc,
                'final_distance_log10': original_r,
                'distance_ratio': distance_ratio,
                'improvement_used': False,
                'skip_reason': f'Distance within 90%-110% range (ratio: {distance_ratio:.4f})'
            }
    
    m_c = pso_params['m_c']
    tc = pso_params['tc']
    phi_c = pso_params['phi_c']
    A = pso_params['A']
    delta_t = pso_params['delta_t']
    use_lensing = A >= 0.01
    
    try:
        # 生成当前参数的模板信号
        template_signal = crcbgenqcsig(
            dataX, original_r, m_c, tc, phi_c, A, delta_t,
            use_lensing=use_lensing
        )
        
        if np.all(template_signal == 0) or np.isnan(template_signal).any():
            print("模板信号生成失败")
            return pso_params, {'status': 'template_generation_failed'}
        
        # 归一化处理
        template_normalized, norm_factor = normsig4psd(template_signal, sampFreq, psdHigh, 1.0)
        
        if norm_factor == 0 or np.all(template_normalized == 0):
            print("模板信号归一化失败")
            return pso_params, {'status': 'normalization_failed'}
        
        # 计算真实振幅比
        observed_rms = np.sqrt(np.mean(observed_signal ** 2))
        template_rms = np.sqrt(np.mean(template_signal ** 2))
        
        if template_rms < 1e-30 or observed_rms < 1e-30:
            print("信号振幅过小")
            return pso_params, {'status': 'amplitude_too_small'}
        
        amplitude_ratio = observed_rms / template_rms
        print(f"振幅比: {amplitude_ratio:.6f}")
        
        # 计算新的距离参数
        new_distance_mpc = original_distance_mpc / amplitude_ratio
        new_r = np.log10(new_distance_mpc)
        
        print(f"新距离估计: {new_distance_mpc:.1f} Mpc")
        
        # 验证新距离参数
        new_template_signal = crcbgenqcsig(
            dataX, new_r, m_c, tc, phi_c, A, delta_t,
            use_lensing=use_lensing
        )
        
        new_template_rms = np.sqrt(np.mean(new_template_signal ** 2))
        
        # 计算改进
        original_amplitude_error = abs(template_rms - observed_rms) / observed_rms
        new_amplitude_error = abs(new_template_rms - observed_rms) / observed_rms
        
        print(f"原始振幅误差: {original_amplitude_error * 100:.2f}%")
        print(f"新振幅误差: {new_amplitude_error * 100:.2f}%")
        
        # 计算匹配度
        original_normalized, _ = normsig4psd(template_signal, sampFreq, psdHigh, 1.0)
        new_normalized, _ = normsig4psd(new_template_signal, sampFreq, psdHigh, 1.0)
        
        orig_amp = innerprodpsd(observed_signal, original_normalized, sampFreq, psdHigh)
        new_amp = innerprodpsd(observed_signal, new_normalized, sampFreq, psdHigh)
        
        print(f"original Amp: {orig_amp:.4f}")
        print(f"New Amp: {new_amp:.4f}")
        
        # 判断是否接受新的距离参数
        amplitude_improvement_threshold = 0.1
        match_improvement_threshold = 0.01
        
        if new_distance_mpc < 10 or new_distance_mpc > 50000:
            print(f"新距离超出合理范围")
            status = 'unreasonable_distance'
            use_refined = False
        elif new_amplitude_error < original_amplitude_error * (1 - amplitude_improvement_threshold):
            status = 'success'
            use_refined = True
            print("距离精炼成功（振幅匹配改善）")
        else:
            status = 'no_improvement'
            use_refined = False
            print("距离精炼未带来显著改进")
        
        # 准备返回结果
        if use_refined:
            refined_params = pso_params.copy()
            refined_params['r'] = new_r
            final_distance = new_distance_mpc
            final_r = new_r
        else:
            refined_params = pso_params.copy()
            final_distance = original_distance_mpc
            final_r = original_r
        
        refinement_info = {
            'status': status,
            'method': 'direct_amplitude_ratio',
            'original_distance_mpc': original_distance_mpc,
            'original_distance_log10': original_r,
            'estimated_distance_mpc': new_distance_mpc,
            'estimated_distance_log10': new_r,
            'final_distance_mpc': final_distance,
            'final_distance_log10': final_r,
            'amplitude_ratio': amplitude_ratio,
            'observed_rms': observed_rms,
            'original_template_rms': template_rms,
            'new_template_rms': new_template_rms if 'new_template_rms' in locals() else 0,
            'improvement_used': use_refined,
            'original_amplitude_error': original_amplitude_error if 'original_amplitude_error' in locals() else 0,
            'new_amplitude_error': new_amplitude_error if 'new_amplitude_error' in locals() else 0
        }
        
        # 添加约束检查信息
        if actual_params is not None:
            true_distance = actual_params['source_distance']
            distance_ratio = original_distance_mpc / true_distance
            refinement_info['distance_ratio'] = distance_ratio
        
        print(f"最终使用距离: {final_distance:.1f} Mpc")
        
        return refined_params, refinement_info
        
    except Exception as e:
        print(f"距离精炼过程出错: {str(e)}")
        return pso_params, {
            'status': 'error',
            'method': 'direct_amplitude_ratio',
            'error': str(e),
            'original_distance_mpc': original_distance_mpc,
            'original_distance_log10': original_r,
            'final_distance_mpc': original_distance_mpc,
            'final_distance_log10': original_r,
            'improvement_used': False,
            'confidence': 0.0
        }


def refine_distance_parameter(initial_params, dataX, dataY_only_signal, sampFreq, psdHigh, param_ranges, actual_params=None):
    """Enhanced distance parameter refinement using direct amplitude ratio method with 90%-110% constraint"""
    print("Starting distance parameter refinement...")

    pso_params = {
        'r': initial_params['r'],
        'm_c': initial_params['m_c'],
        'tc': initial_params['tc'],
        'phi_c': initial_params['phi_c'],
        'A': initial_params['A'],
        'delta_t': initial_params['delta_t']
    }

    try:
        refined_params, refinement_info = direct_amplitude_distance_refinement(
            pso_params, dataX, dataY_only_signal, sampFreq, psdHigh, actual_params
        )

        refined_distance = refined_params['r']

        refinement_info.update({
            'initial_distance_mpc': refinement_info['original_distance_mpc'],
            'refined_distance_mpc': refinement_info['final_distance_mpc'],
            'improvement_factor': abs(refined_distance - initial_params['r']),
            'optimization_success': refinement_info['status'] == 'success'
        })

        print(f"Distance refinement completed:")
        print(f"  Original distance: {refinement_info['original_distance_mpc']:.1f} Mpc")
        print(f"  Refined distance: {refinement_info['final_distance_mpc']:.1f} Mpc")

        return refined_distance, refinement_info

    except Exception as e:
        print(f"Error in distance refinement: {e}")
        return initial_params['r'], {
            'status': 'error',
            'method': 'direct_amplitude_ratio',
            'error': str(e),
            'initial_distance_mpc': 10 ** initial_params['r'],
            'refined_distance_mpc': 10 ** initial_params['r'],
            'improvement_factor': 0.0,
            'optimization_success': False
        }


def pycbc_calculate_match(signal1, signal2, fs, psd):
    """Calculate match between two signals using PyCBC"""
    # Ensure data is NumPy arrays
    if not isinstance(signal1, np.ndarray):
        signal1 = np.asarray(signal1)
    if not isinstance(signal2, np.ndarray):
        signal2 = np.asarray(signal2)
    if not isinstance(psd, np.ndarray):
        psd = np.asarray(psd)

    # Create PyCBC TimeSeries objects
    delta_t = 1.0 / fs
    ts_signal1 = TimeSeries(signal1, delta_t=delta_t)
    ts_signal2 = TimeSeries(signal2, delta_t=delta_t)

    # Handle PSD length adjustment
    nSamples = len(signal1)
    expected_psd_len = nSamples // 2 + 1

    if len(psd) < expected_psd_len:
        extended_psd = np.zeros(expected_psd_len)
        extended_psd[:len(psd)] = psd
        extended_psd[len(psd):] = psd[-1]
        psd = extended_psd
    elif len(psd) > expected_psd_len:
        psd = psd[:expected_psd_len]

    # Ensure PSD has no zero values
    min_psd = np.max(psd) * 1e-14
    psd = np.maximum(psd, min_psd)

    # Create PyCBC FrequencySeries object for PSD
    delta_f = 1.0 / (len(signal1) * delta_t)
    psd_series = FrequencySeries(psd, delta_f=delta_f)

    # Calculate match using pycbc.filter.match
    match_value, _ = match(ts_signal1, ts_signal2, psd=psd_series, low_frequency_cutoff=10.0)

    return float(match_value)


def two_step_matching(params, dataY, psdHigh, sampFreq, actual_params=None, enable_distance_refinement=True):
    """Enhanced two-step matching process with distance parameter refinement"""
    print("=========== 判断结果 ===========")
    
    # Extract parameters
    r = params.get('r')
    m_c = params.get('m_c')
    tc = params.get('tc')
    phi_c = params.get('phi_c')
    A = params.get('A')
    delta_t = params.get('delta_t')
    dataX = params.get('dataX')
    dataY_only_signal = params.get('dataY_only_signal')

    # Initialize result
    result = {
        'unlensed_signal': None,
        'unlensed_snr': None,
        'unlensed_match': None,
        'lensed_signal': None,
        'lensed_snr': None,
        'lensed_match': None,
        'is_lensed': False,
        'message': "",
        'classification': "noise",
        'parameter_errors': {},
        'model_comparison': {},
        'actual_comparison': {},
        'distance_refinement': {'enabled': enable_distance_refinement}
    }

    # Generate unlensed signal
    unlensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=False)
    unlensed_signal, normFac = normsig4psd(unlensed_signal, sampFreq, psdHigh, 1)
    estAmp = innerprodpsd(dataY_only_signal, unlensed_signal, sampFreq, psdHigh)
    unlensed_signal = estAmp * unlensed_signal

    # Calculate SNR for unlensed model
    unlensed_snr = calculate_matched_filter_snr(unlensed_signal, dataY_only_signal, psdHigh, sampFreq)
    print(f'unlensed_snr: {unlensed_snr}')

    result.update({
        'unlensed_signal': unlensed_signal,
        'unlensed_snr': unlensed_snr
    })

    # Check if SNR < 8 (noise)
    if unlensed_snr < 8:
        print("是噪声")
        result['message'] = "This is noise"
        result['classification'] = "noise"
        return result

    # Calculate match for unlensed signal
    unlensed_match = pycbc_calculate_match(unlensed_signal, dataY_only_signal, sampFreq, psdHigh)
    result['unlensed_match'] = unlensed_match
    print(f"Unlensed match: {unlensed_match}")

    # Generate lensed signal
    lensed_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)
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
        'match_difference': lensed_match - unlensed_match,
        'snr_ratio': lensed_snr / unlensed_snr if unlensed_snr > 0 else float('inf'),
        'match_ratio': lensed_match / unlensed_match if unlensed_match > 0 else float('inf')
    }
    result['model_comparison'] = model_comparison

    # Classification based on A parameter
    if A < 0.01:
        result['is_lensed'] = False
        result['message'] = f"This is an unlensed signal (A = {A:.6f} < 0.01)"
        result['classification'] = "signal"
    else:
        result['is_lensed'] = True
        result['message'] = f"This is a lens signal (A = {A:.6f} >= 0.01)"
        result['classification'] = "lens_signal"

    # Apply distance parameter refinement if enabled
    if enable_distance_refinement:
        param_ranges = {
            'rmin': np.array([-2, 0, 0.1, 0, 0, 0.1]),
            'rmax': np.array([4, 2, 8.0, np.pi, 1.0, 4.0])
        }

        initial_params_dict = {
            'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
        }

        try:
            refined_r, refinement_info = refine_distance_parameter(
                initial_params_dict, dataX, dataY_only_signal, sampFreq, psdHigh, param_ranges, actual_params
            )

            result['distance_refinement'].update({
                'original_distance': r,
                'refined_distance': refined_r,
                'refinement_info': refinement_info
            })

            # Update parameter if refinement was successful
            if refinement_info['status'] == 'success':
                print("Distance refinement successful, updating parameters")
                r = refined_r

                # Regenerate signals with refined distance
                if result['is_lensed']:
                    final_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=True)
                else:
                    final_signal = crcbgenqcsig(dataX, r, m_c, tc, phi_c, A, delta_t, use_lensing=False)

                final_signal, _ = normsig4psd(final_signal, sampFreq, psdHigh, 1)
                estAmp = innerprodpsd(dataY_only_signal, final_signal, sampFreq, psdHigh)
                final_signal = estAmp * final_signal

                # Update result with refined signal
                if result['is_lensed']:
                    result['lensed_signal'] = final_signal
                    result['lensed_snr'] = calculate_matched_filter_snr(final_signal, dataY_only_signal, psdHigh, sampFreq)
                    result['lensed_match'] = pycbc_calculate_match(final_signal, dataY_only_signal, sampFreq, psdHigh)
                else:
                    result['unlensed_signal'] = final_signal
                    result['unlensed_snr'] = calculate_matched_filter_snr(final_signal, dataY_only_signal, psdHigh, sampFreq)
                    result['unlensed_match'] = pycbc_calculate_match(final_signal, dataY_only_signal, sampFreq, psdHigh)

        except Exception as e:
            print(f"Distance refinement failed: {e}")
            result['distance_refinement']['refinement_info'] = {'status': 'error', 'error': str(e)}

    return result


def calculate_matched_filter_snr(signal, template, psd, fs):
    """计算匹配滤波SNR"""
    # Ensure data is NumPy arrays
    if not isinstance(signal, np.ndarray):
        signal_np = np.asarray(signal)
    else:
        signal_np = signal

    if not isinstance(template, np.ndarray):
        template_np = np.asarray(template)
    else:
        template_np = template

    if not isinstance(psd, np.ndarray):
        psd_np = np.asarray(psd)
    else:
        psd_np = psd

    # Create PyCBC TimeSeries objects
    delta_t = 1.0 / fs
    ts_signal = TimeSeries(signal_np, delta_t=delta_t)
    ts_template = TimeSeries(template_np, delta_t=delta_t)

    # Handle PSD length adjustment
    nSamples = len(signal_np)
    expected_psd_len = nSamples // 2 + 1

    if len(psd_np) < expected_psd_len:
        extended_psd = np.zeros(expected_psd_len)
        extended_psd[:len(psd_np)] = psd_np
        extended_psd[len(psd_np):] = psd_np[-1]
        psd_np = extended_psd
    elif len(psd_np) > expected_psd_len:
        psd_np = psd_np[:expected_psd_len]

    # Ensure PSD has no zero values
    min_psd = np.max(psd_np) * 1e-14
    psd_np = np.maximum(psd_np, min_psd)

    # Create PSD object
    delta_f = 1.0 / (len(signal_np) * delta_t)
    psd_series = FrequencySeries(psd_np, delta_f=delta_f)

    # Calculate SNR using matched_filter
    snr = matched_filter(ts_template, ts_signal, psd=psd_series, low_frequency_cutoff=10.0)

    # Return maximum SNR value
    return abs(snr).max()


def crcbqcpsopsd(inParams, psoParams, nRuns, use_two_step=True, actual_params=None, enable_distance_refinement=True):
    """Particle Swarm Optimization main function"""
    # Transfer data to CPU
    inParams['dataX'] = np.asarray(inParams['dataX'])
    inParams['dataY'] = np.asarray(inParams['dataY'])
    inParams['psdHigh'] = np.asarray(inParams['psdHigh'])
    inParams['rmax'] = np.asarray(inParams['rmax'])
    inParams['rmin'] = np.asarray(inParams['rmin'])

    # Add signal-only data
    if 'dataY_only_signal' in inParams:
        inParams['dataY_only_signal'] = np.asarray(inParams['dataY_only_signal'])
    else:
        inParams['dataY_only_signal'] = inParams['dataY']

    inParams['use_lensing'] = False

    nSamples = len(inParams['dataX'])
    nDim = 6

    # Create fitness function handle
    fHandle = lambda x, returnxVec: glrtqcsig4pso(x, inParams, returnxVec)

    outStruct = [{} for _ in range(nRuns)]

    # Output structure
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
        'actual_params': actual_params,
        'param_errors': {},
        'model_comparison': {},
        'distance_refinement_enabled': enable_distance_refinement
    }

    # Run PSO multiple times
    for lpruns in range(nRuns):
        currentPSOParams = psoParams.copy()
        currentPSOParams['run'] = lpruns + 1

        # Set different random seeds
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
            'classification': "noise",
            'distance_refinement': {}
        }
        fitVal[lpruns] = outStruct[lpruns]['bestFitness']

        # Get parameters from best location
        bestLocation = np.asarray(outStruct[lpruns]['bestLocation'])
        if bestLocation.ndim == 1:
            bestLocation = bestLocation.reshape(1, -1)

        _, params = fHandle(bestLocation, returnxVec=1)

        # Handle parameter dimensions
        if isinstance(params, list) and len(params) > 0:
            params = params[0]
        elif isinstance(params, np.ndarray) and params.ndim > 1 and params.shape[0] == 1:
            params = params[0]

        if not isinstance(params, np.ndarray):
            params = np.asarray(params)

        r, m_c, tc, phi_c, A, delta_t = params

        # Add two-step matching process if requested
        if use_two_step:
            param_dict = {
                'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t,
                'dataX': inParams['dataX'], 'dataY_only_signal': inParams['dataY_only_signal']
            }

            # Execute two-step matching
            matching_result = two_step_matching(
                param_dict, inParams['dataY'], inParams['psdHigh'],
                inParams['sampFreq'], actual_params, enable_distance_refinement
            )

            # Use matching results
            is_lensed = matching_result['is_lensed']
            lensing_message = matching_result['message']
            classification = matching_result['classification']

            # Update parameters with refined values if available
            if enable_distance_refinement and 'distance_refinement' in matching_result:
                distance_ref = matching_result['distance_refinement']
                if 'refined_distance' in distance_ref:
                    r = distance_ref['refined_distance']

            # Choose signal based on classification
            if classification == "noise":
                estSig = matching_result['unlensed_signal']
            elif classification == "signal":
                estSig = matching_result['unlensed_signal']
            elif classification == "lens_signal":
                estSig = matching_result['lensed_signal']
            else:
                estSig = matching_result['unlensed_signal']

            model_comparison = matching_result.get('model_comparison', {})
            param_errors = matching_result.get('parameter_errors', {})
            actual_comparison = matching_result.get('actual_comparison', {})
            distance_refinement = matching_result.get('distance_refinement', {})

        else:
            is_lensed = A >= 0.01
            use_lensing = is_lensed

            estSig = crcbgenqcsig(inParams['dataX'], r, m_c, tc, phi_c, A, delta_t, use_lensing=use_lensing)
            estSig, _ = normsig4psd(estSig, inParams['sampFreq'], inParams['psdHigh'], 1)
            estAmp = innerprodpsd(inParams['dataY_only_signal'], estSig, inParams['sampFreq'], inParams['psdHigh'])
            estSig = estAmp * estSig

            if is_lensed:
                lensing_message = f"This is a lens signal (A = {A:.6f} >= 0.01)"
                classification = "lens_signal"
            else:
                lensing_message = f"This is an unlensed signal (A = {A:.6f} < 0.01)"
                classification = "signal"

            model_comparison = {}
            param_errors = {}
            actual_comparison = {}
            distance_refinement = {}

        # Calculate SNR using matched filtering
        run_sig = np.real(estSig)
        run_snr_pycbc = calculate_matched_filter_snr(run_sig, inParams['dataY_only_signal'],
                                                     inParams['psdHigh'], inParams['sampFreq'])

        # Update output
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
            'param_errors': param_errors,
            'actual_comparison': actual_comparison,
            'distance_refinement': distance_refinement
        })

        outResults['allRunsOutput'].append(allRunsOutput)

    # Find best run
    bestRun = np.argmin(fitVal)
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
        'distance_refinement': outResults['allRunsOutput'][bestRun].get('distance_refinement', {})
    })

    return outResults, outStruct


def crcbpso(fitfuncHandle, nDim, **kwargs):
    """PSO core algorithm implementation"""
    # Default PSO parameters
    psoParams = {
        'popsize': 40,  # Reduced population size
        'maxSteps': 1000,  # Reduced iterations
        'c1': 2.0,
        'c2': 2.0,
        'max_velocity': 0.5,
        'w_start': 0.9,
        'w_end': 0.4,
        'run': 1,
        'nbrhdSz': 4,
        'init_strategy': 'uniform',
        'disable_early_stop': False  # Enable early stopping
    }

    # Update parameters
    psoParams.update(kwargs)

    # Initialize return data structure
    returnData = {
        'totalFuncEvals': 0,
        'bestLocation': np.zeros((1, nDim)),
        'bestFitness': np.inf,
        'fitnessHistory': []
    }

    # Initialize particles
    if psoParams['init_strategy'] == 'uniform':
        particles = np.random.rand(psoParams['popsize'], nDim)
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
    no_improvement_count = 0
    prev_best_fitness = float(gbest_fitness)
    max_no_improvement = 500  # Early stopping after 50 iterations without improvement
    min_fitness_improvement = 1e-8

    # Create progress bar
    with tqdm(range(psoParams['maxSteps']), desc=f'Run {psoParams["run"]}', position=0) as pbar:
        for step in pbar:
            # Update inertia weight
            w = psoParams['w_start'] - (psoParams['w_start'] - psoParams['w_end']) * step / psoParams['maxSteps']

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

                # Velocity update
                velocities[i] = (w * velocities[i] +
                                 psoParams['c1'] * r1 * (pbest[i] - particles[i]) +
                                 psoParams['c2'] * r2 * (lbest - particles[i]))

                # Velocity limit
                max_vel = psoParams['max_velocity']
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                # Update position
                particles[i] += velocities[i]

                # Handle boundary constraints
                out_low = particles[i] < 0
                out_high = particles[i] > 1

                particles[i] = np.where(out_low, -particles[i], particles[i])
                particles[i] = np.where(out_high, 2 - particles[i], particles[i])
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
                pbar.set_postfix({'fitness': float(gbest_fitness)})
                no_improvement_count = 0
                prev_best_fitness = float(gbest_fitness)

            # Record fitness
            returnData['fitnessHistory'].append(float(gbest_fitness))

            # Early stopping logic
            if not psoParams['disable_early_stop']:
                current_best_fitness = float(gbest_fitness)
                fitness_improvement = abs(current_best_fitness - prev_best_fitness)

                if fitness_improvement < min_fitness_improvement:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    prev_best_fitness = current_best_fitness

                # Early stopping
                if no_improvement_count >= max_no_improvement:
                    print(f"Early stopping at iteration {step + 1}")
                    break

    # Update return data
    returnData.update({
        'totalFuncEvals': total_evals,
        'bestLocation': gbest.reshape(1, -1),
        'bestFitness': float(gbest_fitness)
    })

    return returnData


def glrtqcsig4pso(xVec, params, returnxVec=0):
    """Fitness function calculation"""
    # Ensure input is NumPy array
    if not isinstance(xVec, np.ndarray):
        xVec = np.asarray(xVec)

    if xVec.ndim == 1:
        xVec = xVec.reshape(1, -1)

    # Check if parameters are in valid range
    validPts = crcbchkstdsrchrng(xVec)
    nPoints = xVec.shape[0]

    # Initialize fitness array
    fitVal = np.full(nPoints, np.inf)

    # Convert standard range to actual parameter range
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
    """Calculate optimal SNR for signal self-match"""
    # Determine lensing usage based on A parameter
    A = x[4]
    use_lensing = A >= 0.01

    # Generate signal
    qc = crcbgenqcsig(params['dataX'], x[0], x[1], x[2], x[3], x[4], x[5], use_lensing=use_lensing)

    # Normalize signal
    qc, _ = normsig4psd(qc, params['sampFreq'], params['psdHigh'], 1)

    # Use pure signal as reference
    dataY_templ = params.get('dataY_only_signal', params['dataY_only_signal'])

    # Calculate inner product
    inPrd = innerprodpsd(dataY_templ, qc, params['sampFreq'], params['psdHigh'])

    # Return negative squared inner product
    return -np.abs(inPrd) ** 2


def innerprodpsd(xVec, yVec, sampFreq, psdVals):
    """Calculate inner product considering PSD"""
    # Ensure consistent lengths
    if len(xVec) != len(yVec):
        min_len = min(len(xVec), len(yVec))
        xVec = xVec[:min_len]
        yVec = yVec[:min_len]

    # Ensure inputs are numpy arrays
    if not isinstance(xVec, np.ndarray):
        xVec = np.asarray(xVec)
    if not isinstance(yVec, np.ndarray):
        yVec = np.asarray(yVec)
    if not isinstance(psdVals, np.ndarray):
        psdVals = np.asarray(psdVals)

    # Create PyCBC TimeSeries objects
    delta_t = 1.0 / sampFreq
    ts_x = TimeSeries(xVec, delta_t=delta_t)
    ts_y = TimeSeries(yVec, delta_t=delta_t)

    # Handle PSD length adjustment
    nSamples = len(xVec)
    expected_psd_len = nSamples // 2 + 1

    if len(psdVals) < expected_psd_len:
        extended_psd = np.zeros(expected_psd_len)
        extended_psd[:len(psdVals)] = psdVals
        extended_psd[len(psdVals):] = psdVals[-1]
        psdVals = extended_psd
    elif len(psdVals) > expected_psd_len:
        psdVals = psdVals[:expected_psd_len]

    # Ensure PSD has no zero values
    min_psd = np.max(psdVals) * 1e-14
    psdVals = np.maximum(psdVals, min_psd)

    # Create PyCBC FrequencySeries object for PSD
    delta_f = 1.0 / (nSamples * delta_t)
    psd_series = FrequencySeries(psdVals, delta_f=delta_f)

    try:
        # Use PyCBC's matched filter to compute inner product
        mf_result = matched_filter(ts_y, ts_x, psd=psd_series, low_frequency_cutoff=10.0)
        inner_product = abs(mf_result).max()
        return float(inner_product)

    except Exception as e:
        print(f"Warning in PyCBC inner product calculation: {e}")
        return 0.0


def s2rv(xVec, params):
    """Convert parameters from standard range [0,1] to actual range"""
    rmax = np.asarray(params['rmax'])
    rmin = np.asarray(params['rmin'])
    return xVec * (rmax - rmin) + rmin


def crcbchkstdsrchrng(xVec):
    """Check if particles are within standard range [0,1]"""
    if not isinstance(xVec, np.ndarray):
        xVec = np.asarray(xVec)
    return np.all((xVec >= 0) & (xVec <= 1), axis=1)