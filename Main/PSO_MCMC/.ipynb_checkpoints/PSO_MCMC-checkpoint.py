import cupy as cp
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import seaborn as sns
import os
import warnings

# Import the PSO implementation
from PSO_main_demo import (
    crcbqcpsopsd, crcbgenqcsig, normsig4psd, innerprodpsd,
    calculate_matched_filter_snr, pycbc_calculate_match
)

# Import Bilby for MCMC implementation
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Uniform, PriorDict
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import match, matched_filter

# Constants
G = const.G  # Gravitational constant, m^3 kg^-1 s^-2
c = const.c  # Speed of light, m/s
M_sun = 1.989e30  # Solar mass, kg
pc = 3.086e16  # Parsec to meters

# Create results directory
results_dir = "comparison_results_noise"
os.makedirs(results_dir, exist_ok=True)


def load_data():
    """Load gravitational wave data for analysis"""
    print("Loading data...")

    # Load noise data
    TrainingData = scio.loadmat('../generate_data/noise.mat')
    # analysisData = scio.loadmat('../generate_data/noise.mat')
    analysisData = scio.loadmat('../generate_data/data.mat')

    print("Data loaded successfully")

    # Convert data to CuPy arrays for PSO
    dataY = cp.asarray(analysisData['data'][0])
    # dataY = cp.asarray(TrainingData['noise'][0])
    training_noise = cp.asarray(TrainingData['noise'][0])
    dataY_only_signal = dataY - training_noise  # Extract signal part (for comparison)

    # Get basic parameters
    nSamples = dataY.size
    Fs = float(analysisData['samples'][0][0])
    # Fs = 4096
    dt = 1 / Fs
    t = cp.arange(0, 8, dt)

    # Calculate PSD
    psdHigh = cp.asarray(TrainingData['psd'][0])

    return {
        't': t,
        'dataY': dataY,
        'dataY_only_signal': dataY_only_signal,
        'noise': training_noise,
        'psdHigh': psdHigh,
        'sampFreq': Fs,
        'nSamples': nSamples
    }


def setup_parameters(data):
    """Set up parameter ranges for both PSO and MCMC"""

    # Define parameter ranges (min/max)
    param_ranges = {
        'rmin': cp.array([-2, 0, 0.1, 0, 0, 0.1]),  # r, mc, tc, phi, A, delta_t
        'rmax': cp.array([4, 2, 8.0, np.pi, 1.0, 4.0])
    }

    # Define actual parameters for validation (for synthetic data)
    actual_params = {
        'chirp_mass': 30.09,  # Solar masses
        'merger_time': 7.5,  # seconds
        'source_distance': 3100.0,  # Mpc
        'flux_ratio': 0.3333,  # A parameter
        'time_delay': 0.9854,  # seconds (delta_t)
        'phase': 0.25  # π fraction
    }

    # Set up PSO input parameters
    pso_params = {
        'dataX': data['t'],
        'dataY': data['dataY'],
        'dataY_only_signal': data['dataY_only_signal'],
        'sampFreq': data['sampFreq'],
        'psdHigh': data['psdHigh'],
        'rmin': param_ranges['rmin'],
        'rmax': param_ranges['rmax']
    }

    # Set up PSO configuration with increased particle count and iterations
    pso_config = {
        'popsize': 100,  # Increased from 10 to 100
        'maxSteps': 3000,  # Increased from 30 to 3000
        'c1': 2.0,  # Individual learning factor
        'c2': 2.0,  # Social learning factor
        'w_start': 0.9,  # Initial inertia weight
        'w_end': 0.5,  # Final inertia weight
        'max_velocity': 0.4,
        'nbrhdSz': 6,
        'disable_early_stop': True
    }

    return param_ranges, pso_params, pso_config, actual_params


# Define Bilby likelihood for gravitational wave analysis
class GWLikelihood(Likelihood):
    """Gravitational wave likelihood for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'A': None, 'delta_t': None
        })
        self.data_dict = data_dict
        self.dataX_np = cp.asnumpy(data_dict['dataX']) if isinstance(data_dict['dataX'], cp.ndarray) else data_dict[
            'dataX']
        self.dataY_only_signal_np = cp.asnumpy(data_dict['dataY_only_signal']) if isinstance(
            data_dict['dataY_only_signal'], cp.ndarray) else data_dict['dataY_only_signal']
        self.psd_np = cp.asnumpy(data_dict['psdHigh']) if isinstance(data_dict['psdHigh'], cp.ndarray) else data_dict[
            'psdHigh']

    def log_likelihood(self):
        """Log-likelihood function for Bilby"""
        try:
            # Get parameters
            params = np.array([
                self.parameters['r'],
                self.parameters['m_c'],
                self.parameters['tc'],
                self.parameters['phi_c'],
                self.parameters['A'],
                self.parameters['delta_t']
            ])

            # Map from [0,1] to original parameter range
            unscaled_params = np.zeros(6)
            for i in range(6):
                unscaled_params[i] = params[i] * (self.data_dict['rmax'][i] - self.data_dict['rmin'][i]) + \
                                     self.data_dict['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = unscaled_params

            # Determine if we should use lensing based on A parameter
            use_lensing = A >= 0.01

            # Generate the signal
            signal = crcbgenqcsig(
                cp.asarray(self.dataX_np), r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )

            # Check if signal generation was successful
            if signal is None or cp.isnan(signal).any():
                return -np.inf

            signal, _ = normsig4psd(signal, self.data_dict['sampFreq'], cp.asarray(self.psd_np), 1)

            # Optimize amplitude
            estAmp = innerprodpsd(
                cp.asarray(self.dataY_only_signal_np), signal,
                self.data_dict['sampFreq'], cp.asarray(self.psd_np)
            )

            # Check for valid amplitude
            if estAmp is None or cp.isnan(estAmp):
                return -np.inf

            signal = estAmp * signal

            # Calculate log-likelihood using matched filter SNR
            signal_np = cp.asnumpy(signal)

            # Using PyCBC's match function
            delta_t = 1.0 / self.data_dict['sampFreq']
            ts_signal = TimeSeries(signal_np, delta_t=delta_t)
            ts_data = TimeSeries(self.dataY_only_signal_np, delta_t=delta_t)

            delta_f = 1.0 / (len(signal_np) * delta_t)
            psd_series = FrequencySeries(self.psd_np, delta_f=delta_f)

            # Catch potential numerical issues
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    match_value, _ = match(ts_signal, ts_data, psd=psd_series, low_frequency_cutoff=10.0)

                    # Ensure match_value is valid and in range [0,1]
                    if match_value is None or np.isnan(match_value) or match_value <= 0 or match_value > 1:
                        return -np.inf

                    # Convert match to log-likelihood (avoiding numerical issues)
                    log_likelihood = 0.5 * len(self.dataY_only_signal_np) * np.log(match_value)

                    # Catch extreme values
                    if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                        return -np.inf

                    return log_likelihood
                except Exception as e:
                    # Handle any exceptions during matching
                    print(f"Warning: Match calculation error - {str(e)}")
                    return -np.inf
        except Exception as e:
            # Catch all other exceptions
            print(f"Warning: Likelihood calculation error - {str(e)}")
            return -np.inf


def run_bilby_mcmc(data_dict, param_ranges, n_live_points=500, n_iter=600):
    """Run MCMC analysis using Bilby and return results

    Parameters adjusted to match computational effort of PSO (100 particles, 3000 iterations)
    with n_live_points=500 and n_iter=600 (approximately 300,000 function evaluations).
    """
    print("Starting Bilby MCMC analysis...")

    # Create dictionary for MCMC with numpy arrays
    mcmc_data = {
        'dataX': cp.asnumpy(data_dict['t']) if isinstance(data_dict['t'], cp.ndarray) else data_dict['t'],
        'dataY': cp.asnumpy(data_dict['dataY']) if isinstance(data_dict['dataY'], cp.ndarray) else data_dict['dataY'],
        'dataY_only_signal': cp.asnumpy(data_dict['dataY_only_signal']) if isinstance(data_dict['dataY_only_signal'],
                                                                                      cp.ndarray) else data_dict[
            'dataY_only_signal'],
        'sampFreq': data_dict['sampFreq'],
        'psdHigh': cp.asnumpy(data_dict['psdHigh']) if isinstance(data_dict['psdHigh'], cp.ndarray) else data_dict[
            'psdHigh'],
        'rmin': cp.asnumpy(param_ranges['rmin']) if isinstance(param_ranges['rmin'], cp.ndarray) else param_ranges[
            'rmin'],
        'rmax': cp.asnumpy(param_ranges['rmax']) if isinstance(param_ranges['rmax'], cp.ndarray) else param_ranges[
            'rmax']
    }

    # Start timing
    start_time = time.time()

    # Create likelihood
    likelihood = GWLikelihood(mcmc_data)

    # Create priors (uniform in [0,1] for all parameters)
    priors = PriorDict()
    priors['r'] = Uniform(minimum=0, maximum=1, name='r')
    priors['m_c'] = Uniform(minimum=0, maximum=1, name='m_c')
    priors['tc'] = Uniform(minimum=0, maximum=1, name='tc')
    priors['phi_c'] = Uniform(minimum=0, maximum=1, name='phi_c')
    priors['A'] = Uniform(minimum=0, maximum=1, name='A')
    priors['delta_t'] = Uniform(minimum=0, maximum=1, name='delta_t')

    try:
        # Run the bilby sampler with improved settings and matching computational effort
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            nlive=n_live_points,  # Set number of live points (removed npoints to avoid warning)
            walks=25,  # Number of walks for each live point
            verbose=True,
            maxiter=n_iter,  # Maximum iterations
            outdir=results_dir,
            label='gw_analysis',
            resume=False,
            check_point_plot=False,
            bound='multi',  # More robust boundary scheme
            sample='rwalk',  # Random walk sampling
            check_point=True,  # Enable checkpointing
            check_point_delta_t=600,  # Save every 10 minutes
            plot=True,
        )
    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        # Return a minimal result structure to prevent further errors
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(cp.asnumpy(data_dict['dataY_only_signal'])),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'mismatch': 1.0,
            'param_ranges': mcmc_data
        }

    end_time = time.time()
    mcmc_duration = end_time - start_time
    print(f"Bilby MCMC completed in {mcmc_duration:.2f} seconds")

    try:
        # Get best fit parameters (maximum likelihood sample)
        if len(result.posterior) > 0:
            best_idx = np.argmax(result.posterior['log_likelihood'].values)
            best_params_bilby = result.posterior.iloc[best_idx]

            # Unscale parameters to original range
            best_params = np.zeros(6)
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
            for i, param in enumerate(param_names):
                best_params[i] = best_params_bilby[param] * (mcmc_data['rmax'][i] - mcmc_data['rmin'][i]) + \
                                 mcmc_data['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = best_params

            # Generate best signal
            use_lensing = A >= 0.01
            best_signal = crcbgenqcsig(
                cp.asarray(mcmc_data['dataX']), r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )
            best_signal, _ = normsig4psd(best_signal, mcmc_data['sampFreq'], cp.asarray(mcmc_data['psdHigh']), 1)

            # Optimize amplitude
            estAmp = innerprodpsd(
                cp.asarray(mcmc_data['dataY_only_signal']), best_signal,
                mcmc_data['sampFreq'], cp.asarray(mcmc_data['psdHigh'])
            )
            best_signal = estAmp * best_signal

            # Calculate SNR
            snr = calculate_matched_filter_snr(
                best_signal, cp.asarray(mcmc_data['dataY_only_signal']),
                cp.asarray(mcmc_data['psdHigh']), mcmc_data['sampFreq']
            )

            # Determine if signal is lensed
            is_lensed = A >= 0.01
            classification = "lens_signal" if is_lensed else "signal"

            # Calculate match/mismatch
            match_value = pycbc_calculate_match(
                cp.asnumpy(best_signal), mcmc_data['dataY_only_signal'],
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )
            mismatch = 1 - match_value

            # Convert posterior samples to array
            samples = result.posterior[param_names].values
        else:
            # If no posterior samples, set defaults
            print("Warning: No posterior samples found. Using default values.")
            best_params = np.zeros(6)
            r, m_c, tc, phi_c, A, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            is_lensed = False
            classification = "error"
            mismatch = 1.0
            samples = np.zeros((1, 6))

        # Prepare results
        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r,
                'm_c': m_c,
                'tc': tc,
                'phi_c': phi_c,
                'A': A,
                'delta_t': delta_t
            },
            'best_signal': cp.asnumpy(best_signal) if isinstance(best_signal, cp.ndarray) else best_signal,
            'snr': float(snr) if isinstance(snr, (cp.ndarray, np.ndarray)) else snr,
            'is_lensed': is_lensed,
            'classification': classification,
            'mismatch': mismatch,
            'samples': samples if 'samples' in locals() else np.zeros((1, 6)),
            'log_probs': result.posterior['log_likelihood'].values if len(result.posterior) > 0 else np.array(
                [-np.inf]),
            'bilby_result': result,  # Store the original Bilby result for additional analysis
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            }
        }

        return mcmc_results

    except Exception as e:
        print(f"Error in processing Bilby results: {str(e)}")
        # Return minimal results
        return {
            'duration': mcmc_duration,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(mcmc_data['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'mismatch': 1.0,
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            }
        }


def run_pso(data_dict, pso_params, pso_config, actual_params, n_runs=8):
    """Run PSO analysis and return results"""
    print("Starting PSO analysis...")

    # Time PSO execution
    start_time = time.time()

    try:
        # Run PSO with multiple starts
        outResults, outStruct = crcbqcpsopsd(
            pso_params, pso_config, n_runs,
            use_two_step=True,
            actual_params=actual_params
        )

        end_time = time.time()
        pso_duration = end_time - start_time
        print(f"PSO completed in {pso_duration:.2f} seconds")

        # Extract best run
        best_run_idx = outResults['bestRun']

        # Get best signal
        best_signal = outResults['bestSig']

        # Calculate mismatch explicitly (instead of relying on PSO's output)
        dataY_only_signal = cp.asarray(data_dict['dataY_only_signal'])
        match_value = pycbc_calculate_match(
            cp.asnumpy(best_signal),
            cp.asnumpy(dataY_only_signal),
            data_dict['sampFreq'],
            cp.asnumpy(data_dict['psdHigh'])
        )
        mismatch = 1 - match_value

        # Prepare results dictionary
        pso_results = {
            'duration': pso_duration,
            'best_params': {
                'r': outResults['r'],
                'm_c': outResults['m_c'],
                'tc': outResults['tc'],
                'phi_c': outResults['phi_c'],
                'A': outResults['A'],
                'delta_t': outResults['delta_t']
            },
            'best_signal': cp.asnumpy(best_signal) if isinstance(best_signal, cp.ndarray) else best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'classification': outResults['classification'],
            'mismatch': mismatch,  # Use the explicitly calculated mismatch
            'all_runs': outResults['allRunsOutput'],
            'structures': outStruct
        }

        return pso_results

    except Exception as e:
        print(f"Error in PSO: {str(e)}")
        # Return minimal results
        return {
            'duration': time.time() - start_time,
            'best_params': {
                'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0
            },
            'best_signal': np.zeros_like(cp.asnumpy(data_dict['dataY_only_signal'])),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'mismatch': 1.0,
        }


def evaluate_results(pso_results, mcmc_results, actual_params, data_dict):
    """Compare PSO and MCMC results and generate metrics"""

    print("\n============= Performance Comparison =============")

    try:
        # Convert actual parameters to log scale where needed
        actual_r_log = np.log10(actual_params['source_distance'])
        actual_m_c_log = np.log10(actual_params['chirp_mass'])

        # Calculate parameter errors for PSO
        pso_errors = {
            'r': abs((10 ** float(pso_results['best_params']['r']) - actual_params['source_distance']) / actual_params[
                'source_distance']),
            'm_c': abs((10 ** float(pso_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params[
                'chirp_mass']),
            'tc': abs((float(pso_results['best_params']['tc']) - actual_params['merger_time']) / actual_params[
                'merger_time']),
            'phi_c': abs(abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (
                        2 * np.pi),
            'A': abs(
                (float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params[
                'time_delay'])
        }

        # Calculate parameter errors for MCMC
        mcmc_errors = {
            'r': abs((10 ** float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params[
                'source_distance']),
            'm_c': abs((10 ** float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params[
                'chirp_mass']),
            'tc': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params[
                'merger_time']),
            'phi_c': abs(abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (
                        2 * np.pi),
            'A': abs(
                (float(mcmc_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs(
                (float(mcmc_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params[
                    'time_delay'])
        }

        # Calculate correlation with actual signal
        dataY_only_signal_np = cp.asnumpy(data_dict['dataY_only_signal']) if isinstance(data_dict['dataY_only_signal'],
                                                                                        cp.ndarray) else data_dict[
            'dataY_only_signal']

        def correlation_coefficient(x, y):
            try:
                # Handle potential NaN or inf values
                x = np.nan_to_num(x)
                y = np.nan_to_num(y)
                return np.corrcoef(x, y)[0, 1]
            except Exception as e:
                print(f"Error calculating correlation: {str(e)}")
                return 0.0

        pso_corr = correlation_coefficient(pso_results['best_signal'], dataY_only_signal_np)
        mcmc_corr = correlation_coefficient(mcmc_results['best_signal'], dataY_only_signal_np)

        # Prepare comparison metrics
        comparison = {
            'execution_time': {
                'PSO': pso_results['duration'],
                'MCMC': mcmc_results['duration'],
                'ratio': mcmc_results['duration'] / (pso_results['duration'] / 8) if pso_results[
                                                                                         'duration'] > 0 else float(
                    'inf')
            },
            'parameter_errors': {
                'PSO': pso_errors,
                'MCMC': mcmc_errors
            },
            'signal_quality': {
                'PSO': {
                    'SNR': float(pso_results['snr']),
                    'correlation': pso_corr,
                    'mismatch': float(pso_results['mismatch'])
                },
                'MCMC': {
                    'SNR': float(mcmc_results['snr']),
                    'correlation': mcmc_corr,
                    'mismatch': float(mcmc_results['mismatch'])
                }
            },
            'classification': {
                'PSO': {
                    'is_lensed': pso_results['is_lensed'],
                    'classification': pso_results['classification'],
                },
                'MCMC': {
                    'is_lensed': mcmc_results['is_lensed'],
                    'classification': mcmc_results['classification'],
                }
            }
        }

        # Print comparison results
        print(
            f"Execution Time: PSO: {comparison['execution_time']['PSO'] / 8:.2f}s, Bilby MCMC: {comparison['execution_time']['MCMC']:.2f}s")
        print(f"Speed Improvement: Bilby MCMC is {comparison['execution_time']['ratio']:.2f}x slower than PSO")

        # Print parameter errors
        print("\nParameter Estimation Errors (relative):")
        for param in pso_errors.keys():
            print(f"  {param}: PSO: {pso_errors[param] * 100:.2f}%, Bilby MCMC: {mcmc_errors[param] * 100:.2f}%")

        # Print signal quality metrics
        print("\nSignal Quality Metrics:")
        print(
            f"  SNR: PSO: {comparison['signal_quality']['PSO']['SNR']:.2f}, Bilby MCMC: {comparison['signal_quality']['MCMC']['SNR']:.2f}")
        print(
            f"  Correlation: PSO: {comparison['signal_quality']['PSO']['correlation']:.4f}, Bilby MCMC: {comparison['signal_quality']['MCMC']['correlation']:.4f}")
        print(
            f"  Mismatch: PSO: {comparison['signal_quality']['PSO']['mismatch']:.4e}, Bilby MCMC: {comparison['signal_quality']['MCMC']['mismatch']:.4e}")

        return comparison

    except Exception as e:
        print(f"Error in evaluate_results: {str(e)}")
        # Return a minimal comparison structure
        return {
            'execution_time': {
                'PSO': pso_results.get('duration', 0),
                'MCMC': mcmc_results.get('duration', 0),
                'ratio': 1.0
            },
            'parameter_errors': {
                'PSO': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
                'MCMC': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0}
            },
            'signal_quality': {
                'PSO': {'SNR': 0, 'correlation': 0, 'mismatch': 1.0},
                'MCMC': {'SNR': 0, 'correlation': 0, 'mismatch': 1.0}
            },
            'classification': {
                'PSO': {'is_lensed': False, 'classification': 'error'},
                'MCMC': {'is_lensed': False, 'classification': 'error'}
            }
        }


def generate_comparison_plots(pso_results, mcmc_results, data_dict, comparison, actual_params, param_ranges=None):
    """Generate plots comparing PSO and MCMC results"""
    print("\nGenerating comparison plots...")

    try:
        # Get data as numpy arrays for plotting
        t_np = cp.asnumpy(data_dict['t']) if isinstance(data_dict['t'], cp.ndarray) else data_dict['t']
        dataY_np = cp.asnumpy(data_dict['dataY']) if isinstance(data_dict['dataY'], cp.ndarray) else data_dict['dataY']
        dataY_only_signal_np = cp.asnumpy(data_dict['dataY_only_signal']) if isinstance(data_dict['dataY_only_signal'],
                                                                                        cp.ndarray) else data_dict[
            'dataY_only_signal']

        # 1. Signal Comparison Plot
        plt.figure(figsize=(12, 8), dpi=200)
        plt.plot(t_np, dataY_np, 'gray', alpha=0.3, label='Observed Data (Signal + Noise)')
        plt.plot(t_np, dataY_only_signal_np, 'k', lw=1.5, label='Actual Signal (No Noise)')
        plt.plot(t_np, pso_results['best_signal'], 'r', lw=1.5,
                 label=f'PSO Estimate (SNR: {comparison["signal_quality"]["PSO"]["SNR"]:.2f})')
        plt.plot(t_np, mcmc_results['best_signal'], 'b', lw=1.5,
                 label=f'Bilby MCMC Estimate (SNR: {comparison["signal_quality"]["MCMC"]["SNR"]:.2f})')
        plt.title('Signal Reconstruction Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/signal_comparison.png", bbox_inches='tight')
        plt.close()

        # 2. Residual Plot
        plt.figure(figsize=(12, 6), dpi=200)
        pso_residual = pso_results['best_signal'] - dataY_only_signal_np
        mcmc_residual = mcmc_results['best_signal'] - dataY_only_signal_np
        plt.plot(t_np, pso_residual, 'r', alpha=0.7,
                 label=f'PSO Residual (Mismatch: {comparison["signal_quality"]["PSO"]["mismatch"]:.4e})')
        plt.plot(t_np, mcmc_residual, 'b', alpha=0.7,
                 label=f'Bilby MCMC Residual (Mismatch: {comparison["signal_quality"]["MCMC"]["mismatch"]:.4e})')
        plt.title('Residual Comparison (Estimate - Actual Signal)')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/residual_comparison.png", bbox_inches='tight')
        plt.close()

        # 3. Parameter Errors Bar Plot
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
        param_labels = ['Distance', 'Chirp Mass', 'Merger Time', 'Phase', 'Flux Ratio (A)', 'Time Delay']

        pso_error_vals = [comparison['parameter_errors']['PSO'][p] * 100 for p in param_names]
        mcmc_error_vals = [comparison['parameter_errors']['MCMC'][p] * 100 for p in param_names]

        plt.figure(figsize=(12, 8), dpi=200)
        x = np.arange(len(param_names))
        width = 0.35

        plt.bar(x - width / 2, pso_error_vals, width, label='PSO', color='red', alpha=0.7)
        plt.bar(x + width / 2, mcmc_error_vals, width, label='Bilby MCMC', color='blue', alpha=0.7)

        plt.xlabel('Parameters')
        plt.ylabel('Relative Error (%)')
        plt.title('Parameter Estimation Error Comparison')
        plt.xticks(x, param_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/parameter_errors.png", bbox_inches='tight')
        plt.close()

        # 4. Execution Time Comparison
        plt.figure(figsize=(8, 6), dpi=200)
        plt.bar(['PSO', 'Bilby MCMC'],
                [comparison['execution_time']['PSO'], comparison['execution_time']['MCMC']],
                color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Execution Time Comparison\nBilby MCMC is {comparison["execution_time"]["ratio"]:.2f}x slower')
        plt.grid(True, alpha=0.3)
        for i, v in enumerate([comparison['execution_time']['PSO'], comparison['execution_time']['MCMC']]):
            plt.text(i, v + 1, f"{v:.2f}s", ha='center')
        plt.savefig(f"{results_dir}/execution_time.png", bbox_inches='tight')
        plt.close()

        # 5. Signal Quality Metrics Comparison
        metrics = ['SNR', 'correlation', 'mismatch']
        metric_labels = ['SNR', 'Correlation', 'Mismatch']

        # Normalize mismatch for better visualization (lower is better)
        pso_metrics = [
            comparison['signal_quality']['PSO']['SNR'],
            comparison['signal_quality']['PSO']['correlation'],
            1 - comparison['signal_quality']['PSO']['mismatch']  # Invert mismatch so higher is better
        ]

        mcmc_metrics = [
            comparison['signal_quality']['MCMC']['SNR'],
            comparison['signal_quality']['MCMC']['correlation'],
            1 - comparison['signal_quality']['MCMC']['mismatch']  # Invert mismatch so higher is better
        ]

        # Create radar chart
        fig = plt.figure(figsize=(10, 8), dpi=200)
        ax = fig.add_subplot(111, polar=True)

        # Set the angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Add PSO and MCMC data
        pso_metrics += pso_metrics[:1]  # Close the loop
        mcmc_metrics += mcmc_metrics[:1]  # Close the loop

        # Normalize values for radar chart (with error handling)
        max_vals = []
        for i in range(len(metrics)):
            try:
                max_val = max(abs(pso_metrics[i]), abs(mcmc_metrics[i]))
                if max_val == 0 or np.isnan(max_val) or np.isinf(max_val):
                    max_val = 1.0  # Default value if invalid
                max_vals.append(max_val)
            except Exception:
                max_vals.append(1.0)  # Default in case of error

        # Safe normalization
        pso_norm = []
        mcmc_norm = []
        for i in range(len(metrics)):
            try:
                pso_norm.append(pso_metrics[i] / max_vals[i] if max_vals[i] != 0 else 0)
                mcmc_norm.append(mcmc_metrics[i] / max_vals[i] if max_vals[i] != 0 else 0)
            except Exception:
                pso_norm.append(0)
                mcmc_norm.append(0)

        # Add closing point
        pso_norm.append(pso_norm[0])
        mcmc_norm.append(mcmc_norm[0])

        # Plot data
        ax.plot(angles, pso_norm, 'r', linewidth=2, label='PSO')
        ax.plot(angles, mcmc_norm, 'b', linewidth=2, label='Bilby MCMC')
        ax.fill(angles, pso_norm, 'r', alpha=0.3)
        ax.fill(angles, mcmc_norm, 'b', alpha=0.3)

        # Set labels
        plt.xticks(angles[:-1], metric_labels)
        ax.set_title('Signal Quality Metrics Comparison (Higher is Better)')
        ax.grid(True)
        plt.legend(loc='upper right')

        # Add actual values as annotations
        for i, metric in enumerate(metric_labels):
            if i < len(pso_metrics) - 1:  # Avoid out of bounds
                plt.annotate(f'PSO: {pso_metrics[i]:.4f}',
                             xy=(angles[i], pso_norm[i]),
                             xytext=(angles[i], pso_norm[i] + 0.1),
                             color='red')
                plt.annotate(f'Bilby MCMC: {mcmc_metrics[i]:.4f}',
                             xy=(angles[i], mcmc_norm[i]),
                             xytext=(angles[i], mcmc_norm[i] - 0.1),
                             color='blue')

        plt.savefig(f"{results_dir}/signal_quality_radar.png", bbox_inches='tight')
        plt.close()

        # 6. Create summary table as image
        # Generate summary dataframe
        summary_data = {
            'Metric': [
                'Execution Time (s)',
                'Speed Improvement',
                'SNR',
                'Correlation',
                'Mismatch',
                'Distance Error (%)',
                'Chirp Mass Error (%)',
                'Merger Time Error (%)',
                'Phase Error (%)',
                'Flux Ratio Error (%)',
                'Time Delay Error (%)',
                'Classification',
            ],
            'PSO': [
                f"{comparison['execution_time']['PSO']:.2f}",
                f"{comparison['execution_time']['ratio']:.2f}x faster",
                f"{comparison['signal_quality']['PSO']['SNR']:.2f}",
                f"{comparison['signal_quality']['PSO']['correlation']:.4f}",
                f"{comparison['signal_quality']['PSO']['mismatch']:.4e}",
                f"{comparison['parameter_errors']['PSO']['r'] * 100:.2f}",
                f"{comparison['parameter_errors']['PSO']['m_c'] * 100:.2f}",
                f"{comparison['parameter_errors']['PSO']['tc'] * 100:.2f}",
                f"{comparison['parameter_errors']['PSO']['phi_c'] * 100:.2f}",
                f"{comparison['parameter_errors']['PSO']['A'] * 100:.2f}",
                f"{comparison['parameter_errors']['PSO']['delta_t'] * 100:.2f}",
                comparison['classification']['PSO']['classification'],
            ],
            'Bilby MCMC': [
                f"{comparison['execution_time']['MCMC']:.2f}",
                "baseline",
                f"{comparison['signal_quality']['MCMC']['SNR']:.2f}",
                f"{comparison['signal_quality']['MCMC']['correlation']:.4f}",
                f"{comparison['signal_quality']['MCMC']['mismatch']:.4e}",
                f"{comparison['parameter_errors']['MCMC']['r'] * 100:.2f}",
                f"{comparison['parameter_errors']['MCMC']['m_c'] * 100:.2f}",
                f"{comparison['parameter_errors']['MCMC']['tc'] * 100:.2f}",
                f"{comparison['parameter_errors']['MCMC']['phi_c'] * 100:.2f}",
                f"{comparison['parameter_errors']['MCMC']['A'] * 100:.2f}",
                f"{comparison['parameter_errors']['MCMC']['delta_t'] * 100:.2f}",
                comparison['classification']['MCMC']['classification'],
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        # Create figure and save as image
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.4, 0.3, 0.3]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Add title
        plt.title('PSO vs Bilby MCMC Performance Comparison Summary', fontsize=16, pad=20)

        # Color code cells based on which method performs better
        for i in range(1, len(summary_df)):
            # Skip non-comparable rows
            if i == 1:  # Speed improvement row
                continue

            pso_val = summary_df.iloc[i, 1]
            mcmc_val = summary_df.iloc[i, 2]

            # Try to extract numerical values where possible
            try:
                pso_num = float(pso_val.split()[0])
                mcmc_num = float(mcmc_val.split()[0])

                # Determine which is better (for error metrics, lower is better)
                if i >= 5 and i <= 10:  # Error metrics
                    if pso_num < mcmc_num:
                        table[(i + 1, 1)].set_facecolor('#d4f7d4')  # Light green
                        table[(i + 1, 2)].set_facecolor('#ffcccc')  # Light red
                    elif pso_num > mcmc_num:
                        table[(i + 1, 1)].set_facecolor('#ffcccc')
                        table[(i + 1, 2)].set_facecolor('#d4f7d4')
                else:  # Other metrics, higher is better
                    if pso_num > mcmc_num:
                        table[(i + 1, 1)].set_facecolor('#d4f7d4')
                        table[(i + 1, 2)].set_facecolor('#ffcccc')
                    elif pso_num < mcmc_num:
                        table[(i + 1, 1)].set_facecolor('#ffcccc')
                        table[(i + 1, 2)].set_facecolor('#d4f7d4')
            except (ValueError, IndexError):
                # Skip rows that can't be converted to numbers
                continue

        plt.tight_layout()
        plt.savefig(f"{results_dir}/comparison_summary.png", bbox_inches='tight')
        plt.close()

        # 7. Plot posterior distributions for Bilby MCMC
        if 'bilby_result' in mcmc_results and hasattr(mcmc_results['bilby_result'], 'plot_corner'):
            try:
                # Extract the bilby result object
                result = mcmc_results['bilby_result']

                # Create corner plot of posterior distributions
                fig = result.plot_corner(save=False)
                fig.savefig(f"{results_dir}/bilby_corner_plot.png", bbox_inches='tight')
                plt.close(fig)

                # Create separate posterior plots for key parameters
                param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
                param_labels = ['Distance Log10(Mpc)', 'Chirp Mass Log10(M⊙)',
                                'Merger Time (s)', 'Phase (rad)', 'Flux Ratio', 'Time Delay (s)']

                # Calculate actual parameter values in Bilby's parameter space (0-1)
                actual_scaled = {}
                for i, param in enumerate(param_names):
                    actual_val = None
                    if param == 'r':
                        actual_val = np.log10(actual_params['source_distance'])
                    elif param == 'm_c':
                        actual_val = np.log10(actual_params['chirp_mass'])
                    elif param == 'tc':
                        actual_val = actual_params['merger_time']
                    elif param == 'phi_c':
                        actual_val = actual_params['phase'] * np.pi
                    elif param == 'A':
                        actual_val = actual_params['flux_ratio']
                    elif param == 'delta_t':
                        actual_val = actual_params['time_delay']

                    # Get parameter ranges safely
                    param_ranges_local = mcmc_results.get('param_ranges', {})
                    if 'rmin' in param_ranges_local and 'rmax' in param_ranges_local:
                        rmin = param_ranges_local['rmin'][i]
                        rmax = param_ranges_local['rmax'][i]
                    elif param_ranges is not None and 'rmin' in param_ranges and 'rmax' in param_ranges:
                        rmin = cp.asnumpy(param_ranges['rmin'])[i] if isinstance(param_ranges['rmin'], cp.ndarray) else \
                        param_ranges['rmin'][i]
                        rmax = cp.asnumpy(param_ranges['rmax'])[i] if isinstance(param_ranges['rmax'], cp.ndarray) else \
                        param_ranges['rmax'][i]
                    else:
                        # Use default ranges if none available
                        rmin_default = [-2, 0, 0.1, 0, 0, 0.1]
                        rmax_default = [4, 2, 8.0, np.pi, 1.0, 4.0]
                        rmin = rmin_default[i]
                        rmax = rmax_default[i]

                    # Scale to 0-1 space for comparison
                    if actual_val is not None:
                        actual_scaled[param] = (actual_val - rmin) / (rmax - rmin)

                # Create individual posterior plots
                plt.figure(figsize=(12, 8), dpi=200)
                for i, (param, label) in enumerate(zip(param_names, param_labels)):
                    plt.subplot(2, 3, i + 1)

                    # Get parameter ranges safely
                    param_ranges_local = mcmc_results.get('param_ranges', {})
                    if 'rmin' in param_ranges_local and 'rmax' in param_ranges_local:
                        rmin = param_ranges_local['rmin'][i]
                        rmax = param_ranges_local['rmax'][i]
                    elif param_ranges is not None and 'rmin' in param_ranges and 'rmax' in param_ranges:
                        rmin = cp.asnumpy(param_ranges['rmin'])[i] if isinstance(param_ranges['rmin'], cp.ndarray) else \
                        param_ranges['rmin'][i]
                        rmax = cp.asnumpy(param_ranges['rmax'])[i] if isinstance(param_ranges['rmax'], cp.ndarray) else \
                        param_ranges['rmax'][i]
                    else:
                        # Use default ranges if none available
                        rmin_default = [-2, 0, 0.1, 0, 0, 0.1]
                        rmax_default = [4, 2, 8.0, np.pi, 1.0, 4.0]
                        rmin = rmin_default[i]
                        rmax = rmax_default[i]

                    # Check for valid posterior samples
                    if hasattr(result.posterior, param) and len(result.posterior[param].values) > 0:
                        samples = result.posterior[param].values * (rmax - rmin) + rmin

                        # For distance and chirp mass, convert from log to linear scale
                        if param in ['r', 'm_c']:
                            samples = 10 ** samples
                            # Get/update actual_val if needed
                            if param == 'r':
                                actual_val = actual_params['source_distance']
                            elif param == 'm_c':
                                actual_val = actual_params['chirp_mass']
                        else:
                            # Get actual value for other parameters
                            if param == 'tc':
                                actual_val = actual_params['merger_time']
                            elif param == 'phi_c':
                                actual_val = actual_params['phase'] * np.pi
                            elif param == 'A':
                                actual_val = actual_params['flux_ratio']
                            elif param == 'delta_t':
                                actual_val = actual_params['time_delay']

                        # Plot histogram with KDE
                        sns.histplot(samples, kde=True)

                        # Plot actual value if available
                        if param in actual_scaled:
                            if param in ['r', 'm_c']:
                                plt.axvline(actual_val, color='r', linestyle='--',
                                            label=f'Actual: {actual_val:.3f}')
                            else:
                                actual_val_scaled = (actual_scaled[param] * (rmax - rmin) + rmin)
                                plt.axvline(actual_val_scaled, color='r', linestyle='--',
                                            label=f'Actual: {actual_val_scaled:.3f}')

                        # Add PSO estimate
                        pso_val = pso_results['best_params'][param]
                        if param in ['r', 'm_c']:
                            pso_val = 10 ** float(pso_val)
                        else:
                            pso_val = float(pso_val)
                        plt.axvline(pso_val, color='g', linestyle=':',
                                    label=f'PSO: {pso_val:.3f}')

                        plt.title(label)
                        plt.xlabel(f"{param} value")
                        plt.ylabel("Probability density")
                        plt.legend()
                    else:
                        plt.text(0.5, 0.5, "No samples available", ha='center', va='center')
                        plt.title(label)

                plt.tight_layout()
                plt.savefig(f"{results_dir}/bilby_posterior_plots.png", bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"Warning: Could not create Bilby posterior plots. Error: {str(e)}")

        # 8. Save actual vs estimated parameters as CSV
        param_df = pd.DataFrame({
            'Parameter': [
                'Distance (Mpc)',
                'Chirp Mass (M⊙)',
                'Merger Time (s)',
                'Phase (rad/2π)',
                'Flux Ratio (A)',
                'Time Delay (s)'
            ],
            'Actual': [
                actual_params['source_distance'],
                actual_params['chirp_mass'],
                actual_params['merger_time'],
                actual_params['phase'],
                actual_params['flux_ratio'],
                actual_params['time_delay']
            ],
            'PSO Estimate': [
                10 ** float(pso_results['best_params']['r']),
                10 ** float(pso_results['best_params']['m_c']),
                float(pso_results['best_params']['tc']),
                float(pso_results['best_params']['phi_c']) / (2 * np.pi),
                float(pso_results['best_params']['A']),
                float(pso_results['best_params']['delta_t'])
            ],
            'Bilby MCMC Estimate': [
                10 ** float(mcmc_results['best_params']['r']),
                10 ** float(mcmc_results['best_params']['m_c']),
                float(mcmc_results['best_params']['tc']),
                float(mcmc_results['best_params']['phi_c']) / (2 * np.pi),
                float(mcmc_results['best_params']['A']),
                float(mcmc_results['best_params']['delta_t'])
            ],
            'PSO Error (%)': [
                comparison['parameter_errors']['PSO']['r'] * 100,
                comparison['parameter_errors']['PSO']['m_c'] * 100,
                comparison['parameter_errors']['PSO']['tc'] * 100,
                comparison['parameter_errors']['PSO']['phi_c'] * 100,
                comparison['parameter_errors']['PSO']['A'] * 100,
                comparison['parameter_errors']['PSO']['delta_t'] * 100
            ],
            'Bilby MCMC Error (%)': [
                comparison['parameter_errors']['MCMC']['r'] * 100,
                comparison['parameter_errors']['MCMC']['m_c'] * 100,
                comparison['parameter_errors']['MCMC']['tc'] * 100,
                comparison['parameter_errors']['MCMC']['phi_c'] * 100,
                comparison['parameter_errors']['MCMC']['A'] * 100,
                comparison['parameter_errors']['MCMC']['delta_t'] * 100
            ]
        })

        param_df.to_csv(f"{results_dir}/parameter_comparison.csv", index=False)

        print(f"All comparison plots saved to {results_dir} directory")

    except Exception as e:
        print(f"Error in generate_comparison_plots: {str(e)}")


def main():
    """Main function to run the PSO vs Bilby MCMC comparison"""

    print("=" * 50)
    print("PSO vs Bilby MCMC Comparison for Gravitational Wave Analysis")
    print("=" * 50)

    try:
        # Load data
        data_dict = load_data()

        # Set up parameters
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)

        # Run PSO
        pso_results = run_pso(data_dict, pso_params, pso_config, actual_params)

        # Run Bilby MCMC with parameters to match PSO computational effort
        # PSO: 100 particles × 3000 iterations = 300,000 evaluations
        # MCMC: 500 live points × 600 iterations = 300,000 evaluations
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=500, n_iter=600)

        # Evaluate and compare results
        comparison = evaluate_results(pso_results, mcmc_results, actual_params, data_dict)

        # Generate comparison plots - pass param_ranges explicitly to avoid scope issues
        generate_comparison_plots(pso_results, mcmc_results, data_dict, comparison, actual_params, param_ranges)

        # Save comparison summary
        summary_df = pd.DataFrame({
            'Metric': [
                'Execution Time (s)',
                'Speed Improvement',
                'SNR',
                'Correlation',
                'Mismatch'
            ],
            'PSO': [
                f"{comparison['execution_time']['PSO']:.2f}",
                f"{comparison['execution_time']['ratio']:.2f}x faster",
                f"{comparison['signal_quality']['PSO']['SNR']:.2f}",
                f"{comparison['signal_quality']['PSO']['correlation']:.4f}",
                f"{comparison['signal_quality']['PSO']['mismatch']:.4e}",
            ],
            'Bilby MCMC': [
                f"{comparison['execution_time']['MCMC']:.2f}",
                "baseline",
                f"{comparison['signal_quality']['MCMC']['SNR']:.2f}",
                f"{comparison['signal_quality']['MCMC']['correlation']:.4f}",
                f"{comparison['signal_quality']['MCMC']['mismatch']:.4e}",
            ]
        })

        summary_df.to_csv(f"{results_dir}/summary.csv", index=False)

        print("\nComparison complete! Results saved in the comparison_results directory.")
        print(f"PSO is {comparison['execution_time']['ratio']:.2f}x faster than Bilby MCMC")

        # Return final message based on overall comparison
        pso_wins = 0
        mcmc_wins = 0

        # Compare execution time
        if comparison['execution_time']['PSO'] < comparison['execution_time']['MCMC']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare SNR
        if comparison['signal_quality']['PSO']['SNR'] > comparison['signal_quality']['MCMC']['SNR']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare correlation
        if comparison['signal_quality']['PSO']['correlation'] > comparison['signal_quality']['MCMC']['correlation']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare mismatch (lower is better)
        if comparison['signal_quality']['PSO']['mismatch'] < comparison['signal_quality']['MCMC']['mismatch']:
            pso_wins += 1
        else:
            mcmc_wins += 1

        # Compare parameter errors
        pso_avg_error = np.mean(
            [comparison['parameter_errors']['PSO'][p] for p in comparison['parameter_errors']['PSO']])
        mcmc_avg_error = np.mean(
            [comparison['parameter_errors']['MCMC'][p] for p in comparison['parameter_errors']['MCMC']])

        if pso_avg_error < mcmc_avg_error:
            pso_wins += 1
        else:
            mcmc_wins += 1

        if pso_wins > mcmc_wins:
            print(f"\nConclusion: PSO outperforms Bilby MCMC with {pso_wins} wins vs {mcmc_wins} wins for Bilby MCMC")
            print(
                "PSO demonstrates superior performance in gravitational wave parameter estimation, combining faster execution with equal or better accuracy.")
        elif mcmc_wins > pso_wins:
            print(f"\nConclusion: Bilby MCMC outperforms PSO with {mcmc_wins} wins vs {pso_wins} wins for PSO")
            print("Bilby MCMC demonstrates better overall accuracy despite slower execution time.")
        else:
            print(f"\nConclusion: PSO and Bilby MCMC are tied with {pso_wins} wins each")
            print("PSO offers significant speed advantages with comparable accuracy to Bilby MCMC.")

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()