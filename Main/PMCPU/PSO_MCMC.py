import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import os
import warnings
from scipy import stats
from scipy.ndimage import gaussian_filter

# Import the PSO implementation
from PSO import (
    crcbqcpsopsd, crcbgenqcsig, normsig4psd, innerprodpsd,
    calculate_matched_filter_snr, pycbc_calculate_match, refine_distance_parameter
)

# Import Bilby for MCMC implementation
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Uniform, PriorDict

from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import match

# Constants
G = const.G
c = const.c
M_sun = 1.989e30
pc = 3.086e16

# Create results directory
results_dir = "L_GW_Result"
os.makedirs(results_dir, exist_ok=True)
print("当前保存目录为：",results_dir)
# Set plotting parameters
plt.rcParams.update({
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})


def load_data():
    """Load gravitational wave data for analysis"""
    print("Loading data...")

    TrainingData = scio.loadmat('../generate_data/noise.mat')
    analysisData = scio.loadmat('../generate_data/data.mat')

    print("Data loaded successfully")

    # Convert data to NumPy arrays
    dataY = np.asarray(analysisData['data'][0])
    training_noise = np.asarray(TrainingData['noise'][0])
    dataY_only_signal = dataY - training_noise

    # Get basic parameters
    Fs = float(analysisData['samples'][0][0])
    dt = 1 / Fs
    t = np.arange(0, 8, dt)

    # Calculate PSD
    psdHigh = np.asarray(TrainingData['psd'][0])

    return {
        't': t,
        'dataY': dataY,
        'dataY_only_signal': dataY_only_signal,
        'noise': training_noise,
        'psdHigh': psdHigh,
        'sampFreq': Fs,
        'nSamples': dataY.size
    }


def setup_parameters(data):
    """Set up parameter ranges for both PSO and MCMC"""

    # Define parameter ranges
    param_ranges = {
        'rmin': np.array([-2, 0, 0.1, 0, 0, 0.1]),
        'rmax': np.array([4, 2, 8.0, np.pi, 1.0, 4.0])
    }

    # Define actual parameters for validation
    actual_params = {
        'chirp_mass': 30.09,
        'merger_time': 7.5,
        'source_distance': 3100.0,
        'flux_ratio': 0.3333,
        'time_delay': 0.9854,
        'phase': 0.25 * np.pi
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

    # Set up PSO configuration
    pso_config = {
        'popsize': 50,
        'maxSteps': 2000,
        'c1': 2.0,
        'c2': 2.0,
        'w_start': 0.9,
        'w_end': 0.5,
        'max_velocity': 0.4,
        'nbrhdSz': 6,
        'disable_early_stop': False  # Enable early stopping
    }

    return param_ranges, pso_params, pso_config, actual_params


class GWLikelihood(Likelihood):
    """Gravitational wave likelihood function for Bilby"""

    def __init__(self, data_dict):
        """Initialize the likelihood with data"""
        super().__init__(parameters={
            'r': None, 'm_c': None, 'tc': None,
            'phi_c': None, 'A': None, 'delta_t': None
        })
        self.data_dict = data_dict

        # Convert data to NumPy arrays
        self.dataX_np = np.asarray(data_dict['dataX'])
        self.dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])
        self.psd_np = np.asarray(data_dict['psdHigh'])

        print(f"GWLikelihood initialized:")
        print(f"  Data shape: {self.dataY_only_signal_np.shape}")
        print(f"  PSD shape: {self.psd_np.shape}")
        print(f"  Sampling frequency: {self.data_dict['sampFreq']}")

    def log_likelihood(self):
        """Log-likelihood function"""
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

            # Check if parameters are in [0,1] range
            if not np.all((params >= 0) & (params <= 1)):
                return -np.inf

            # Map from [0,1] to original parameter range
            unscaled_params = np.zeros(6)
            for i in range(6):
                unscaled_params[i] = params[i] * (self.data_dict['rmax'][i] - self.data_dict['rmin'][i]) + \
                                     self.data_dict['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = unscaled_params

            # Parameter range checks
            if r < -2 or r > 4 or m_c < 0 or m_c > 2 or tc < 0.1 or tc > 8.0:
                return -np.inf

            # Determine lensing usage - Modified threshold for better distinction
            use_lensing = A >= 0.05  # Changed from 0.01 to 0.05 for clearer separation

            # Generate signal
            signal = crcbgenqcsig(
                self.dataX_np, r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )

            if signal is None or np.isnan(signal).any() or np.all(signal == 0):
                return -np.inf

            # Normalize signal
            signal_normalized, normFac = normsig4psd(signal, self.data_dict['sampFreq'], self.psd_np, 1)

            if normFac == 0 or np.isnan(normFac) or np.all(signal_normalized == 0):
                return -np.inf

            # Optimize amplitude
            estAmp = innerprodpsd(
                self.dataY_only_signal_np, signal_normalized,
                self.data_dict['sampFreq'], self.psd_np
            )

            if estAmp is None or np.isnan(estAmp) or abs(estAmp) < 1e-15:
                return -np.inf

            signal_final = estAmp * signal_normalized

            # Calculate log-likelihood using PyCBC match function
            delta_t = 1.0 / self.data_dict['sampFreq']
            ts_signal = TimeSeries(signal_final, delta_t=delta_t)
            ts_data = TimeSeries(self.dataY_only_signal_np, delta_t=delta_t)

            # Handle PSD length adjustment
            nSamples = len(signal_final)
            expected_psd_len = nSamples // 2 + 1
            psd_adjusted = self.psd_np.copy()

            if len(psd_adjusted) < expected_psd_len:
                extended_psd = np.zeros(expected_psd_len)
                extended_psd[:len(psd_adjusted)] = psd_adjusted
                extended_psd[len(psd_adjusted):] = psd_adjusted[-1]
                psd_adjusted = extended_psd
            elif len(psd_adjusted) > expected_psd_len:
                psd_adjusted = psd_adjusted[:expected_psd_len]

            # Ensure PSD has no zero values
            min_psd = np.max(psd_adjusted) * 1e-14
            psd_adjusted = np.maximum(psd_adjusted, min_psd)

            delta_f = 1.0 / (len(signal_final) * delta_t)
            psd_series = FrequencySeries(psd_adjusted, delta_f=delta_f)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                match_value, _ = match(ts_signal, ts_data, psd=psd_series, low_frequency_cutoff=10.0)

                if match_value is None or np.isnan(match_value) or match_value <= 0 or match_value > 1:
                    return -np.inf

                # Convert match to log-likelihood
                log_likelihood_value = 0.5 * len(self.dataY_only_signal_np) * np.log(max(match_value, 1e-10))

                if np.isnan(log_likelihood_value) or np.isinf(log_likelihood_value):
                    return -np.inf

                return log_likelihood_value

        except Exception:
            return -np.inf


def run_bilby_mcmc(data_dict, param_ranges, n_live_points, n_iter, enable_distance_refinement=True, actual_params=None):
    """Bilby MCMC execution function"""
    print("Starting Bilby MCMC analysis...")

    # Create dictionary for MCMC
    mcmc_data = {
        'dataX': np.asarray(data_dict['t']),
        'dataY': np.asarray(data_dict['dataY']),
        'dataY_only_signal': np.asarray(data_dict['dataY_only_signal']),
        'sampFreq': data_dict['sampFreq'],
        'psdHigh': np.asarray(data_dict['psdHigh']),
        'rmin': np.asarray(param_ranges['rmin']),
        'rmax': np.asarray(param_ranges['rmax'])
    }

    start_time = time.time()

    try:
        # Create likelihood
        likelihood = GWLikelihood(mcmc_data)

        # Create priors
        priors = PriorDict()
        priors['r'] = Uniform(minimum=0, maximum=1, name='r')
        priors['m_c'] = Uniform(minimum=0, maximum=1, name='m_c')
        priors['tc'] = Uniform(minimum=0, maximum=1, name='tc')
        priors['phi_c'] = Uniform(minimum=0, maximum=1, name='phi_c')
        priors['A'] = Uniform(minimum=0, maximum=1, name='A')
        priors['delta_t'] = Uniform(minimum=0, maximum=1, name='delta_t')

        # Run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler='dynesty',
            nlive=n_live_points,
            walks=40,
            verbose=True,
            maxiter=n_iter,
            outdir=results_dir,
            label='gw_analysis',
            resume=False,
            check_point_plot=False,
            bound='multi',
            sample='rwalk',
            check_point=True,
            check_point_delta_t=600,
            plot=True,
        )

        # Extract iteration history
        mcmc_iteration_history = []
        if hasattr(result, 'sampler') and hasattr(result.sampler, 'results'):
            sampler_results = result.sampler.results
            if hasattr(sampler_results, 'logz'):
                mcmc_iteration_history = sampler_results.logz
            elif len(result.posterior) > 0:
                mcmc_iteration_history = result.posterior['log_likelihood'].values

    except Exception as e:
        print(f"Error in Bilby MCMC: {str(e)}")
        end_time = time.time()
        return {
            'duration': end_time - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': mcmc_data,
            'iteration_history': [],
            'posterior_samples': None,
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }

    end_time = time.time()
    mcmc_duration = end_time - start_time
    print(f"Bilby MCMC completed in {mcmc_duration:.2f} seconds")

    try:
        # Get best fit parameters
        if len(result.posterior) > 0:
            best_idx = np.argmax(result.posterior['log_likelihood'].values)
            best_params_bilby = result.posterior.iloc[best_idx]

            # Unscale parameters
            best_params = np.zeros(6)
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
            for i, param in enumerate(param_names):
                best_params[i] = best_params_bilby[param] * (mcmc_data['rmax'][i] - mcmc_data['rmin'][i]) + \
                                 mcmc_data['rmin'][i]

            r, m_c, tc, phi_c, A, delta_t = best_params

            print(f"MCMC Best Parameters:")
            print(f"  r (log10 distance): {r:.4f}")
            print(f"  m_c (log10 chirp mass): {m_c:.4f}")
            print(f"  tc (merger time): {tc:.4f}")
            print(f"  phi_c (phase): {phi_c:.4f}")
            print(f"  A (flux ratio): {A:.4f}")
            print(f"  delta_t (time delay): {delta_t:.4f}")

            # Apply distance refinement if enabled
            distance_refinement = {'enabled': enable_distance_refinement}

            if enable_distance_refinement:
                print("Applying distance parameter refinement...")
                initial_params_dict = {
                    'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
                }

                try:
                    refined_r, refinement_info = refine_distance_parameter(
                        initial_params_dict,
                        mcmc_data['dataX'],
                        mcmc_data['dataY_only_signal'],
                        mcmc_data['sampFreq'],
                        mcmc_data['psdHigh'],
                        param_ranges,
                        actual_params
                    )

                    distance_refinement.update({
                        'original_distance': r,
                        'refined_distance': refined_r,
                        'refinement_info': refinement_info
                    })

                    if refinement_info['status'] == 'success':
                        print("Distance refinement successful")
                        r = refined_r

                except Exception as e:
                    print(f"Distance refinement failed: {e}")
                    distance_refinement.update({
                        'original_distance': r,
                        'refined_distance': r,
                        'refinement_info': {'status': 'error', 'error': str(e)}
                    })

            # Generate best signal
            use_lensing = A >= 0.05  # Changed threshold for consistency
            best_signal = crcbgenqcsig(
                mcmc_data['dataX'], r, m_c, tc, phi_c, A, delta_t,
                use_lensing=use_lensing
            )

            # Apply normalization and amplitude optimization
            best_signal, normFac = normsig4psd(best_signal, mcmc_data['sampFreq'], mcmc_data['psdHigh'], 1)
            estAmp = innerprodpsd(
                mcmc_data['dataY_only_signal'], best_signal,
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )
            best_signal = estAmp * best_signal

            # Calculate SNR
            snr = calculate_matched_filter_snr(
                best_signal, mcmc_data['dataY_only_signal'],
                mcmc_data['psdHigh'], mcmc_data['sampFreq']
            )

            # Determine classification
            is_lensed = A >= 0.05  # Changed threshold for consistency
            if snr < 8:
                classification = "noise"
            elif is_lensed:
                classification = "lens_signal"
            else:
                classification = "signal"

            # Calculate match
            match_value = pycbc_calculate_match(
                best_signal, mcmc_data['dataY_only_signal'],
                mcmc_data['sampFreq'], mcmc_data['psdHigh']
            )

            print(f"MCMC Results Summary:")
            print(f"  SNR: {snr:.2f}")
            print(f"  Match: {match_value:.4f}")
            print(f"  Classification: {classification}")

            samples = result.posterior[param_names].values
            posterior_samples = result.posterior.copy()

        else:
            print("Warning: No posterior samples found.")
            best_params = np.zeros(6)
            r, m_c, tc, phi_c, A, delta_t = best_params
            best_signal = np.zeros_like(mcmc_data['dataY_only_signal'])
            snr = 0
            is_lensed = False
            classification = "error"
            match_value = 0.0
            samples = np.zeros((1, 6))
            posterior_samples = None
            distance_refinement = {'enabled': enable_distance_refinement, 'status': 'no_samples'}

        mcmc_results = {
            'duration': mcmc_duration,
            'best_params': {
                'r': r, 'm_c': m_c, 'tc': tc, 'phi_c': phi_c, 'A': A, 'delta_t': delta_t
            },
            'best_signal': best_signal,
            'snr': float(snr),
            'is_lensed': is_lensed,
            'classification': classification,
            'match': match_value,
            'samples': samples if 'samples' in locals() else np.zeros((1, 6)),
            'log_probs': result.posterior['log_likelihood'].values if len(result.posterior) > 0 else np.array([-np.inf]),
            'bilby_result': result,
            'posterior_samples': posterior_samples,
            'param_ranges': {
                'rmin': mcmc_data['rmin'],
                'rmax': mcmc_data['rmax']
            },
            'iteration_history': mcmc_iteration_history,
            'distance_refinement': distance_refinement
        }

        return mcmc_results

    except Exception as e:
        print(f"Error in processing Bilby results: {str(e)}")
        return {
            'duration': mcmc_duration,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(mcmc_data['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'param_ranges': {'rmin': mcmc_data['rmin'], 'rmax': mcmc_data['rmax']},
            'iteration_history': [],
            'posterior_samples': None,
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def run_pso(data_dict, pso_params, pso_config, actual_params, n_runs=1, enable_distance_refinement=True):
    """Run PSO analysis"""
    print("Starting PSO analysis...")

    start_time = time.time()

    try:
        outResults, outStruct = crcbqcpsopsd(
            pso_params, pso_config, n_runs,
            use_two_step=True,
            actual_params=actual_params,
            enable_distance_refinement=enable_distance_refinement
        )

        end_time = time.time()
        pso_duration = end_time - start_time
        print(f"PSO completed in {pso_duration:.2f} seconds")

        # Extract best run
        best_run_idx = outResults['bestRun']
        best_signal = outResults['bestSig']

        # Calculate match
        dataY_only_signal = np.asarray(data_dict['dataY_only_signal'])
        match_value = pycbc_calculate_match(
            best_signal, dataY_only_signal,
            data_dict['sampFreq'], data_dict['psdHigh']
        )

        # Extract iteration history
        pso_iteration_history = []
        if 'fitnessHistory' in outStruct[best_run_idx]:
            fitness_history = outStruct[best_run_idx]['fitnessHistory']
            pso_iteration_history = [-f for f in fitness_history]

        distance_refinement = outResults.get('distance_refinement', {'enabled': enable_distance_refinement})

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
            'best_signal': best_signal,
            'snr': outResults['allRunsOutput'][best_run_idx]['SNR_pycbc'],
            'is_lensed': outResults['is_lensed'],
            'classification': outResults['classification'],
            'match': match_value,
            'all_runs': outResults['allRunsOutput'],
            'structures': outStruct,
            'iteration_history': pso_iteration_history,
            'distance_refinement': distance_refinement
        }

        return pso_results

    except Exception as e:
        print(f"Error in PSO: {str(e)}")
        return {
            'duration': time.time() - start_time,
            'best_params': {'r': 0, 'm_c': 0, 'tc': 0, 'phi_c': 0, 'A': 0, 'delta_t': 0},
            'best_signal': np.zeros_like(data_dict['dataY_only_signal']),
            'snr': 0,
            'is_lensed': False,
            'classification': "error",
            'match': 0.0,
            'iteration_history': [],
            'distance_refinement': {'enabled': enable_distance_refinement, 'status': 'error'}
        }


def evaluate_results(pso_results, mcmc_results, actual_params, data_dict):
    """Compare PSO and MCMC results"""
    print("\n============= Performance Comparison =============")

    try:
        # Convert actual parameters
        actual_r_log = np.log10(actual_params['source_distance'])
        actual_m_c_log = np.log10(actual_params['chirp_mass'])

        # Calculate parameter errors for PSO
        pso_errors = {
            'r': abs((10 ** float(pso_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']),
            'm_c': abs((10 ** float(pso_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']),
            'tc': abs((float(pso_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']),
            'phi_c': abs(abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'])) / (2 * np.pi),
            'A': abs((float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay'])
        }

        # Calculate parameter errors for MCMC
        mcmc_errors = {
            'r': abs((10 ** float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']),
            'm_c': abs((10 ** float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']),
            'tc': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']),
            'phi_c': abs(abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'])) / (2 * np.pi),
            'A': abs((float(mcmc_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs((float(mcmc_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay'])
        }

        # Calculate match with actual signal
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])
        pso_match = pycbc_calculate_match(pso_results['best_signal'], dataY_only_signal_np,
                                          data_dict['sampFreq'], data_dict['psdHigh'])
        mcmc_match = pycbc_calculate_match(mcmc_results['best_signal'], dataY_only_signal_np,
                                           data_dict['sampFreq'], data_dict['psdHigh'])

        comparison = {
            'execution_time': {
                'PSO': pso_results['duration'],
                'MCMC': mcmc_results['duration'],
                'ratio': mcmc_results['duration'] / pso_results['duration'] if pso_results['duration'] > 0 else float('inf')
            },
            'parameter_errors': {
                'PSO': pso_errors,
                'MCMC': mcmc_errors
            },
            'signal_quality': {
                'PSO': {
                    'SNR': float(pso_results['snr']),
                    'match': pso_match,
                    'match_final': float(pso_results['match'])
                },
                'MCMC': {
                    'SNR': float(mcmc_results['snr']),
                    'match': mcmc_match,
                    'match_final': float(mcmc_results['match'])
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
            },
            'distance_refinement': {
                'PSO': pso_results.get('distance_refinement', {}),
                'MCMC': mcmc_results.get('distance_refinement', {})
            }
        }

        print(f"Execution Time: PSO: {comparison['execution_time']['PSO']:.2f}s, MCMC: {comparison['execution_time']['MCMC']:.2f}s")
        print(f"Speed Improvement: MCMC is {comparison['execution_time']['ratio']:.2f}x slower than PSO")

        # Print parameter errors
        print("\nParameter Estimation Errors (relative):")
        for param in pso_errors.keys():
            print(f"  {param}: PSO: {pso_errors[param] * 100:.2f}%, MCMC: {mcmc_errors[param] * 100:.2f}%")

        # Print signal quality metrics
        print("\nSignal Quality Metrics:")
        print(f"  SNR: PSO: {comparison['signal_quality']['PSO']['SNR']:.2f}, MCMC: {comparison['signal_quality']['MCMC']['SNR']:.2f}")
        print(f"  Match: PSO: {comparison['signal_quality']['PSO']['match']:.4f}, MCMC: {comparison['signal_quality']['MCMC']['match']:.4f}")

        return comparison

    except Exception as e:
        print(f"Error in evaluate_results: {str(e)}")
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
                'PSO': {'SNR': 0, 'match': 0, 'match_final': 0.0},
                'MCMC': {'SNR': 0, 'match': 0, 'match_final': 0.0}
            },
            'classification': {
                'PSO': {'is_lensed': False, 'classification': 'error'},
                'MCMC': {'is_lensed': False, 'classification': 'error'}
            },
            'distance_refinement': {
                'PSO': {'enabled': False},
                'MCMC': {'enabled': False}
            }
        }


def create_parameter_comparison_table(pso_results, mcmc_results, actual_params, param_ranges):
    """Create a comprehensive parameter comparison table without refinement status"""
    print("\nCreating parameter comparison table...")

    try:
        # Parameter names and their descriptions
        param_info = {
            'r': {'name': 'Distance', 'unit': 'Mpc', 'transform': lambda x: 10 ** x},
            'm_c': {'name': 'Chirp Mass', 'unit': 'M☉', 'transform': lambda x: 10 ** x},
            'tc': {'name': 'Merger Time', 'unit': 's', 'transform': lambda x: x},
            'phi_c': {'name': 'Phase', 'unit': 'rad', 'transform': lambda x: x },
            'A': {'name': 'Flux Ratio', 'unit': '', 'transform': lambda x: x},
            'delta_t': {'name': 'Time Delay', 'unit': 's', 'transform': lambda x: x}
        }

        # Actual parameter values
        actual_values = {
            'r': actual_params['source_distance'],
            'm_c': actual_params['chirp_mass'],
            'tc': actual_params['merger_time'],
            'phi_c': actual_params['phase'],
            'A': actual_params['flux_ratio'],
            'delta_t': actual_params['time_delay']
        }

        # Create comparison data
        table_data = []

        for param in param_info.keys():
            # Get estimated values
            pso_raw = float(pso_results['best_params'][param])
            mcmc_raw = float(mcmc_results['best_params'][param])

            # Transform to physical units
            pso_value = param_info[param]['transform'](pso_raw)
            mcmc_value = param_info[param]['transform'](mcmc_raw)
            actual_value = actual_values[param]

            # Calculate errors
            pso_error = abs((pso_value - actual_value) / actual_value) * 100
            mcmc_error = abs((mcmc_value - actual_value) / actual_value) * 100

            # Add to table
            table_data.append({
                'Parameter': param_info[param]['name'],
                'Unit': param_info[param]['unit'],
                'True Value': f"{actual_value:.4f}",
                'PSO Estimate': f"{pso_value:.4f}",
                'PSO Error (%)': f"{pso_error:.2f}",
                'MCMC Estimate': f"{mcmc_value:.4f}",
                'MCMC Error (%)': f"{mcmc_error:.2f}"
            })

        # Create DataFrame and save as CSV
        param_df = pd.DataFrame(table_data)
        param_df.to_csv(f"{results_dir}/parameter_comparison_table.csv", index=False)

        print(f"Parameter comparison table saved to {results_dir}/parameter_comparison_table.csv")

        return param_df

    except Exception as e:
        print(f"Error creating parameter comparison table: {str(e)}")
        return pd.DataFrame()


def create_performance_comparison_table(pso_results, mcmc_results, comparison, results_dir="."):
    """Create performance comparison table for PSO and MCMC algorithms"""
    print("\nCreating performance comparison table...")
    
    try:
        # Create performance comparison data
        performance_data = []
        
        # Algorithm performance metrics
        metrics = [
            {
                'Metric': 'Execution Time (seconds)',
                'PSO': f"{comparison['execution_time']['PSO']:.2f}",
                'MCMC': f"{comparison['execution_time']['MCMC']:.2f}",
                'PSO Advantage': f"{comparison['execution_time']['ratio']:.2f}x faster"
            },
            {
                'Metric': 'Signal-to-Noise Ratio (SNR)',
                'PSO': f"{comparison['signal_quality']['PSO']['SNR']:.2f}",
                'MCMC': f"{comparison['signal_quality']['MCMC']['SNR']:.2f}",
                'PSO Advantage': "Higher" if comparison['signal_quality']['PSO']['SNR'] > comparison['signal_quality']['MCMC']['SNR'] else "Lower"
            },
            {
                'Metric': 'Template Match',
                'PSO': f"{comparison['signal_quality']['PSO']['match_final']:.4f}",
                'MCMC': f"{comparison['signal_quality']['MCMC']['match_final']:.4f}",
                'PSO Advantage': "Higher" if comparison['signal_quality']['PSO']['match_final'] > comparison['signal_quality']['MCMC']['match_final'] else "Lower"
            },
            {
                'Metric': 'Classification',
                'PSO': comparison['classification']['PSO']['classification'],
                'MCMC': comparison['classification']['MCMC']['classification'],
                'PSO Advantage': "Same" if comparison['classification']['PSO']['classification'] == comparison['classification']['MCMC']['classification'] else "Different"
            },
            {
                'Metric': 'Lensing Detection',
                'PSO': "Yes" if comparison['classification']['PSO']['is_lensed'] else "No",
                'MCMC': "Yes" if comparison['classification']['MCMC']['is_lensed'] else "No",
                'PSO Advantage': "Same" if comparison['classification']['PSO']['is_lensed'] == comparison['classification']['MCMC']['is_lensed'] else "Different"
            }
        ]
        
        # Add distance parameter error comparison
        dist_error_pso = comparison['parameter_errors']['PSO']['r'] * 100
        dist_error_mcmc = comparison['parameter_errors']['MCMC']['r'] * 100
        metrics.append({
            'Metric': 'Distance Error (%)',
            'PSO': f"{dist_error_pso:.2f}",
            'MCMC': f"{dist_error_mcmc:.2f}",
            'PSO Advantage': "Lower" if dist_error_pso < dist_error_mcmc else "Higher"
        })
        
        # Add chirp mass parameter error comparison
        mass_error_pso = comparison['parameter_errors']['PSO']['m_c'] * 100
        mass_error_mcmc = comparison['parameter_errors']['MCMC']['m_c'] * 100
        metrics.append({
            'Metric': 'Chirp Mass Error (%)',
            'PSO': f"{mass_error_pso:.2f}",
            'MCMC': f"{mass_error_mcmc:.2f}",
            'PSO Advantage': "Lower" if mass_error_pso < mass_error_mcmc else "Higher"
        })
        
        # Add merger time parameter error comparison
        time_error_pso = comparison['parameter_errors']['PSO']['tc'] * 100
        time_error_mcmc = comparison['parameter_errors']['MCMC']['tc'] * 100
        metrics.append({
            'Metric': 'Merger Time Error (%)',
            'PSO': f"{time_error_pso:.2f}",
            'MCMC': f"{time_error_mcmc:.2f}",
            'PSO Advantage': "Lower" if time_error_pso < time_error_mcmc else "Higher"
        })
        
        performance_data.extend(metrics)
        
        # Create DataFrame and save as CSV
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(f"{results_dir}/algorithm_performance_comparison.csv", index=False)
        
        print(f"Performance comparison table saved to {results_dir}/algorithm_performance_comparison.csv")
        
        # Display table summary
        print("\nAlgorithm Performance Summary:")
        print("="*60)
        for metric in performance_data:
            print(f"{metric['Metric']:<25}: PSO={metric['PSO']:<12} MCMC={metric['MCMC']:<12} ({metric['PSO Advantage']})")
        
        return performance_df
        
    except Exception as e:
        print(f"Error creating performance comparison table: {str(e)}")
        return pd.DataFrame()


def create_mcmc_corner_plot_with_true_values(mcmc_results, param_ranges, actual_params, results_dir="."):
    """
    Create enhanced corner plot with true value markers for MCMC results
    Modified according to user requirements:
    1. If signal is not lensed, exclude A and delta_t parameters
    2. Remove parameter units from labels  
    3. Remove density label from histograms
    4. Optimize layout to reduce whitespace and center peak distributions
    """
    print("\nGenerating MCMC corner plot with true values...")
    
    try:
        if mcmc_results['posterior_samples'] is None or len(mcmc_results['posterior_samples']) == 0:
            print("No MCMC posterior samples available for plotting")
            return
            
        posterior = mcmc_results['posterior_samples']
        
        # Determine if signal is lensed
        best_A = float(mcmc_results['best_params']['A'])
        is_lensed = best_A >= 0.05
        
        # Select parameters based on lensing status
        if is_lensed:
            param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
            param_labels = [r'$\log_{10}(D_L)$', r'$\log_{10}(M_c)$', r'$t_c$', 
                           r'$\phi_c$', r'$A$', r'$\tau$']
            print("Detected lensed signal - including all parameters in corner plot")
        else:
            param_names = ['r', 'm_c', 'tc', 'phi_c']  # Exclude A and delta_t
            param_labels = [r'$\log_{10}(D_L)$', r'$\log_{10}(M_c)$', r'$t_c$', r'$\phi_c$']
            print("Detected non-lensed signal - excluding A and delta_t from corner plot")
        
        n_params = len(param_names)
        
        # Convert to physical units
        physical_samples = np.zeros((len(posterior), n_params))
        actual_physical = np.zeros(n_params)
        
        for i, param in enumerate(param_names):
            # Unscale from [0,1] to original parameter range
            param_idx = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t'].index(param)
            unscaled = posterior[param].values * (param_ranges['rmax'][param_idx] - param_ranges['rmin'][param_idx]) + \
                       param_ranges['rmin'][param_idx]
            
            # For distance and chirp mass, keep in log space for plotting
            if param == 'r':
                physical_samples[:, i] = unscaled  # Keep log10(distance)
                actual_physical[i] = np.log10(actual_params['source_distance'])
            elif param == 'm_c':
                physical_samples[:, i] = unscaled  # Keep log10(chirp_mass)
                actual_physical[i] = np.log10(actual_params['chirp_mass'])
            else:
                physical_samples[:, i] = unscaled
                if param == 'tc':
                    actual_physical[i] = actual_params['merger_time']
                elif param == 'phi_c':
                    actual_physical[i] = actual_params['phase']
                elif param == 'A':
                    actual_physical[i] = actual_params['flux_ratio']
                elif param == 'delta_t':
                    actual_physical[i] = actual_params['time_delay']
        
        # Create figure with optimized spacing
        fig = plt.figure(figsize=(4*n_params, 4*n_params))
        gs = fig.add_gridspec(n_params, n_params, hspace=0.02, wspace=0.02)
        
        # Color scheme
        scatter_color = '#4472C4'
        contour_colors = ['#C5504B', '#70AD47', '#FFC000']
        true_color = '#000000'
        
        for i in range(n_params):
            for j in range(n_params):
                if i > j:
                    # 2D contour and scatter plot
                    ax = fig.add_subplot(gs[i, j])
                    
                    # Sample subset for faster plotting
                    n_plot = min(2000, len(physical_samples))
                    idx = np.random.choice(len(physical_samples), n_plot, replace=False)
                    
                    x_data = physical_samples[idx, j]
                    y_data = physical_samples[idx, i]
                    
                    # Calculate optimized range to center the distribution
                    x_percentiles = np.percentile(x_data, [5, 50, 95])
                    y_percentiles = np.percentile(y_data, [5, 50, 95])
                    
                    x_median, y_median = x_percentiles[1], y_percentiles[1]
                    x_range_half = max(x_percentiles[2] - x_median, x_median - x_percentiles[0]) * 1.2
                    y_range_half = max(y_percentiles[2] - y_median, y_median - y_percentiles[0]) * 1.2
                    
                    x_range = [x_median - x_range_half, x_median + x_range_half]
                    y_range = [y_median - y_range_half, y_median + y_range_half]
                    
                    # Extend to include true values
                    x_true = actual_physical[j]
                    y_true = actual_physical[i]
                    
                    if x_true < x_range[0] or x_true > x_range[1]:
                        x_range = [min(x_range[0], x_true - x_range_half*0.1), 
                                  max(x_range[1], x_true + x_range_half*0.1)]
                    if y_true < y_range[0] or y_true > y_range[1]:
                        y_range = [min(y_range[0], y_true - y_range_half*0.1), 
                                  max(y_range[1], y_true + y_range_half*0.1)]
                    
                    # Create histogram
                    n_bins = 35
                    H, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=n_bins, 
                                                        range=[x_range, y_range], density=True)
                    
                    # Smooth the histogram
                    H_smooth = gaussian_filter(H.T, sigma=0.8)
                    
                    # Calculate contour levels
                    if np.max(H_smooth) > 0:
                        H_flat = H_smooth.ravel()
                        H_sorted = np.sort(H_flat)[::-1]
                        H_cumsum = np.cumsum(H_sorted)
                        H_cumsum = H_cumsum / H_cumsum[-1]
                        
                        target_fractions = [0.39, 0.68, 0.95]
                        levels = []
                        
                        for frac in target_fractions:
                            idx = np.argmax(H_cumsum >= frac)
                            if idx > 0 and idx < len(H_sorted):
                                levels.append(H_sorted[idx])
                        
                        levels = sorted(list(set([l for l in levels if l > 0])))
                        
                        if len(levels) < 2:
                            max_val = np.max(H_smooth)
                            levels = [max_val * 0.15, max_val * 0.4, max_val * 0.7]
                            levels = [l for l in levels if l > 0]
                    else:
                        levels = []
                    
                    # Create coordinate grids
                    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, 
                                      (y_edges[:-1] + y_edges[1:]) / 2)
                    
                    # Plot contours
                    if len(levels) >= 2:
                        try:
                            ax.contour(X, Y, H_smooth, levels=levels, 
                                     colors=contour_colors[:len(levels)], 
                                     linewidths=[1.0, 1.5, 2.0][:len(levels)], 
                                     alpha=0.8)
                        except:
                            pass
                    
                    # Scatter plot
                    n_scatter = min(800, len(x_data))
                    if n_scatter < len(x_data):
                        scatter_idx = np.random.choice(len(x_data), n_scatter, replace=False)
                        x_scatter = x_data[scatter_idx]
                        y_scatter = y_data[scatter_idx]
                    else:
                        x_scatter = x_data
                        y_scatter = y_data
                        
                    ax.scatter(x_scatter, y_scatter, c=scatter_color, alpha=0.3, s=1.0, rasterized=True)
                    
                    # True value lines
                    ax.axvline(x_true, color=true_color, linestyle='--', 
                              linewidth=2, alpha=0.8, zorder=10)
                    ax.axhline(y_true, color=true_color, linestyle='--', 
                              linewidth=2, alpha=0.8, zorder=10)
                    
                    # Set limits
                    ax.set_xlim(x_range)
                    ax.set_ylim(y_range)
                    
                    # Labels and ticks
                    if i == n_params - 1:  # Bottom row
                        ax.set_xlabel(param_labels[j], fontsize=16, fontweight='bold')
                        ax.tick_params(axis='x', labelsize=12)
                    else:
                        ax.set_xticklabels([])
                    
                    if j == 0:  # Left column
                        ax.set_ylabel(param_labels[i], fontsize=16, fontweight='bold')
                        ax.tick_params(axis='y', labelsize=12)
                    else:
                        ax.set_yticklabels([])
                    
                    # Grid
                    ax.grid(True, alpha=0.15, linewidth=0.5)
                    
                elif i == j:
                    # 1D histogram - optimized and centered
                    ax = fig.add_subplot(gs[i, j])
                    
                    data = physical_samples[:, i]
                    true_val = actual_physical[i]
                    
                    # Calculate optimized range to center the distribution
                    data_percentiles = np.percentile(data, [5, 50, 95])
                    data_median = data_percentiles[1]
                    data_range_half = max(data_percentiles[2] - data_median, 
                                         data_median - data_percentiles[0]) * 1.2
                    
                    data_range = [data_median - data_range_half, data_median + data_range_half]
                    
                    # Extend to include true value if needed
                    if true_val < data_range[0] or true_val > data_range[1]:
                        data_range = [min(data_range[0], true_val - data_range_half*0.1), 
                                     max(data_range[1], true_val + data_range_half*0.1)]
                    
                    # Create histogram
                    n_bins = 40
                    counts, bins, patches = ax.hist(data, bins=n_bins, range=data_range, 
                                                   density=True, alpha=0.6, color=scatter_color,
                                                   edgecolor='black', linewidth=0.5)
                    
                    # Add KDE curve
                    try:
                        kde = stats.gaussian_kde(data)
                        x_kde = np.linspace(data_range[0], data_range[1], 200)
                        y_kde = kde(x_kde)
                        ax.plot(x_kde, y_kde, color='darkred', linewidth=2.5, alpha=0.8)
                    except:
                        pass
                    
                    # True value line
                    ax.axvline(true_val, color=true_color, linestyle='--', 
                              linewidth=2.5, alpha=0.9, zorder=10)
                    
                    # Mean line
                    mean_val = np.mean(data)
                    ax.axvline(mean_val, color='blue', linestyle='-', linewidth=2, alpha=0.7)
                    
                    # Set limits
                    ax.set_xlim(data_range)
                    
                    # Labels - Remove density label as requested
                    if i == n_params - 1:  # Bottom
                        ax.set_xlabel(param_labels[i], fontsize=16, fontweight='bold')
                        ax.tick_params(axis='x', labelsize=12)
                    else:
                        ax.set_xticklabels([])
                    
                    # Remove y-axis labels for histograms as requested
                    ax.set_yticklabels([])
                    ax.tick_params(axis='y', left=False)
                    
                    # Grid
                    ax.grid(True, alpha=0.15, linewidth=0.5)
                    
                    # Add statistics text box (smaller)
                    std_val = np.std(data)
                    stats_text = f'μ: {mean_val:.3f}\nσ: {std_val:.3f}\nTrue: {true_val:.3f}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        
                else:
                    # Upper triangle - add legend and information
                    ax = fig.add_subplot(gs[i, j])
                    ax.axis('off')
                    
                    # Add legend in the top-right corner
                    if i == 0 and j == n_params - 1:
                        from matplotlib.lines import Line2D
        
        # Add overall title with lensing status
        title_suffix = " (Lensed Signal)" if is_lensed else " (Non-lensed Signal)"
        fig.suptitle('MCMC Posterior Analysis' + title_suffix, 
                    fontsize=20, fontweight='bold', y=0.98)

        # Save the plot
        plot_filename = f"gw_analysis_corner_{'lensed' if is_lensed else 'unlensed'}.png"
        plt.savefig(f"{results_dir}/{plot_filename}",
                    bbox_inches='tight', dpi=300, facecolor='white')

        # Save as PDF format
        plot_filename_pdf = f"gw_analysis_corner_{'lensed' if is_lensed else 'unlensed'}.pdf"
        plt.savefig(f"{results_dir}/{plot_filename}",
                    bbox_inches='tight', format='pdf', facecolor='white')

        plt.close()

        print(f"Corner plot saved to {results_dir}/{plot_filename}")
        print(f"Corner plot PDF saved to {results_dir}/{plot_filename}")
        
        print(f"Corner plot saved to {results_dir}/{plot_filename}")
        
    except Exception as e:
        print(f"Error in corner plot: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_publication_plots(pso_results, mcmc_results, data_dict, comparison, actual_params):
    """Generate publication-quality plots - only waveform reconstruction"""
    print("\nGenerating publication-quality waveform reconstruction plot...")

    try:
        # Get data as numpy arrays
        t_np = np.asarray(data_dict['t'])
        dataY_np = np.asarray(data_dict['dataY'])
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])

        # Figure: Waveform Reconstruction Comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Top panel: Signal comparison
        ax1.plot(t_np, dataY_only_signal_np, 'k-', linewidth=2, label='Injected Signal', alpha=0.8)
        ax1.plot(t_np, pso_results['best_signal'], 'r--', linewidth=2,
                 label=f'PSO Reconstruction (SNR = {comparison["signal_quality"]["PSO"]["SNR"]:.1f})', alpha=0.9)
        ax1.plot(t_np, mcmc_results['best_signal'], 'b:', linewidth=2,
                 label=f'MCMC Reconstruction (SNR = {comparison["signal_quality"]["MCMC"]["SNR"]:.1f})', alpha=0.9)

        ax1.set_ylabel(r'Strain', fontsize=14)
        ax1.set_title('Gravitational Wave Signal Reconstruction', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([t_np[0], t_np[-1]])

        # Bottom panel: Residuals
        pso_residual = (pso_results['best_signal'] - dataY_only_signal_np)
        mcmc_residual = (mcmc_results['best_signal'] - dataY_only_signal_np)

        ax2.plot(t_np, pso_residual, 'r-', linewidth=1.5, alpha=0.8,
                 label=f'PSO Residual (Match = {comparison["signal_quality"]["PSO"]["match_final"]:.3f})')
        ax2.plot(t_np, mcmc_residual, 'b-', linewidth=1.5, alpha=0.8,
                 label=f'MCMC Residual (Match = {comparison["signal_quality"]["MCMC"]["match_final"]:.3f})')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        ax2.set_xlabel('Time [s]', fontsize=14)
        ax2.set_ylabel(r'Residual', fontsize=14)
        ax2.set_title('Reconstruction Residuals', fontsize=16, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([t_np[0], t_np[-1]])

        plt.tight_layout()
        plt.savefig(f"{results_dir}/waveform_reconstruction.png", bbox_inches='tight', dpi=300)

        # Save as PDF format
        plt.savefig(f"{results_dir}/waveform_reconstruction.pdf", bbox_inches='tight', format='pdf')

        plt.close()

        print(f"Waveform reconstruction plot saved to {results_dir}/waveform_reconstruction.png")

    except Exception as e:
        print(f"Error in generate_publication_plots: {str(e)}")


def main():
    """
    主函数：运行PSO与MCMC算法，输出参数估计结果
    """
    
    print("=" * 60)
    print("引力波参数估计结果")
    print("=" * 60)

    try:
        # 1. 数据加载
        print("\n[步骤 1/4] 加载引力波数据...")
        data_dict = load_data()
        print("✓ 数据加载完成")

        # 2. 参数设置
        print("\n[步骤 2/4] 设置算法参数...")
        param_ranges, pso_params, pso_config, actual_params = setup_parameters(data_dict)
        print("✓ 参数设置完成")
        
        # 显示真实参数值
        print("\n" + "="*40)
        print("真实参数值 (参考)")
        print("="*40)
        print(f"距离:           {actual_params['source_distance']:.2f} Mpc")
        print(f"啁啾质量:       {actual_params['chirp_mass']:.2f} M☉")
        print(f"合并时间:       {actual_params['merger_time']:.4f} s")
        print(f"相位:           {actual_params['phase']:.4f} π")
        print(f"流量比:         {actual_params['flux_ratio']:.4f}")
        print(f"时间延迟:       {actual_params['time_delay']:.4f} s")

        # 3. 运行PSO算法
        print("\n[步骤 3/4] 运行PSO算法...")
        pso_results = run_pso(data_dict, pso_params, pso_config, actual_params,
                              n_runs=1, enable_distance_refinement=True)
        print("✓ PSO算法完成")

        # 4. 运行MCMC算法
        print("\n[步骤 4/4] 运行MCMC算法...")
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=200, n_iter=500,
                                      enable_distance_refinement=True, actual_params=actual_params)
        print("✓ MCMC算法完成")

        # 输出PSO参数结果
        print("\n" + "="*50)
        print("PSO 参数估计结果")
        print("="*50)
        
        pso_params_result = pso_results['best_params']
        
        # 转换为物理单位
        pso_distance = 10**pso_params_result['r']
        pso_chirp_mass = 10**pso_params_result['m_c']
        pso_merger_time = pso_params_result['tc']
        pso_phase = pso_params_result['phi_c']
        pso_flux_ratio = pso_params_result['A']
        pso_time_delay = pso_params_result['delta_t']
        
        print(f"距离:           {pso_distance:.2f} Mpc")
        print(f"啁啾质量:       {pso_chirp_mass:.2f} M☉")
        print(f"合并时间:       {pso_merger_time:.4f} s")
        print(f"相位:           {pso_phase:.4f} π")
        print(f"流量比:         {pso_flux_ratio:.4f}")
        print(f"时间延迟:       {pso_time_delay:.4f} s")
        print(f"执行时间:       {pso_results['duration']:.2f} 秒")
        print(f"信噪比:         {pso_results['snr']:.4f}")

        # 输出MCMC参数结果
        print("\n" + "="*50)
        print("MCMC 参数估计结果")
        print("="*50)
        
        mcmc_params_result = mcmc_results['best_params']
        
        # 转换为物理单位
        mcmc_distance = 10**mcmc_params_result['r']
        mcmc_chirp_mass = 10**mcmc_params_result['m_c']
        mcmc_merger_time = mcmc_params_result['tc']
        mcmc_phase = mcmc_params_result['phi_c']
        mcmc_flux_ratio = mcmc_params_result['A']
        mcmc_time_delay = mcmc_params_result['delta_t']
        
        print(f"距离:           {mcmc_distance:.2f} Mpc")
        print(f"啁啾质量:       {mcmc_chirp_mass:.2f} M☉")
        print(f"合并时间:       {mcmc_merger_time:.4f} s")
        print(f"相位:           {mcmc_phase:.4f} π")
        print(f"流量比:         {mcmc_flux_ratio:.4f}")
        print(f"时间延迟:       {mcmc_time_delay:.4f} s")
        print(f"执行时间:       {mcmc_results['duration']:.2f} 秒")
        print(f"信噪比:         {mcmc_results['snr']:.4f}")

        # 性能比较
        comparison = evaluate_results(pso_results, mcmc_results, actual_params, data_dict)

        # 创建参数对比表（移除Refinement Status列）
        param_df = create_parameter_comparison_table(pso_results, mcmc_results, actual_params, param_ranges)

        # 创建性能对比表（新增）
        performance_df = create_performance_comparison_table(pso_results, mcmc_results, comparison, results_dir)

        # 生成MCMC后验分析图（包含真实值）- 根据透镜状态调整参数
        if mcmc_results['posterior_samples'] is not None:
            create_mcmc_corner_plot_with_true_values(mcmc_results, param_ranges, actual_params, results_dir)

        # 生成波形重建图（仅保留此图）
        generate_publication_plots(pso_results, mcmc_results, data_dict, comparison, actual_params)

        # 保存简化的结果
        print("\n保存结果文件...")
        
        # 创建简化的结果汇总
        results_summary = {
            'True_Parameters': {
                'Distance_Mpc': actual_params['source_distance'],
                'Chirp_Mass_Msun': actual_params['chirp_mass'],
                'Merger_Time_s': actual_params['merger_time'],
                'Phase_pi': actual_params['phase'],
                'Flux_Ratio': actual_params['flux_ratio'],
                'Time_Delay_s': actual_params['time_delay']
            },
            'PSO_Results': {
                'Distance_Mpc': pso_distance,
                'Chirp_Mass_Msun': pso_chirp_mass,
                'Merger_Time_s': pso_merger_time,
                'Phase_pi': pso_phase,
                'Flux_Ratio': pso_flux_ratio,
                'Time_Delay_s': pso_time_delay,
                'Duration_seconds': pso_results['duration'],
                'SNR': pso_results['snr']
            },
            'MCMC_Results': {
                'Distance_Mpc': mcmc_distance,
                'Chirp_Mass_Msun': mcmc_chirp_mass,
                'Merger_Time_s': mcmc_merger_time,
                'Phase_pi': mcmc_phase,
                'Flux_Ratio': mcmc_flux_ratio,
                'Time_Delay_s': mcmc_time_delay,
                'Duration_seconds': mcmc_results['duration'],
                'SNR': mcmc_results['snr']
            }
        }
        
        # 保存到JSON文件
        import json
        with open(f"{results_dir}/results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("✓ 结果已保存到 L_Result_circulate/results_summary.json")
        
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n✅ 参数估计完成!")
    return True


if __name__ == "__main__":
    main()