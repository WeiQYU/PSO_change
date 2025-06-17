import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
import pandas as pd
import scipy.constants as const
from tqdm import tqdm
import os
import warnings

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
results_dir = "L_Result_py"
os.makedirs(results_dir, exist_ok=True)

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
        'phase': 0.25
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

            # Determine lensing usage
            use_lensing = A >= 0.01

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


def run_bilby_mcmc(data_dict, param_ranges, n_live_points, n_iter, enable_distance_refinement=True):
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
            walks=25,
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
                        param_ranges
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
            use_lensing = A >= 0.01
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
            is_lensed = A >= 0.01
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
            'phi_c': abs(abs(float(pso_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (2 * np.pi),
            'A': abs((float(pso_results['best_params']['A']) - actual_params['flux_ratio']) / actual_params['flux_ratio']),
            'delta_t': abs((float(pso_results['best_params']['delta_t']) - actual_params['time_delay']) / actual_params['time_delay'])
        }

        # Calculate parameter errors for MCMC
        mcmc_errors = {
            'r': abs((10 ** float(mcmc_results['best_params']['r']) - actual_params['source_distance']) / actual_params['source_distance']),
            'm_c': abs((10 ** float(mcmc_results['best_params']['m_c']) - actual_params['chirp_mass']) / actual_params['chirp_mass']),
            'tc': abs((float(mcmc_results['best_params']['tc']) - actual_params['merger_time']) / actual_params['merger_time']),
            'phi_c': abs(abs(float(mcmc_results['best_params']['phi_c']) - actual_params['phase'] * np.pi)) / (2 * np.pi),
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
    """Create a comprehensive parameter comparison table"""
    print("\nCreating parameter comparison table...")

    try:
        # Parameter names and their descriptions
        param_info = {
            'r': {'name': 'Distance', 'unit': 'Mpc', 'transform': lambda x: 10 ** x},
            'm_c': {'name': 'Chirp Mass', 'unit': 'M☉', 'transform': lambda x: 10 ** x},
            'tc': {'name': 'Merger Time', 'unit': 's', 'transform': lambda x: x},
            'phi_c': {'name': 'Phase', 'unit': 'rad', 'transform': lambda x: x / np.pi},
            'A': {'name': 'Flux Ratio', 'unit': '', 'transform': lambda x: x},
            'delta_t': {'name': 'Time Delay', 'unit': 's', 'transform': lambda x: x}
        }

        # Actual parameter values
        actual_values = {
            'r': actual_params['source_distance'],
            'm_c': actual_params['chirp_mass'],
            'tc': actual_params['merger_time'],
            'phi_c': actual_params['phase'] * np.pi,
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

            # Add distance refinement info for distance parameter
            refinement_note = ""
            if param == 'r':
                pso_dist_ref = pso_results.get('distance_refinement', {})
                mcmc_dist_ref = mcmc_results.get('distance_refinement', {})

                pso_refined = ""
                mcmc_refined = ""

                if pso_dist_ref.get('enabled', False) and 'refinement_info' in pso_dist_ref:
                    if pso_dist_ref['refinement_info']['status'] == 'success':
                        confidence = pso_dist_ref['refinement_info'].get('confidence', 0)
                        pso_refined = f" (Refined, conf: {confidence:.2f})"

                if mcmc_dist_ref.get('enabled', False) and 'refinement_info' in mcmc_dist_ref:
                    if mcmc_dist_ref['refinement_info']['status'] == 'success':
                        confidence = mcmc_dist_ref['refinement_info'].get('confidence', 0)
                        mcmc_refined = f" (Refined, conf: {confidence:.2f})"

                refinement_note = f"PSO{pso_refined}, MCMC{mcmc_refined}"

            # Add to table
            table_data.append({
                'Parameter': param_info[param]['name'],
                'Unit': param_info[param]['unit'],
                'True Value': f"{actual_value:.4f}",
                'PSO Estimate': f"{pso_value:.4f}",
                'PSO Error (%)': f"{pso_error:.2f}",
                'MCMC Estimate': f"{mcmc_value:.4f}",
                'MCMC Error (%)': f"{mcmc_error:.2f}",
                'Refinement Status': refinement_note
            })

        # Create DataFrame and save as CSV
        param_df = pd.DataFrame(table_data)
        param_df.to_csv(f"{results_dir}/parameter_comparison_table.csv", index=False)

        print(f"Parameter comparison table saved to {results_dir}/parameter_comparison_table.csv")

        return param_df

    except Exception as e:
        print(f"Error creating parameter comparison table: {str(e)}")
        return pd.DataFrame()


def plot_mcmc_posterior_analysis(mcmc_results, param_ranges, actual_params):
    """Create comprehensive MCMC posterior analysis plots"""
    print("\nGenerating MCMC posterior analysis plots...")

    try:
        if mcmc_results['posterior_samples'] is None or len(mcmc_results['posterior_samples']) == 0:
            print("No MCMC posterior samples available for plotting")
            return

        posterior = mcmc_results['posterior_samples']
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
        param_labels = ['Distance\n[log10(Mpc)]', 'Chirp Mass\n[log10(M☉)]', 'Merger Time\n[s]',
                        'Phase\n[rad]', 'Flux Ratio\n[A]', 'Time Delay\n[s]']

        # Convert to physical units
        physical_samples = np.zeros_like(posterior[param_names].values)
        actual_physical = np.zeros(6)

        for i, param in enumerate(param_names):
            # Unscale from [0,1] to original parameter range
            unscaled = posterior[param].values * (param_ranges['rmax'][i] - param_ranges['rmin'][i]) + \
                       param_ranges['rmin'][i]

            # Transform to physical units for certain parameters
            if param == 'r':
                physical_samples[:, i] = 10 ** unscaled
                actual_physical[i] = actual_params['source_distance']
            elif param == 'm_c':
                physical_samples[:, i] = 10 ** unscaled
                actual_physical[i] = actual_params['chirp_mass']
            elif param == 'phi_c':
                physical_samples[:, i] = unscaled / np.pi
                actual_physical[i] = actual_params['phase']
            else:
                physical_samples[:, i] = unscaled
                if param == 'tc':
                    actual_physical[i] = actual_params['merger_time']
                elif param == 'A':
                    actual_physical[i] = actual_params['flux_ratio']
                elif param == 'delta_t':
                    actual_physical[i] = actual_params['time_delay']

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # Corner plot (posterior distributions)
        gs = fig.add_gridspec(6, 6, hspace=0.3, wspace=0.3)

        for i in range(6):
            for j in range(6):
                if i > j:
                    # 2D scatter plot
                    ax = fig.add_subplot(gs[i, j])

                    # Sample subset for faster plotting
                    n_plot = min(1000, len(physical_samples))
                    idx = np.random.choice(len(physical_samples), n_plot, replace=False)

                    scatter = ax.scatter(physical_samples[idx, j], physical_samples[idx, i],
                                         c=posterior['log_likelihood'].values[idx],
                                         alpha=0.6, s=1, cmap='viridis')

                    # Add true value cross
                    ax.axvline(actual_physical[j], color='red', linestyle='--', alpha=0.8, linewidth=2, label='True')
                    ax.axhline(actual_physical[i], color='red', linestyle='--', alpha=0.8, linewidth=2)

                    ax.set_xlabel(param_labels[j] if i == 5 else '', fontsize=10)
                    ax.set_ylabel(param_labels[i] if j == 0 else '', fontsize=10)
                    ax.tick_params(labelsize=8)

                elif i == j:
                    # 1D histogram
                    ax = fig.add_subplot(gs[i, j])

                    ax.hist(physical_samples[:, i], bins=50, alpha=0.7, density=True, color='steelblue',
                            edgecolor='black')
                    ax.axvline(actual_physical[i], color='red', linestyle='--', linewidth=2, label='True')

                    # Add statistics
                    mean_val = np.mean(physical_samples[:, i])
                    std_val = np.std(physical_samples[:, i])
                    ax.axvline(mean_val, color='blue', linestyle='-', linewidth=2, alpha=0.8, label='Mean')

                    ax.set_xlabel(param_labels[i], fontsize=10)
                    ax.set_ylabel('Density', fontsize=10)
                    ax.tick_params(labelsize=8)

                    # Add text with statistics
                    ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                            transform=ax.transAxes, verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    if i == 0:
                        ax.legend(fontsize=8)
                else:
                    # Empty upper triangle
                    ax = fig.add_subplot(gs[i, j])
                    ax.axis('off')

        plt.suptitle('MCMC Posterior Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(f"{results_dir}/mcmc_posterior_corner_plot.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Trace plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, (param, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]

            # Plot trace
            ax.plot(physical_samples[:, i], alpha=0.7, linewidth=0.5, color='steelblue')
            ax.axhline(actual_physical[i], color='red', linestyle='--', linewidth=2, alpha=0.8, label='True Value')

            # Add rolling mean
            window = min(100, len(physical_samples) // 10)
            if window > 1:
                rolling_mean = pd.Series(physical_samples[:, i]).rolling(window=window).mean()
                ax.plot(rolling_mean, color='orange', linewidth=2, alpha=0.8, label='Rolling Mean')

            ax.set_xlabel('Sample Number', fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'Trace Plot: {param_labels[i].split()[0]}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        plt.suptitle('MCMC Parameter Trace Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/mcmc_trace_plots.png", bbox_inches='tight', dpi=300)
        plt.close()

        print(f"MCMC posterior analysis plots saved to {results_dir}:")
        print("  - mcmc_posterior_corner_plot.png")
        print("  - mcmc_trace_plots.png")

    except Exception as e:
        print(f"Error in plotting MCMC posterior analysis: {str(e)}")


def generate_publication_plots(pso_results, mcmc_results, data_dict, comparison, actual_params):
    """Generate publication-quality plots"""
    print("\nGenerating publication-quality comparison plots...")

    try:
        # Get data as numpy arrays
        t_np = np.asarray(data_dict['t'])
        dataY_np = np.asarray(data_dict['dataY'])
        dataY_only_signal_np = np.asarray(data_dict['dataY_only_signal'])

        # Figure 1: Waveform Reconstruction Comparison
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
        plt.close()

        # Figure 2: Parameter Estimation Accuracy
        param_names = ['r', 'm_c', 'tc', 'phi_c', 'A', 'delta_t']
        param_labels = ['Distance\n[Mpc]', 'Chirp Mass\n[M☉]', 'Merger Time\n[s]',
                        'Phase\n[rad]', 'Flux Ratio\n[A]', 'Time Delay\n[s]']

        pso_error_vals = [comparison['parameter_errors']['PSO'][p] * 100 for p in param_names]
        mcmc_error_vals = [comparison['parameter_errors']['MCMC'][p] * 100 for p in param_names]

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(param_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pso_error_vals, width, label='PSO',
                       color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x + width / 2, mcmc_error_vals, width, label='MCMC',
                       color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.8)

        ax.set_xlabel('Parameters', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Error [%]', fontsize=14, fontweight='bold')
        ax.set_title('Parameter Estimation Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        max_error = max(max(pso_error_vals), max(mcmc_error_vals))
        ax.set_ylim(0, min(max_error * 1.1, 100))

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.05,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.05,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/parameter_accuracy.png", bbox_inches='tight', dpi=300)
        plt.close()

        # Figure 3: Performance Metrics Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Remove spines
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Execution time comparison
        methods = ['PSO', 'MCMC']
        times = [comparison['execution_time']['PSO'], comparison['execution_time']['MCMC']]
        colors = ['#d62728', '#1f77b4']

        bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Execution Time [s]', fontsize=12, fontweight='bold')
        ax1.set_title('Computational Efficiency', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, time in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(times) * 0.02,
                     f'{time:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # SNR comparison
        snr_values = [comparison['signal_quality']['PSO']['SNR'],
                      comparison['signal_quality']['MCMC']['SNR']]
        bars = ax2.bar(methods, snr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Signal Quality (SNR)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, snr in zip(bars, snr_values):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(snr_values) * 0.02,
                     f'{snr:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Match comparison
        match_values = [comparison['signal_quality']['PSO']['match_final'],
                        comparison['signal_quality']['MCMC']['match_final']]
        bars = ax3.bar(methods, match_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax3.set_ylabel('Template Match', fontsize=12, fontweight='bold')
        ax3.set_title('Waveform Matching Accuracy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1])

        for bar, match in zip(bars, match_values):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                     f'{match:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # PSO Fitness Evolution
        if 'iteration_history' in pso_results and len(pso_results['iteration_history']) > 0:
            iterations = range(len(pso_results['iteration_history']))
            ax4.plot(iterations, pso_results['iteration_history'], 'r-', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Fitness Value', fontsize=12, fontweight='bold')
            ax4.set_title('PSO Algorithm\nFitness Evolution', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)

            if len(pso_results['iteration_history']) > 10:
                final_fitness = pso_results['iteration_history'][-1]
                ax4.text(0.95, 0.95, f'Final: {final_fitness:.2e}',
                         transform=ax4.transAxes, ha='right', va='top',
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No PSO iteration\nhistory available',
                     transform=ax4.transAxes, ha='center', va='center',
                     fontsize=12, fontweight='bold')
            ax4.set_title('PSO Algorithm\nFitness Evolution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{results_dir}/performance_metrics.png", bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Publication-quality figures saved to {results_dir}:")
        print("  - waveform_reconstruction.png")
        print("  - parameter_accuracy.png")
        print("  - performance_metrics.png")

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
        mcmc_results = run_bilby_mcmc(data_dict, param_ranges, n_live_points=250, n_iter=400,
                                      enable_distance_refinement=True)
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
        pso_phase = pso_params_result['phi_c'] / np.pi
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
        mcmc_phase = mcmc_params_result['phi_c'] / np.pi
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

        # 创建参数对比表
        param_df = create_parameter_comparison_table(pso_results, mcmc_results, actual_params, param_ranges)

        # 生成MCMC后验分析图
        if mcmc_results['posterior_samples'] is not None:
            plot_mcmc_posterior_analysis(mcmc_results, param_ranges, actual_params)

        # 生成发表质量的对比图
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
        
        # 保存到CSV文件
        import json
        with open(f"{results_dir}/results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("✓ 结果已保存到 L_Result/results_summary.json")
        
    except Exception as e:
        print(f"\n❌ 执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n✅ 参数估计完成!")
    return True


if __name__ == "__main__":
    main()